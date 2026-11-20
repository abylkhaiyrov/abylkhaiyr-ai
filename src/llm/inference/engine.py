from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from llm.inference.sampling import SamplingParams, apply_repetition_penalty, select_token
from llm.model.cache import KVCache
from llm.model.transformer import CausalLM
from llm.tokenizer.bpe import BPETokenizer, Message

REPLACEMENT_CHAR = "�"


@dataclass
class Generation:
    text: str
    token_ids: list[int]
    finish_reason: str


def partial_stop_length(text: str, stops: list[str]) -> int:
    longest = 0
    for stop in stops:
        for size in range(min(len(stop) - 1, len(text)), 0, -1):
            if text.endswith(stop[:size]):
                longest = max(longest, size)
                break
    return longest


def truncate_at_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    positions = [text.find(stop) for stop in stops if stop and stop in text]
    if not positions:
        return text, False
    return text[: min(positions)], True


class InferenceEngine:
    def __init__(self, model: CausalLM, tokenizer: BPETokenizer, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_positions = model.config.max_position_embeddings

    def fit_prompt(self, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        budget = max(self.max_positions - max_new_tokens, 1)
        if len(prompt_ids) <= budget:
            return prompt_ids
        keep_bos = prompt_ids[:1] if prompt_ids[0] == self.tokenizer.bos_id else []
        return keep_bos + prompt_ids[len(prompt_ids) - (budget - len(keep_bos)) :]

    def fit_messages(
        self, messages: list[Message], reserved_tokens: int
    ) -> tuple[list[Message], int]:
        budget = max(self.max_positions - reserved_tokens, 1)
        kept = list(messages)
        dropped = 0
        while len(self.tokenizer.encode_conversation(kept, True)) > budget:
            removable = [index for index, item in enumerate(kept) if item.role != "system"]
            if len(removable) <= 1:
                break
            kept.pop(removable[0])
            dropped += 1
            if kept[removable[0]].role == "assistant":
                kept.pop(removable[0])
                dropped += 1
        return kept, dropped

    def _generator(self, params: SamplingParams) -> torch.Generator | None:
        if params.seed is None:
            return None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(params.seed)
        return generator

    def _next_token(
        self, logits: Tensor, history: list[int], params: SamplingParams, generator
    ) -> int:
        logits = logits.float().cpu().unsqueeze(0)
        history_tensor = torch.tensor([history], dtype=torch.long)
        logits = apply_repetition_penalty(logits, history_tensor, params.repetition_penalty)
        return int(select_token(logits, params, generator).item())

    @torch.inference_mode()
    def stream_ids(self, prompt_ids: list[int], params: SamplingParams) -> Iterator[int]:
        prompt_ids = self.fit_prompt(prompt_ids, params.max_new_tokens)
        generator = self._generator(params)
        history = list(prompt_ids)
        penalty_start = 0 if params.penalize_prompt else len(prompt_ids)
        cache = self.model.new_cache() if params.use_cache else None
        pending = torch.tensor([prompt_ids], device=self.device)
        for _ in range(params.max_new_tokens):
            if len(history) >= self.max_positions:
                return
            if cache is not None:
                logits = self.model(pending, cache=cache)[0, -1]
            else:
                logits = self.model(torch.tensor([history], device=self.device))[0, -1]
            token = self._next_token(logits, history[penalty_start:], params, generator)
            history.append(token)
            yield token
            if token == self.tokenizer.eos_id:
                return
            pending = torch.tensor([[token]], device=self.device)

    def stream_text(
        self, prompt_ids: list[int], params: SamplingParams, stop_token_ids: set[int] | None = None
    ) -> Iterator[str]:
        generated: list[int] = []
        emitted = 0
        stops = stop_token_ids or set()
        for token in self.stream_ids(prompt_ids, params):
            if token == self.tokenizer.eos_id or token in stops:
                break
            generated.append(token)
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            text, stopped = truncate_at_stop(text, params.stop_sequences)
            if stopped:
                if len(text) > emitted:
                    yield text[emitted:]
                return
            if text.endswith(REPLACEMENT_CHAR):
                continue
            safe = len(text) - partial_stop_length(text, params.stop_sequences)
            if safe > emitted:
                yield text[emitted:safe]
                emitted = safe
        text, _ = truncate_at_stop(
            self.tokenizer.decode(generated, skip_special_tokens=True), params.stop_sequences
        )
        if len(text) > emitted:
            yield text[emitted:]

    def generate_ids(
        self, prompt_ids: list[int], params: SamplingParams, stop_token_ids: set[int] | None = None
    ) -> Generation:
        if params.beam_width > 1:
            return self.beam_search(prompt_ids, params)
        tokens: list[int] = []
        reason = "length"
        stops = stop_token_ids or set()
        for token in self.stream_ids(prompt_ids, params):
            if token == self.tokenizer.eos_id or token in stops:
                reason = "stop"
                break
            tokens.append(token)
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            if params.stop_sequences and truncate_at_stop(text, params.stop_sequences)[1]:
                reason = "stop"
                break
        text, _ = truncate_at_stop(
            self.tokenizer.decode(tokens, skip_special_tokens=True), params.stop_sequences
        )
        return Generation(text, tokens, reason)

    def _chat_stop_tokens(self) -> set[int]:
        return {
            self.tokenizer.bos_id,
            self.tokenizer.pad_id,
            self.tokenizer.role_ids["user"],
            self.tokenizer.role_ids["system"],
        }

    def generate(self, prompt: str, params: SamplingParams) -> Generation:
        return self.generate_ids(self.tokenizer.encode(prompt, add_bos=True), params)

    def _chat_ids(self, messages: list[Message], reserved_tokens: int) -> list[int]:
        fitted, _ = self.fit_messages(messages, reserved_tokens)
        return self.tokenizer.encode_conversation(fitted, add_generation_prompt=True)

    def chat(self, messages: list[Message], params: SamplingParams) -> Generation:
        ids = self._chat_ids(messages, params.max_new_tokens)
        return self.generate_ids(ids, params, stop_token_ids=self._chat_stop_tokens())

    def stream_chat(self, messages: list[Message], params: SamplingParams) -> Iterator[str]:
        ids = self._chat_ids(messages, params.max_new_tokens)
        return self.stream_text(ids, params, stop_token_ids=self._chat_stop_tokens())

    def stream_chat_continuation(
        self, messages: list[Message], prefix: str, params: SamplingParams
    ) -> Iterator[str]:
        prefix_ids = self.tokenizer.encode(prefix) if prefix else []
        ids = self._chat_ids(messages, params.max_new_tokens + len(prefix_ids))
        ids.extend(prefix_ids)
        return self.stream_text(ids, params, stop_token_ids=self._chat_stop_tokens())

    @torch.inference_mode()
    def generate_batch(self, prompts: list[str], params: SamplingParams) -> list[Generation]:
        encoded = [
            self.fit_prompt(self.tokenizer.encode(p, add_bos=True), params.max_new_tokens)
            for p in prompts
        ]
        width = max(len(ids) for ids in encoded)
        pad = self.tokenizer.pad_id
        input_ids = torch.tensor(
            [[pad] * (width - len(i)) + i for i in encoded], device=self.device
        )
        mask = torch.tensor(
            [[0] * (width - len(i)) + [1] * len(i) for i in encoded], device=self.device
        )
        generator = self._generator(params)
        cache = self.model.new_cache()
        histories = [list(ids) for ids in encoded]
        outputs: list[list[int]] = [[] for _ in prompts]
        finished = [False] * len(prompts)
        reasons = ["length"] * len(prompts)
        pending = input_ids
        for _ in range(params.max_new_tokens):
            if mask.shape[1] >= self.max_positions:
                break
            logits = self.model(pending, attention_mask=mask, cache=cache)[:, -1]
            next_tokens = []
            for row in range(len(prompts)):
                if finished[row]:
                    next_tokens.append(pad)
                    continue
                scope = histories[row] if params.penalize_prompt else outputs[row]
                token = self._next_token(logits[row], scope, params, generator)
                histories[row].append(token)
                if token == self.tokenizer.eos_id:
                    finished[row], reasons[row] = True, "stop"
                    next_tokens.append(pad)
                    continue
                outputs[row].append(token)
                text = self.tokenizer.decode(outputs[row])
                if params.stop_sequences and truncate_at_stop(text, params.stop_sequences)[1]:
                    finished[row], reasons[row] = True, "stop"
                next_tokens.append(token)
            if all(finished):
                break
            pending = torch.tensor([[t] for t in next_tokens], device=self.device)
            mask = torch.cat(
                (mask, torch.ones(len(prompts), 1, dtype=mask.dtype, device=self.device)), 1
            )
        return [
            Generation(truncate_at_stop(self.tokenizer.decode(o), params.stop_sequences)[0], o, r)
            for o, r in zip(outputs, reasons, strict=True)
        ]

    @torch.inference_mode()
    def beam_search(self, prompt_ids: list[int], params: SamplingParams) -> Generation:
        width = params.beam_width
        prompt_ids = self.fit_prompt(prompt_ids, params.max_new_tokens)
        cache: KVCache = self.model.new_cache()
        logits = self.model(torch.tensor([prompt_ids], device=self.device), cache=cache)[:, -1]
        cache.reorder(torch.zeros(width, dtype=torch.long, device=self.device))
        log_probs = torch.log_softmax(logits.float(), dim=-1).expand(width, -1).clone()
        scores = torch.zeros(width, device=self.device)
        scores[1:] = float("-inf")
        sequences: list[list[int]] = [[] for _ in range(width)]
        finished: list[tuple[float, list[int]]] = []
        for step in range(params.max_new_tokens):
            if len(prompt_ids) + step >= self.max_positions:
                break
            candidates = (scores.unsqueeze(1) + log_probs).view(-1)
            top_scores, top_indices = candidates.topk(2 * width)
            vocab = log_probs.shape[-1]
            next_scores: list[float] = []
            next_sources: list[int] = []
            next_tokens: list[int] = []
            for score, index in zip(top_scores.tolist(), top_indices.tolist(), strict=True):
                source, token = divmod(index, vocab)
                if token == self.tokenizer.eos_id:
                    length = len(sequences[source]) + 1
                    finished.append((score / length**params.length_penalty, sequences[source]))
                elif len(next_scores) < width:
                    next_scores.append(score)
                    next_sources.append(source)
                    next_tokens.append(token)
            if not next_scores or len(finished) >= width:
                break
            sources = torch.tensor(next_sources, device=self.device)
            cache.reorder(sources)
            sequences = [sequences[s] + [t] for s, t in zip(next_sources, next_tokens, strict=True)]
            scores = torch.tensor(next_scores, device=self.device)
            step_input = torch.tensor([[t] for t in next_tokens], device=self.device)
            log_probs = torch.log_softmax(
                self.model(step_input, cache=cache)[:, -1].float(), dim=-1
            )
        for score, sequence in zip(scores.tolist(), sequences, strict=True):
            if score != float("-inf") and sequence:
                finished.append((score / len(sequence) ** params.length_penalty, sequence))
        best = max(finished, key=lambda item: item[0])[1] if finished else []
        reason = "stop" if best and len(best) < params.max_new_tokens else "length"
        text, _ = truncate_at_stop(self.tokenizer.decode(best), params.stop_sequences)
        return Generation(text, best, reason)
