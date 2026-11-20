from __future__ import annotations

from collections.abc import Generator, Iterator
from dataclasses import dataclass, replace

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
        tail = budget - len(keep_bos)
        return keep_bos + (prompt_ids[-tail:] if tail else [])

    def fit_messages(
        self, messages: list[Message], reserved_tokens: int
    ) -> tuple[list[Message], int]:
        budget = max(self.max_positions - reserved_tokens, self._chat_context_floor())
        entries = [(message, len(self.tokenizer.encode_message(message))) for message in messages]
        size = 2 + sum(length for _, length in entries)
        system = [message for message, _ in entries[:-1] if message.role == "system"]
        if size > budget and system and budget >= 16:
            last = entries[-1][0]
            instruction = "\n".join(item.content for item in system)
            first_text = self._fit_content(instruction, min(budget // 4, 128), suffix=False)
            remaining = budget - 6 - len(self.tokenizer.encode(first_text))
            last_text = self._fit_content(last.content, remaining)
            return [Message("system", first_text), Message(last.role, last_text)], len(messages) - 2
        while size > budget and len(entries) > 1:
            first = next((i for i, (m, _) in enumerate(entries) if m.role != "system"), 0)
            next_turn = next(
                (i for i in range(first + 1, len(entries)) if entries[i][0].role == "user"),
                None,
            )
            if next_turn is None:
                break
            size -= sum(length for _, length in entries[first:next_turn])
            del entries[first:next_turn]
        while size > budget and len(entries) > 1:
            _, length = entries.pop(0)
            size -= length
        kept = [message for message, _ in entries]
        if kept and size > budget:
            message = kept[-1]
            kept[-1] = Message(message.role, self._fit_content(message.content, budget - 4))
        return kept, len(messages) - len(kept)

    def _fit_content(self, text: str, token_budget: int, *, suffix: bool = True) -> str:
        low, high = 0, len(text)
        while low < high:
            size = (low + high + 1) // 2
            fragment = text[-size:] if suffix else text[:size]
            if len(self.tokenizer.encode(fragment)) <= token_budget:
                low = size
            else:
                high = size - 1
        return (text[-low:] if suffix else text[:low]) if low else ""

    def _chat_params(self, params: SamplingParams) -> SamplingParams:
        return replace(
            params,
            max_new_tokens=min(
                params.max_new_tokens, self.max_positions - self._chat_context_floor()
            ),
        )

    def _chat_context_floor(self) -> int:
        return min(max(8, self.max_positions // 4), 128, self.max_positions - 1)

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
    ) -> Generator[str, None, str]:
        generated: list[int] = []
        emitted = 0
        stops = stop_token_ids or set()
        reason = "length"
        for token in self.stream_ids(prompt_ids, params):
            if token == self.tokenizer.eos_id or token in stops:
                reason = "stop"
                break
            generated.append(token)
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            text, stopped = truncate_at_stop(text, params.stop_sequences)
            if stopped:
                if len(text) > emitted:
                    yield text[emitted:]
                return "stop"
            if text.endswith(REPLACEMENT_CHAR):
                continue
            safe = len(text) - partial_stop_length(text, params.stop_sequences)
            if safe > emitted:
                yield text[emitted:safe]
                emitted = safe
        text, _ = truncate_at_stop(
            self.tokenizer.decode(generated, skip_special_tokens=True), params.stop_sequences
        )
        text = text.rstrip(REPLACEMENT_CHAR)
        if len(text) > emitted:
            yield text[emitted:]
        return reason

    def generate_ids(
        self, prompt_ids: list[int], params: SamplingParams, stop_token_ids: set[int] | None = None
    ) -> Generation:
        if params.beam_width > 1:
            return self.beam_search(prompt_ids, params, stop_token_ids)
        tokens: list[int] = []
        reason = "length"
        stops = stop_token_ids or set()
        for token in self.stream_ids(prompt_ids, params):
            if token == self.tokenizer.eos_id or token in stops:
                reason = "stop"
                break
            tokens.append(token)
            if (
                params.stop_sequences
                and truncate_at_stop(
                    self.tokenizer.decode(tokens, skip_special_tokens=True), params.stop_sequences
                )[1]
            ):
                reason = "stop"
                break
        text, _ = truncate_at_stop(
            self.tokenizer.decode(tokens, skip_special_tokens=True), params.stop_sequences
        )
        return Generation(text.rstrip(REPLACEMENT_CHAR), tokens, reason)

    def _chat_stop_tokens(self) -> set[int]:
        return {
            self.tokenizer.bos_id,
            self.tokenizer.pad_id,
            *self.tokenizer.role_ids.values(),
        }

    def generate(self, prompt: str, params: SamplingParams) -> Generation:
        return self.generate_ids(self.tokenizer.encode(prompt, add_bos=True), params)

    def _chat_ids(self, messages: list[Message], reserved_tokens: int) -> list[int]:
        fitted, _ = self.fit_messages(messages, reserved_tokens)
        return self.tokenizer.encode_conversation(fitted, add_generation_prompt=True)

    def chat_prompt_tokens(self, messages: list[Message], max_new_tokens: int) -> int:
        return len(self._chat_ids(messages, max_new_tokens))

    def chat(self, messages: list[Message], params: SamplingParams) -> Generation:
        params = self._chat_params(params)
        ids = self._chat_ids(messages, params.max_new_tokens)
        return self.generate_ids(ids, params, stop_token_ids=self._chat_stop_tokens())

    def stream_chat(
        self, messages: list[Message], params: SamplingParams
    ) -> Generator[str, None, str]:
        params = self._chat_params(params)
        ids = self._chat_ids(messages, params.max_new_tokens)
        return self.stream_text(ids, params, stop_token_ids=self._chat_stop_tokens())

    def stream_chat_continuation(
        self, messages: list[Message], prefix: str, params: SamplingParams
    ) -> Iterator[str]:
        params = self._chat_params(params)
        prefix_budget = self.max_positions - params.max_new_tokens - self._chat_context_floor()
        prefix_ids = self.tokenizer.encode(self._fit_content(prefix, prefix_budget))
        ids = self._chat_ids(messages, params.max_new_tokens + len(prefix_ids))
        ids.extend(prefix_ids)
        return self.stream_text(ids, params, stop_token_ids=self._chat_stop_tokens())

    @torch.inference_mode()
    def generate_batch(self, prompts: list[str], params: SamplingParams) -> list[Generation]:
        encoded = [
            self.fit_prompt(self.tokenizer.encode(p, add_bos=True), params.max_new_tokens)
            for p in prompts
        ]
        return self._generate_batch(encoded, params, set())

    def chat_batch(
        self, conversations: list[list[Message]], params: SamplingParams
    ) -> list[Generation]:
        params = self._chat_params(params)
        encoded = [self._chat_ids(messages, params.max_new_tokens) for messages in conversations]
        return self._generate_batch(encoded, params, self._chat_stop_tokens())

    @torch.inference_mode()
    def _generate_batch(
        self, encoded: list[list[int]], params: SamplingParams, stop_ids: set[int]
    ) -> list[Generation]:
        count = len(encoded)
        if not count:
            return []
        if params.beam_width > 1 or not params.use_cache:
            return [self.generate_ids(ids, params, stop_ids) for ids in encoded]
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
        outputs: list[list[int]] = [[] for _ in encoded]
        finished = [False] * count
        reasons = ["length"] * count
        pending = input_ids
        for _ in range(params.max_new_tokens):
            if mask.shape[1] >= self.max_positions:
                break
            logits = self.model(pending, attention_mask=mask, cache=cache)[:, -1].float().cpu()
            next_tokens = []
            for row in range(count):
                if finished[row]:
                    next_tokens.append(pad)
                    continue
                scope = histories[row] if params.penalize_prompt else outputs[row]
                token = self._next_token(logits[row], scope, params, generator)
                histories[row].append(token)
                if token == self.tokenizer.eos_id or token in stop_ids:
                    finished[row], reasons[row] = True, "stop"
                    next_tokens.append(pad)
                    continue
                outputs[row].append(token)
                if (
                    params.stop_sequences
                    and truncate_at_stop(
                        self.tokenizer.decode(outputs[row], skip_special_tokens=True),
                        params.stop_sequences,
                    )[1]
                ):
                    finished[row], reasons[row] = True, "stop"
                next_tokens.append(token)
            if all(finished):
                break
            pending = torch.tensor([[t] for t in next_tokens], device=self.device)
            mask = torch.cat((mask, torch.ones(count, 1, dtype=mask.dtype, device=self.device)), 1)
        return [
            Generation(
                truncate_at_stop(
                    self.tokenizer.decode(o, skip_special_tokens=True), params.stop_sequences
                )[0].rstrip(REPLACEMENT_CHAR),
                o,
                r,
            )
            for o, r in zip(outputs, reasons, strict=True)
        ]

    @torch.inference_mode()
    def beam_search(
        self, prompt_ids: list[int], params: SamplingParams, stop_ids: set[int] | None = None
    ) -> Generation:
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
                if token == self.tokenizer.eos_id or (stop_ids is not None and token in stop_ids):
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
