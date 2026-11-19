from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from llm.core.errors import DataError
from llm.data.binary import read_jsonl
from llm.evaluation import suites
from llm.evaluation.metrics import (
    pass_at_k,
    perplexity_from,
    repeated_ngram_ratio,
    score_token_stream,
)
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.optimization.memory import device_memory_used_gb, ram_used_gb
from llm.reasoning.tools import answers_match, extract_final_answer, run_code_tests
from llm.tokenizer.bpe import Message
from torch import Tensor

CODE_SAMPLES = 4
CODE_MAX_TOKENS = 160
ANSWER_MAX_TOKENS = 64
GSM8K_MAX_TOKENS = 200
WINDOW_COUNT = 64
SPEED_TOKENS = 64


@dataclass
class EvaluationConfig:
    data_dir: str
    max_windows: int = WINDOW_COUNT
    arithmetic_cases: int = 30
    gsm8k_cases: int = 50
    seed: int = 0


def greedy(max_new_tokens: int, **kwargs) -> SamplingParams:
    return SamplingParams(greedy=True, max_new_tokens=max_new_tokens, **kwargs)


class Evaluator:
    def __init__(self, engine: InferenceEngine, config: EvaluationConfig) -> None:
        self.engine = engine
        self.config = config
        self.tokenizer = engine.tokenizer
        self.device = engine.device

    def _answer(self, prompt: str, max_tokens: int) -> str:
        return self.engine.chat([Message("user", prompt)], greedy(max_tokens)).text

    def language_modeling(self) -> dict[str, float]:
        path = Path(self.config.data_dir) / "tokens" / "test.bin"
        if not path.is_file():
            raise DataError(f"test tokens not found: {path}")
        tokens = np.memmap(path, dtype=np.uint16 if self.tokenizer.vocab_size <= 1 << 16 else np.uint32, mode="r")
        length = self.engine.max_positions
        count = min((len(tokens) - 1) // length, self.config.max_windows)
        windows = [torch.from_numpy(tokens[i * length : i * length + length + 1].astype(np.int64)) for i in range(count)]
        total, n, correct = score_token_stream(self.engine.model, windows, self.device)
        return {"test_loss": total / n, "test_perplexity": perplexity_from(total, n), "next_token_accuracy": correct / n}

    def multilingual_perplexity(self) -> dict[str, float]:
        results: dict[str, float] = {}
        test_file = Path(self.config.data_dir) / "corpus" / "test.jsonl"
        by_language: dict[str, list[Tensor]] = {}
        for row in read_jsonl(test_file):
            language = row.get("language")
            if language is None or len(by_language.get(language, [])) >= self.config.max_windows // 4:
                continue
            ids = [self.tokenizer.bos_id, *self.tokenizer.encode(row["text"])][: self.engine.max_positions + 1]
            if len(ids) > 8:
                by_language.setdefault(language, []).append(torch.tensor(ids))
        for language, windows in by_language.items():
            total, n, _ = score_token_stream(self.engine.model, windows, self.device)
            results[f"perplexity_{language}"] = perplexity_from(total, n)
        return results

    def arithmetic(self) -> dict[str, float]:
        cases = suites.arithmetic_cases(self.config.arithmetic_cases, self.config.seed)
        hits = sum(answers_match(extract_final_answer(self._answer(c.prompt, ANSWER_MAX_TOKENS)), c.reference) for c in cases)
        return {"arithmetic_exact_match": hits / len(cases)}

    def gsm8k(self) -> dict[str, float]:
        from datasets import load_dataset

        rows = list(load_dataset("openai/gsm8k", "main", split=f"test[:{self.config.gsm8k_cases}]"))
        hits = 0
        for row in rows:
            reference = row["answer"].split("####")[-1].strip()
            hits += answers_match(extract_final_answer(self._answer(row["question"], GSM8K_MAX_TOKENS)), reference)
        return {"gsm8k_exact_match": hits / len(rows)}

    def factual(self) -> dict[str, float]:
        answers = [(c, self._answer(c.prompt, ANSWER_MAX_TOKENS)) for c in suites.FACTUAL_CASES]
        hits = sum(c.reference in a.lower() for c, a in answers)
        return {"factual_accuracy": hits / len(answers), "factual_wrong_rate": 1 - hits / len(answers)}

    def instruction_following(self) -> dict[str, float]:
        cases = suites.instruction_cases()
        hits = sum(suites.check_constraint(c, self._answer(c.prompt, ANSWER_MAX_TOKENS)) for c in cases)
        return {"instruction_following_accuracy": hits / len(cases)}

    def code_generation(self) -> dict[str, float]:
        passes = {1: [], CODE_SAMPLES: []}
        for case in suites.CODE_CASES:
            correct = 0
            for index in range(CODE_SAMPLES):
                params = SamplingParams(max_new_tokens=CODE_MAX_TOKENS, temperature=0.8, top_p=0.95, seed=index)
                code = self.engine.chat([Message("user", case.prompt)], params).text
                correct += run_code_tests(strip_fences(code), case.tests).passed
            for k in passes:
                passes[k].append(pass_at_k(CODE_SAMPLES, correct, k))
        return {f"code_pass_at_{k}": float(np.mean(v)) for k, v in passes.items()}

    def repetition(self) -> dict[str, float]:
        ratios = []
        for prompt in suites.MULTILINGUAL_PROMPTS.values():
            ids = self.engine.generate(prompt, greedy(SPEED_TOKENS * 2)).token_ids
            ratios.append(repeated_ngram_ratio(ids))
        return {"repeated_trigram_ratio_greedy": float(np.mean(ratios))}

    def speed_and_memory(self) -> dict[str, float]:
        self.engine.generate("Warm up", greedy(8))
        started = time.perf_counter()
        generated = len(self.engine.generate("The", SamplingParams(max_new_tokens=SPEED_TOKENS, temperature=1.0, seed=0)).token_ids)
        elapsed = time.perf_counter() - started
        return {
            "generation_tokens_per_second": generated / elapsed,
            "ram_gb": ram_used_gb(),
            "device_memory_gb": device_memory_used_gb(self.device),
        }

    def run(self, include_dataset_suites: bool = True) -> dict[str, float]:
        results: dict[str, float] = {}
        stages = [
            self.language_modeling, self.multilingual_perplexity, self.arithmetic,
            self.factual, self.instruction_following, self.code_generation,
            self.repetition, self.speed_and_memory,
        ]
        if include_dataset_suites:
            stages.insert(3, self.gsm8k)
        for stage in stages:
            results.update(stage())
        return results

    def save(self, results: dict[str, float], path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(results, indent=2), encoding="utf-8")


def strip_fences(text: str) -> str:
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines)
