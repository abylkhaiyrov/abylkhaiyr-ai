from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from llm.core.errors import DataError
from llm.data.binary import read_jsonl
from llm.data.mathgen import MATH_BENCH_FILE, answer_is_correct, read_dataset
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
from llm.reasoning.engine import Budget, Problem, ReasoningEngine, classify_task
from llm.reasoning.tools import (
    answers_match,
    extract_final_answer,
    extract_text_answer,
    run_code_tests,
)
from llm.tokenizer.bpe import Message

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
    benchmark_dir: str = "data/benchmarks"
    math_bench_cases: int = 60


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
        tokens = np.memmap(
            path, dtype=np.uint16 if self.tokenizer.vocab_size <= 1 << 16 else np.uint32, mode="r"
        )
        length = self.engine.max_positions
        count = min((len(tokens) - 1) // length, self.config.max_windows)
        windows = [
            torch.from_numpy(tokens[i * length : i * length + length + 1].astype(np.int64))
            for i in range(count)
        ]
        total, n, correct = score_token_stream(self.engine.model, windows, self.device)
        return {
            "test_loss": total / n,
            "test_perplexity": perplexity_from(total, n),
            "next_token_accuracy": correct / n,
        }

    def multilingual_perplexity(self) -> dict[str, float]:
        results: dict[str, float] = {}
        test_file = Path(self.config.data_dir) / "corpus" / "test.jsonl"
        by_language: dict[str, list[Tensor]] = {}
        for row in read_jsonl(test_file):
            language = row.get("language")
            if (
                language is None
                or len(by_language.get(language, [])) >= self.config.max_windows // 4
            ):
                continue
            ids = [self.tokenizer.bos_id, *self.tokenizer.encode(row["text"])][
                : self.engine.max_positions + 1
            ]
            if len(ids) > 8:
                by_language.setdefault(language, []).append(torch.tensor(ids))
        for language, windows in by_language.items():
            total, n, _ = score_token_stream(self.engine.model, windows, self.device)
            results[f"perplexity_{language}"] = perplexity_from(total, n)
        return results

    def arithmetic(self) -> dict[str, float]:
        cases = suites.arithmetic_cases(self.config.arithmetic_cases, self.config.seed)
        hits = sum(
            answers_match(
                extract_final_answer(self._answer(c.prompt, ANSWER_MAX_TOKENS)), c.reference
            )
            for c in cases
        )
        return {"arithmetic_exact_match": hits / len(cases)}

    def gsm8k(self) -> dict[str, float]:
        from datasets import load_dataset

        rows = list(load_dataset("openai/gsm8k", "main", split=f"test[:{self.config.gsm8k_cases}]"))
        hits = 0
        for row in rows:
            reference = row["answer"].split("####")[-1].strip()
            hits += answers_match(
                extract_final_answer(self._answer(row["question"], GSM8K_MAX_TOKENS)), reference
            )
        return {"gsm8k_exact_match": hits / len(rows)}

    def math_benchmark(self) -> dict[str, float]:
        path = Path(self.config.benchmark_dir) / MATH_BENCH_FILE
        if not path.is_file():
            raise DataError(f"math benchmark not found: {path}")
        problems = read_dataset(path)[: self.config.math_bench_cases]
        reasoner = ReasoningEngine(self.engine)
        budget = Budget(
            max_candidates=3, max_retries=0, max_new_tokens_per_candidate=GSM8K_MAX_TOKENS
        )
        model_hits = tool_hits = 0
        by_level: dict[int, list[bool]] = {}
        for problem in problems:
            reply = self._answer(problem.question, GSM8K_MAX_TOKENS)
            extracted = (
                extract_text_answer(reply)
                if problem.category == "logic"
                else extract_final_answer(reply)
            )
            hit = answer_is_correct(problem, extracted)
            solution = reasoner.solve(
                Problem(problem.question, classify_task(problem.question)), budget
            )
            tool_hit = answer_is_correct(problem, solution.answer)
            model_hits, tool_hits = model_hits + hit, tool_hits + tool_hit
            by_level.setdefault(problem.level, []).append(hit)
        results = {
            "math_bench_model_accuracy": model_hits / len(problems),
            "math_bench_tool_accuracy": tool_hits / len(problems),
        }
        results.update(
            {f"math_bench_level_{k}": float(np.mean(v)) for k, v in sorted(by_level.items())}
        )
        return results

    def factual(self) -> dict[str, float]:
        answers = [(c, self._answer(c.prompt, ANSWER_MAX_TOKENS)) for c in suites.FACTUAL_CASES]
        hits = sum(c.reference in a.lower() for c, a in answers)
        return {
            "factual_accuracy": hits / len(answers),
            "factual_wrong_rate": 1 - hits / len(answers),
        }

    def instruction_following(self) -> dict[str, float]:
        cases = suites.instruction_cases()
        hits = sum(
            suites.check_constraint(c, self._answer(c.prompt, ANSWER_MAX_TOKENS)) for c in cases
        )
        return {"instruction_following_accuracy": hits / len(cases)}

    def multiturn_dialogue(self) -> dict[str, float]:
        hits = 0
        for case in suites.MULTITURN_CASES:
            history: list[Message] = []
            for user_text, assistant_text in case.turns:
                history.append(Message("user", user_text))
                history.append(Message("assistant", assistant_text))
            history.append(Message("user", case.final_prompt))
            reply = self.engine.chat(history, greedy(ANSWER_MAX_TOKENS)).text
            hits += case.expected_substring in reply.lower()
        return {"multiturn_dialogue_accuracy": hits / len(suites.MULTITURN_CASES)}

    def context_usage(self) -> dict[str, float]:
        hits = 0
        for case in suites.CONTEXT_CASES:
            reply = self._answer(case.prompt, ANSWER_MAX_TOKENS)
            hits += case.reference in reply.lower()
        return {"context_retrieval_accuracy": hits / len(suites.CONTEXT_CASES)}

    def logical_reasoning(self) -> dict[str, float]:
        reasoner = ReasoningEngine(self.engine)
        budget = Budget(
            max_candidates=3, max_retries=0, max_new_tokens_per_candidate=ANSWER_MAX_TOKENS
        )
        model_hits = tool_hits = 0
        for case in suites.LOGIC_CASES:
            raw_reply = self._answer(case.prompt, ANSWER_MAX_TOKENS)
            model_hits += case.expected.lower() in raw_reply.lower()
            solution = reasoner.solve(Problem(case.prompt, "logic"), budget)
            tool_answer = (solution.answer or solution.text).lower()
            tool_hits += case.expected.lower() in tool_answer
        total = len(suites.LOGIC_CASES)
        return {
            "logic_model_accuracy": model_hits / total,
            "logic_tool_accuracy": tool_hits / total,
        }

    def coherence(self) -> dict[str, float]:
        non_empty = 0
        distinct_ratios: list[float] = []
        for prompt in suites.MULTILINGUAL_PROMPTS.values():
            text = self.engine.generate(prompt, greedy(SPEED_TOKENS)).text.strip()
            words = text.split()
            if len(words) >= 4:
                non_empty += 1
            if len(words) >= 2:
                bigrams = list(itertools.pairwise(words))
                distinct_ratios.append(len(set(bigrams)) / len(bigrams))
            else:
                distinct_ratios.append(0.0)
        return {
            "non_empty_response_rate": non_empty / len(suites.MULTILINGUAL_PROMPTS),
            "coherence_distinct_bigram": float(np.mean(distinct_ratios)),
        }

    def code_generation(self) -> dict[str, float]:
        passes: dict[int, list[float]] = {1: [], CODE_SAMPLES: []}
        reasoner = ReasoningEngine(self.engine)
        budget = Budget(
            max_candidates=2, max_retries=1, max_new_tokens_per_candidate=CODE_MAX_TOKENS
        )
        tool_hits = 0
        for case in suites.CODE_CASES:
            correct = 0
            for index in range(CODE_SAMPLES):
                params = SamplingParams(
                    max_new_tokens=CODE_MAX_TOKENS, temperature=0.8, top_p=0.95, seed=index
                )
                code = self.engine.chat([Message("user", case.prompt)], params).text
                correct += run_code_tests(strip_fences(code), case.tests).passed
            for k, scores in passes.items():
                scores.append(pass_at_k(CODE_SAMPLES, correct, k))
            solution = reasoner.solve(Problem(case.prompt, "code", tests=case.tests), budget)
            tool_hits += int(solution.verified)
        results = {f"code_pass_at_{k}": float(np.mean(v)) for k, v in passes.items()}
        results["code_tool_pass_rate"] = tool_hits / len(suites.CODE_CASES)
        return results

    def qualitative_samples(self) -> list[dict[str, str]]:
        reasoner = ReasoningEngine(self.engine)
        budget = Budget(max_candidates=2, max_retries=0, max_new_tokens_per_candidate=96)
        prompts = [
            ("multilingual_en", suites.MULTILINGUAL_PROMPTS["en"], "general"),
            ("multilingual_ru", suites.MULTILINGUAL_PROMPTS["ru"], "general"),
            ("multilingual_kk", suites.MULTILINGUAL_PROMPTS["kk"], "general"),
            ("instruction", suites.instruction_cases()[0].prompt, "general"),
            ("math", "What is 17 * 14?", "math"),
            ("logic", suites.LOGIC_CASES[0].prompt, "logic"),
            ("context", suites.CONTEXT_CASES[0].prompt, "general"),
            ("code", suites.CODE_CASES[0].prompt, "code"),
        ]
        samples: list[dict[str, str]] = []
        for category, prompt, kind in prompts:
            without_tools = self._answer(prompt, 96)
            with_tools = without_tools
            if kind in {"math", "logic", "code"}:
                tests = suites.CODE_CASES[0].tests if kind == "code" else None
                solution = reasoner.solve(Problem(prompt, kind, tests=tests), budget)
                with_tools = solution.answer or solution.text
            samples.append(
                {
                    "category": category,
                    "prompt": prompt,
                    "without_tools": without_tools,
                    "with_tools": with_tools,
                }
            )
        return samples

    def repetition(self) -> dict[str, float]:
        ratios = []
        for prompt in suites.MULTILINGUAL_PROMPTS.values():
            ids = self.engine.generate(prompt, greedy(SPEED_TOKENS * 2)).token_ids
            ratios.append(repeated_ngram_ratio(ids))
        return {"repeated_trigram_ratio_greedy": float(np.mean(ratios))}

    def speed_and_memory(self) -> dict[str, float]:
        self.engine.generate("Warm up", greedy(8))
        started = time.perf_counter()
        generated = len(
            self.engine.generate(
                "The", SamplingParams(max_new_tokens=SPEED_TOKENS, temperature=1.0, seed=0)
            ).token_ids
        )
        elapsed = time.perf_counter() - started
        return {
            "generation_tokens_per_second": generated / elapsed,
            "ram_gb": ram_used_gb(),
            "device_memory_gb": device_memory_used_gb(self.device),
        }

    def run(self, include_dataset_suites: bool = True) -> dict[str, float]:
        results: dict[str, float] = {}
        stages = [
            self.language_modeling,
            self.multilingual_perplexity,
            self.arithmetic,
            self.math_benchmark,
            self.logical_reasoning,
            self.factual,
            self.instruction_following,
            self.multiturn_dialogue,
            self.context_usage,
            self.coherence,
            self.code_generation,
            self.repetition,
            self.speed_and_memory,
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
