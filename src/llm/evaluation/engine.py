from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
import torch
from torch import Tensor

from llm.core.errors import DataError
from llm.data.binary import read_jsonl
from llm.data.mathgen import MATH_BENCH_FILE, answer_is_correct, read_dataset
from llm.evaluation import suites
from llm.evaluation.conversation import (
    WordList,
    engine_responder,
    run_conversation_benchmark,
)
from llm.evaluation.hooks import EvaluationHooks, NullHooks
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
from llm.reasoning.solvers import solve_with_tools
from llm.reasoning.tools import (
    answers_match,
    extract_final_answer,
    extract_text_answer,
    run_code_tests,
)
from llm.reasoning.triage import triage
from llm.tokenizer.bpe import Message

_T = TypeVar("_T")
CODE_SAMPLES = 4
CODE_MAX_TOKENS = 160
ANSWER_MAX_TOKENS = 64
GSM8K_MAX_TOKENS = 200
WINDOW_COUNT = 64
SPEED_TOKENS = 64
CONVERSATION_PARAMS = SamplingParams(greedy=True, max_new_tokens=96, repetition_penalty=1.1)


@dataclass
class EvaluationConfig:
    data_dir: str
    max_windows: int = WINDOW_COUNT
    arithmetic_cases: int = 30
    gsm8k_cases: int = 50
    seed: int = 0
    benchmark_dir: str = "data/benchmarks"
    math_bench_cases: int = 60
    corpus_split: Literal["test", "validation"] = "test"


BASE_SUITES = (
    "language_modeling",
    "multilingual_perplexity",
    "arithmetic",
    "math_benchmark",
    "tool_solvers",
    "reasoning_triage",
    "logical_reasoning",
    "factual",
    "instruction_following",
    "multiturn_dialogue",
    "context_usage",
    "coherence",
    "conversation",
    "code_generation",
    "repetition",
    "speed_and_memory",
)
DATASET_SUITE_POSITION = 3


def suite_count(include_dataset_suites: bool) -> int:
    return len(BASE_SUITES) + int(include_dataset_suites)


def greedy(max_new_tokens: int, **kwargs) -> SamplingParams:
    return SamplingParams(greedy=True, max_new_tokens=max_new_tokens, **kwargs)


class Evaluator:
    def __init__(
        self,
        engine: InferenceEngine,
        config: EvaluationConfig,
        hooks: EvaluationHooks | None = None,
    ) -> None:
        self.engine = engine
        self.config = config
        self.tokenizer = engine.tokenizer
        self.device = engine.device
        self.hooks: EvaluationHooks = hooks or NullHooks()

    def _answer(self, prompt: str, max_tokens: int) -> str:
        return self.engine.chat([Message("user", prompt)], greedy(max_tokens)).text

    def language_modeling(self) -> dict[str, float]:
        path = Path(self.config.data_dir) / "tokens" / f"{self.config.corpus_split}.bin"
        if not path.is_file():
            raise DataError(f"held-out tokens not found: {path}")
        tokens = np.memmap(
            path, dtype=np.uint16 if self.tokenizer.vocab_size <= 1 << 16 else np.uint32, mode="r"
        )
        length = self.engine.max_positions
        count = min((len(tokens) - 1) // length, self.config.max_windows)
        windows = [
            torch.from_numpy(tokens[i * length : i * length + length + 1].astype(np.int64))
            for i in range(count)
        ]
        self.hooks.items_planned(len(windows))
        total, n, correct = score_token_stream(
            self.engine.model, windows, self.device, self.hooks.item_done
        )
        return {
            f"{self.config.corpus_split}_loss": total / n,
            f"{self.config.corpus_split}_perplexity": perplexity_from(total, n),
            "next_token_accuracy": correct / n,
        }

    def multilingual_perplexity(self) -> dict[str, float]:
        results: dict[str, float] = {}
        test_file = Path(self.config.data_dir) / "corpus" / f"{self.config.corpus_split}.jsonl"
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
        self.hooks.items_planned(sum(len(windows) for windows in by_language.values()))
        for language, windows in by_language.items():
            total, n, _ = score_token_stream(
                self.engine.model, windows, self.device, self.hooks.item_done
            )
            results[f"perplexity_{language}"] = perplexity_from(total, n)
        return results

    def arithmetic(self) -> dict[str, float]:
        cases = suites.arithmetic_cases(self.config.arithmetic_cases, self.config.seed)
        self.hooks.items_planned(len(cases))
        hits = sum(
            self._tick(
                answers_match(
                    extract_final_answer(self._answer(c.prompt, ANSWER_MAX_TOKENS)), c.reference
                )
            )
            for c in cases
        )
        return {"arithmetic_exact_match": hits / len(cases)}

    def gsm8k(self) -> dict[str, float]:
        from datasets import load_dataset

        rows = list(load_dataset("openai/gsm8k", "main", split=f"test[:{self.config.gsm8k_cases}]"))
        hits = 0
        self.hooks.items_planned(len(rows))
        for row in rows:
            reference = row["answer"].split("####")[-1].strip()
            hits += self._tick(
                answers_match(
                    extract_final_answer(self._answer(row["question"], GSM8K_MAX_TOKENS)),
                    reference,
                )
            )
        return {"gsm8k_exact_match": hits / len(rows)}

    def math_benchmark(self) -> dict[str, float]:
        path = Path(self.config.benchmark_dir) / MATH_BENCH_FILE
        if not path.is_file():
            raise DataError(f"math benchmark not found: {path}")
        problems = read_dataset(path)[: self.config.math_bench_cases]
        verifier_only = ReasoningEngine(self.engine, use_solvers=False)
        with_tools = ReasoningEngine(self.engine)
        budget = Budget(
            max_candidates=3, max_retries=0, max_new_tokens_per_candidate=GSM8K_MAX_TOKENS
        )
        model_hits = verifier_hits = tool_hits = 0
        by_level: dict[int, list[bool]] = {}
        self.hooks.items_planned(len(problems))
        for problem in problems:
            reply = self._answer(problem.question, GSM8K_MAX_TOKENS)
            extracted = (
                extract_text_answer(reply)
                if problem.category == "logic"
                else extract_final_answer(reply)
            )
            hit = answer_is_correct(problem, extracted)
            task = Problem(problem.question, classify_task(problem.question))
            verifier_hit = answer_is_correct(problem, verifier_only.solve(task, budget).answer)
            tool_hit = answer_is_correct(problem, with_tools.solve(task, budget).answer)
            model_hits += hit
            verifier_hits += verifier_hit
            tool_hits += tool_hit
            by_level.setdefault(problem.level, []).append(hit)
            self.hooks.item_done()
        results = {
            "math_bench_model_accuracy": model_hits / len(problems),
            "math_bench_verifier_accuracy": verifier_hits / len(problems),
            "math_bench_tool_accuracy": tool_hits / len(problems),
        }
        results.update(
            {f"math_bench_level_{k}": float(np.mean(v)) for k, v in sorted(by_level.items())}
        )
        return results

    def tool_solvers(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, cases in (
            ("dev", suites.TOOL_CASES),
            ("holdout", suites.TOOL_HOLDOUT_CASES),
        ):
            answered = correct = 0
            for case in cases:
                solution = solve_with_tools(case.prompt)
                answered += solution is not None
                correct += solution is not None and (
                    answers_match(solution.answer, case.reference)
                    or solution.answer.strip().lower() == case.reference.lower()
                )
            metrics[f"tool_{name}_accuracy"] = correct / len(cases)
            metrics[f"tool_{name}_coverage"] = answered / len(cases)
        return metrics

    def reasoning_triage(self) -> dict[str, float]:
        cases = suites.REASONING_HOLDOUT
        results = [(case, triage(case.prompt)) for case in cases]
        solved = [(case, result) for case, result in results if case.status == "solved"]
        unsolvable = [(case, result) for case, result in results if case.status != "solved"]
        status_hits = sum(result.status == case.status for case, result in results)
        answer_hits = sum((result.answer or "").strip() == case.answer for case, result in solved)
        false_solves = sum(result.status == "solved" for case, result in unsolvable)
        return {
            "triage_holdout_status_accuracy": status_hits / len(cases),
            "triage_holdout_answer_accuracy": answer_hits / len(solved),
            "triage_holdout_false_solve_rate": false_solves / len(unsolvable),
        }

    def factual(self) -> dict[str, float]:
        self.hooks.items_planned(len(suites.FACTUAL_CASES))
        answers = [
            (c, self._tick(self._answer(c.prompt, ANSWER_MAX_TOKENS))) for c in suites.FACTUAL_CASES
        ]
        hits = sum(c.reference in a.lower() for c, a in answers)
        return {
            "factual_accuracy": hits / len(answers),
            "factual_wrong_rate": 1 - hits / len(answers),
        }

    def instruction_following(self) -> dict[str, float]:
        cases = suites.instruction_cases()
        self.hooks.items_planned(len(cases))
        hits = sum(
            self._tick(suites.check_constraint(c, self._answer(c.prompt, ANSWER_MAX_TOKENS)))
            for c in cases
        )
        return {"instruction_following_accuracy": hits / len(cases)}

    def multiturn_dialogue(self) -> dict[str, float]:
        hits = 0
        self.hooks.items_planned(len(suites.MULTITURN_CASES))
        for case in suites.MULTITURN_CASES:
            history: list[Message] = []
            for user_text, assistant_text in case.turns:
                history.append(Message("user", user_text))
                history.append(Message("assistant", assistant_text))
            history.append(Message("user", case.final_prompt))
            reply = self.engine.chat(history, greedy(ANSWER_MAX_TOKENS)).text
            hits += self._tick(case.expected_substring in reply.lower())
        return {"multiturn_dialogue_accuracy": hits / len(suites.MULTITURN_CASES)}

    def context_usage(self) -> dict[str, float]:
        hits = 0
        self.hooks.items_planned(len(suites.CONTEXT_CASES))
        for case in suites.CONTEXT_CASES:
            reply = self._answer(case.prompt, ANSWER_MAX_TOKENS)
            hits += self._tick(case.reference in reply.lower())
        return {"context_retrieval_accuracy": hits / len(suites.CONTEXT_CASES)}

    def logical_reasoning(self) -> dict[str, float]:
        reasoner = ReasoningEngine(self.engine)
        budget = Budget(
            max_candidates=3, max_retries=0, max_new_tokens_per_candidate=ANSWER_MAX_TOKENS
        )
        model_hits = tool_hits = 0
        self.hooks.items_planned(len(suites.LOGIC_CASES))
        for case in suites.LOGIC_CASES:
            raw_reply = self._answer(case.prompt, ANSWER_MAX_TOKENS)
            model_hits += case.expected.lower() in raw_reply.lower()
            solution = reasoner.solve(Problem(case.prompt, "logic"), budget)
            tool_answer = (solution.answer or solution.text).lower()
            tool_hits += case.expected.lower() in tool_answer
            self.hooks.item_done()
        total = len(suites.LOGIC_CASES)
        return {
            "logic_model_accuracy": model_hits / total,
            "logic_tool_accuracy": tool_hits / total,
        }

    def coherence(self) -> dict[str, float]:
        non_empty = 0
        distinct_ratios: list[float] = []
        self.hooks.items_planned(len(suites.MULTILINGUAL_PROMPTS))
        for prompt in suites.MULTILINGUAL_PROMPTS.values():
            self.hooks.item_done()
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

    def conversation(self) -> dict[str, float]:
        corpus = Path(self.config.data_dir) / "corpus" / "validation.jsonl"
        words = WordList.from_jsonl(corpus) if corpus.is_file() else None
        report = run_conversation_benchmark(
            engine_responder(self.engine, CONVERSATION_PARAMS), words
        )
        self.conversation_samples = report.samples()
        return report.metrics

    def code_generation(self) -> dict[str, float]:
        passes: dict[int, list[float]] = {1: [], CODE_SAMPLES: []}
        reasoner = ReasoningEngine(self.engine)
        budget = Budget(
            max_candidates=2, max_retries=1, max_new_tokens_per_candidate=CODE_MAX_TOKENS
        )
        tool_hits = 0
        self.hooks.items_planned(len(suites.CODE_CASES))
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
            self.hooks.item_done()
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
        self.hooks.items_planned(len(suites.MULTILINGUAL_PROMPTS))
        for prompt in suites.MULTILINGUAL_PROMPTS.values():
            self.hooks.item_done()
            ids = self.engine.generate(prompt, greedy(SPEED_TOKENS * 2)).token_ids
            ratios.append(repeated_ngram_ratio(ids))
        return {"repeated_trigram_ratio_greedy": float(np.mean(ratios))}

    def speed_and_memory(self) -> dict[str, float]:
        self.hooks.items_planned(1)
        self.engine.generate("Warm up", greedy(8))
        started = time.perf_counter()
        generated = len(
            self.engine.generate(
                "The", SamplingParams(max_new_tokens=SPEED_TOKENS, temperature=1.0, seed=0)
            ).token_ids
        )
        elapsed = time.perf_counter() - started
        self.hooks.item_done()
        return {
            "generation_tokens_per_second": generated / elapsed,
            "ram_gb": ram_used_gb(),
            "device_memory_gb": device_memory_used_gb(self.device),
        }

    def suites_for_run(self, include_dataset_suites: bool = True) -> list:
        names = list(BASE_SUITES)
        if include_dataset_suites:
            names.insert(DATASET_SUITE_POSITION, "gsm8k")
        return [getattr(self, name) for name in names]

    def run(self, include_dataset_suites: bool = True) -> dict[str, float]:
        results: dict[str, float] = {}
        stages = self.suites_for_run(include_dataset_suites)
        for index, stage in enumerate(stages):
            self.hooks.suite_started(stage.__name__, index, len(stages))
            started = time.monotonic()
            metrics = stage()
            results.update(metrics)
            self.hooks.suite_finished(stage.__name__, metrics, time.monotonic() - started)
        return results

    def _tick(self, value: _T) -> _T:
        self.hooks.item_done()
        return value

    def save(self, results: dict[str, float], path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(results, indent=2), encoding="utf-8")


def strip_fences(text: str) -> str:
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines)
