from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["higher", "lower"]
Section = Literal["language", "conversation", "reasoning", "code", "multilingual", "efficiency"]
Unit = Literal["ratio", "number", "seconds", "gigabytes", "rate"]


@dataclass(frozen=True)
class Metric:
    key: str
    section: Section
    direction: Direction
    unit: Unit = "ratio"
    language: str | None = None
    tools: bool | None = None


METRICS: tuple[Metric, ...] = (
    Metric("test_perplexity", "language", "lower", "number"),
    Metric("test_loss", "language", "lower", "number"),
    Metric("next_token_accuracy", "language", "higher"),
    Metric("coherence_distinct_bigram", "language", "higher"),
    Metric("conv_wellformed_word_rate", "language", "higher"),
    Metric("conv_pass_rate", "conversation", "higher"),
    Metric("conv_check_pass_rate", "conversation", "higher"),
    Metric("semantic_relevance", "conversation", "higher"),
    Metric("instruction_following_accuracy", "conversation", "higher"),
    Metric("multiturn_dialogue_accuracy", "conversation", "higher"),
    Metric("context_retrieval_accuracy", "conversation", "higher"),
    Metric("conv_prompt_echo_rate", "conversation", "lower"),
    Metric("conv_repetition_ratio", "conversation", "lower"),
    Metric("repeated_trigram_ratio_greedy", "conversation", "lower"),
    Metric("conv_non_empty_rate", "conversation", "higher"),
    Metric("conv_completion_rate", "conversation", "higher"),
    Metric("conv_language_stability", "conversation", "higher"),
    Metric("conv_hallucination_rate", "conversation", "lower"),
    Metric("arithmetic_exact_match", "reasoning", "higher", tools=False),
    Metric("gsm8k_exact_match", "reasoning", "higher", tools=False),
    Metric("math_bench_model_accuracy", "reasoning", "higher", tools=False),
    Metric("math_bench_tool_accuracy", "reasoning", "higher", tools=True),
    Metric("logic_model_accuracy", "reasoning", "higher", tools=False),
    Metric("logic_tool_accuracy", "reasoning", "higher", tools=True),
    Metric("factual_accuracy", "reasoning", "higher"),
    Metric("factual_wrong_rate", "reasoning", "lower"),
    Metric("code_pass_at_1", "code", "higher"),
    Metric("code_pass_at_4", "code", "higher"),
    Metric("code_tool_pass_rate", "code", "higher", tools=True),
    Metric("perplexity_ru", "multilingual", "lower", "number", language="ru"),
    Metric("perplexity_en", "multilingual", "lower", "number", language="en"),
    Metric("perplexity_kk", "multilingual", "lower", "number", language="kk"),
    Metric("conv_lang_ru", "multilingual", "higher", language="ru"),
    Metric("conv_lang_en", "multilingual", "higher", language="en"),
    Metric("conv_lang_kk", "multilingual", "higher", language="kk"),
    Metric("generation_tokens_per_second", "efficiency", "higher", "rate"),
    Metric("ram_gb", "efficiency", "lower", "gigabytes"),
    Metric("device_memory_gb", "efficiency", "lower", "gigabytes"),
)

BY_KEY = {item.key: item for item in METRICS}


def metric_catalog() -> list[dict[str, object]]:
    return [
        {
            "key": item.key,
            "section": item.section,
            "direction": item.direction,
            "unit": item.unit,
            "language": item.language,
            "tools": item.tools,
        }
        for item in METRICS
    ]
