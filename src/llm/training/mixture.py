from __future__ import annotations

from collections.abc import Mapping

LANGUAGE_SUFFIXES = ("en", "ru", "kk")
METRIC_PREFIX = "conv_lang_"


def language_scores(metrics: Mapping[str, float]) -> dict[str, float]:
    return {
        language: float(metrics[f"{METRIC_PREFIX}{language}"])
        for language in LANGUAGE_SUFFIXES
        if f"{METRIC_PREFIX}{language}" in metrics
    }


def group_language(name: str) -> str | None:
    suffix = name.rsplit("_", 1)[-1]
    return suffix if suffix in LANGUAGE_SUFFIXES and "_" in name else None


def adjust_for_languages(
    weights: Mapping[str, float], scores: Mapping[str, float], strength: float = 1.0
) -> dict[str, float]:
    adjusted = dict(weights)
    for name, weight in weights.items():
        language = group_language(name)
        if language is not None and language in scores:
            adjusted[name] = weight * (1.0 + strength * (1.0 - scores[language]))
    return adjusted
