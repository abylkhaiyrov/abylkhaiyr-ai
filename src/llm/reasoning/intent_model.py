from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable

from pydantic import BaseModel, Field


def features(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()[:512]
    words = re.findall(r"[^\W\d_]+|\d+", normalized)
    result = {f"word:{word}" for word in words}
    for word in words:
        marked = f"^{word}$"
        for size in (3, 4):
            result.update(marked[i : i + size] for i in range(len(marked) - size + 1))
    return result


class IntentPrediction(BaseModel):
    label: str | None
    similarity: float
    margin: float


class IntentModel(BaseModel):
    weights: dict[str, dict[str, float]]
    minimum_similarity: float = Field(default=0.12, ge=0, le=1)
    minimum_margin: float = Field(default=0.025, ge=0, le=1)

    @classmethod
    def fit(cls, examples: Iterable[tuple[str, str]]) -> IntentModel:
        labels: Counter[str] = Counter()
        by_label: dict[str, Counter[str]] = defaultdict(Counter)
        frequencies: Counter[str] = Counter()
        for text, label in examples:
            units = features(text)
            if units:
                labels[label] += 1
                by_label[label].update(units)
                frequencies.update(units)
        if len(labels) < 2:
            raise ValueError("intent fitting needs at least two labels")
        total = labels.total()
        centroids = {}
        for label, counts in by_label.items():
            weighted = {
                unit: count / labels[label] * math.log1p(total / frequencies[unit])
                for unit, count in counts.items()
                if count >= 2
            }
            norm = math.sqrt(sum(value**2 for value in weighted.values()))
            if norm:
                centroids[label] = {unit: value / norm for unit, value in weighted.items()}
        return cls(weights=centroids)

    def predict(self, text: str) -> IntentPrediction:
        units = features(text)
        if not units or len(self.weights) < 2:
            return IntentPrediction(label=None, similarity=0.0, margin=0.0)
        norm = math.sqrt(len(units))
        ranked = sorted(
            (
                (sum(weights.get(unit, 0.0) for unit in units) / norm, label)
                for label, weights in self.weights.items()
            ),
            reverse=True,
        )
        similarity, label = ranked[0]
        margin = similarity - ranked[1][0]
        accepted = similarity >= self.minimum_similarity and margin >= self.minimum_margin
        return IntentPrediction(
            label=label if accepted else None, similarity=similarity, margin=margin
        )
