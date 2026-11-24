from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from llm.data.refusals import assistant_answers, refusal_kinds, template_of

GROUP_SALT = 7919


def family_of(row: dict) -> str | None:
    for answer in assistant_answers(row):
        kinds = refusal_kinds(row, answer)
        if kinds:
            return f"{kinds[0]}:{template_of(answer)}"
    return None


def family_labels(rows: Sequence[dict]) -> list[str | None]:
    return [family_of(row) for row in rows]


def quotas(sizes: dict[str, int], others: int, cap: float) -> dict[str, int]:
    limits = dict(sizes)
    while True:
        epoch = others + sum(limits.values())
        bound = max(1, math.floor(cap * epoch))
        tightened = {name: min(size, bound) for name, size in limits.items()}
        if tightened == limits:
            return limits
        limits = tightened


class ExposureCap:
    def __init__(self, labels: Sequence[str | None], cap: float, seed: int) -> None:
        self.seed = seed
        self.others = np.asarray([i for i, label in enumerate(labels) if label is None])
        members: dict[str, list[int]] = {}
        for index, label in enumerate(labels):
            if label is not None:
                members.setdefault(label, []).append(index)
        generator = np.random.default_rng((seed, GROUP_SALT))
        self.families = {
            name: generator.permutation(np.asarray(indices))
            for name, indices in sorted(members.items())
        }
        self.quota = quotas(
            {name: len(indices) for name, indices in self.families.items()}, len(self.others), cap
        )

    @property
    def active(self) -> bool:
        return any(self.quota[name] < len(indices) for name, indices in self.families.items())

    @property
    def epoch_size(self) -> int:
        return len(self.others) + sum(self.quota.values())

    def selected(self, epoch: int) -> np.ndarray:
        picked = [self.others]
        for name, indices in self.families.items():
            take = self.quota[name]
            start = (epoch * take) % len(indices)
            picked.append(np.take(indices, np.arange(start, start + take), mode="wrap"))
        return np.concatenate(picked)

    def order(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng((self.seed, epoch))
        return rng.permutation(self.selected(epoch))


def capped_labels(prefix: Path) -> list[str | None]:
    from llm.data.binary import read_jsonl, with_extension

    return family_labels(list(read_jsonl(with_extension(prefix, ".jsonl"))))
