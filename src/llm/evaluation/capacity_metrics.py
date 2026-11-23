from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np

from llm.data.binary import token_dtype
from llm.evaluation.diagnostic import EncodedOption, Scored, encode_options, score_encoded
from llm.evaluation.diagnostic_bank import Item
from llm.inference.engine import InferenceEngine
from llm.tokenizer.bpe import BPETokenizer


@dataclass(frozen=True)
class Prediction:
    id: str
    task: str
    language: str
    family: str
    answer: int
    normalized: int
    total: int
    option_only: int
    unigram: int
    random: int
    shortest: int
    numeric_rank: int
    scores: tuple[float, ...]
    option_lengths: tuple[int, ...]

    def as_scored(self, method: str) -> Scored:
        return Scored(
            self.id,
            self.language,
            self.task,
            self.task,
            0.25,
            getattr(self, method) == self.answer,
            self.total == self.answer,
        )


def unigram_probabilities(path: Path, tokenizer: BPETokenizer) -> np.ndarray:
    vocab = tokenizer.vocab_size
    tokens = np.memmap(path, mode="r", dtype=token_dtype(vocab))
    counts = np.ones(vocab, dtype=np.int64)
    for start in range(0, len(tokens), 1 << 20):
        counts += np.bincount(tokens[start : start + (1 << 20)], minlength=vocab)
    return np.log(counts / counts.sum())


def winner(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def score_bank(
    engine: InferenceEngine,
    items: Sequence[Item],
    unigram: np.ndarray,
    batch_size: int = 8,
) -> Iterable[Prediction]:
    ordered = sorted(items, key=lambda item: len(engine.tokenizer.encode(item.prefix)))
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        options = [
            encode_options(engine, item.prefix, item.options, item.separator) for item in batch
        ]
        flattened = list(chain.from_iterable(options))
        conditional = score_encoded(engine, flattened)
        bos = [engine.tokenizer.bos_id]
        independent = score_encoded(
            engine, [EncodedOption(bos, option.continuation) for option in flattened]
        )
        for index, (item, group) in enumerate(zip(batch, options, strict=True)):
            offset = index * 4
            scored = conditional[offset : offset + 4]
            priors = independent[offset : offset + 4]
            lengths = tuple(count for _, count in scored)
            raw = tuple(value for value, _ in scored)
            means = [value / count for value, count in scored]
            frequencies = [float(unigram[option.continuation].mean()) for option in group]
            shortest = min(range(4), key=lengths.__getitem__)
            random_choice = random.Random(f"stage17-baseline/{item.id}").randrange(4)
            rank = (
                sorted(range(4), key=lambda j: int(item.options[j]))[1]
                if all(option.isdecimal() for option in item.options)
                else random_choice
            )
            yield Prediction(
                item.id,
                item.task,
                item.language,
                item.family,
                item.answer,
                winner(means),
                winner(raw),
                winner([value / count for value, count in priors]),
                winner(frequencies),
                random_choice,
                shortest,
                rank,
                raw,
                lengths,
            )


def paired_summary(
    pairs: Sequence[tuple[Sequence[Prediction], Sequence[Prediction]]],
    *,
    method: str = "normalized",
    repetitions: int = 10_000,
) -> dict:
    if not pairs or repetitions < 1:
        raise ValueError("paired evaluation requires seeds and bootstrap draws")
    keys: dict[str, tuple[str, str, int]] = {}
    differences: list[dict[str, int]] = []
    first_rates, second_rates = [], []
    for left, right in pairs:
        first = {row.id: row for row in left}
        second = {row.id: row for row in right}
        if (
            not first
            or first.keys() != second.keys()
            or len(first) != len(left)
            or len(second) != len(right)
        ):
            raise ValueError("paired predictions must have unique identical task IDs")
        shape = {key: (row.task, row.family, row.answer) for key, row in first.items()}
        if any(shape[key] != (row.task, row.family, row.answer) for key, row in second.items()):
            raise ValueError("paired families and answer keys differ")
        if keys and keys != shape:
            raise ValueError("seed comparisons must use the same task bank")
        keys = shape
        wins_left = {key: int(getattr(row, method) == row.answer) for key, row in first.items()}
        wins_right = {key: int(getattr(row, method) == row.answer) for key, row in second.items()}
        first_rates.append(statistics.mean(wins_left.values()))
        second_rates.append(statistics.mean(wins_right.values()))
        differences.append({key: wins_right[key] - wins_left[key] for key in keys})
    strata: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key, (task, family, _) in keys.items():
        strata[task][family].append(statistics.mean(seed[key] for seed in differences))
    rng = np.random.default_rng(717)
    sums = np.zeros(repetitions)
    denominator = np.zeros(repetitions, dtype=np.int64)
    for families in strata.values():
        members = list(families.values())
        counts = np.asarray([len(family) for family in members], dtype=np.int64)
        totals = np.asarray([sum(family) for family in members])
        draws = rng.integers(0, len(members), size=(repetitions, len(members)))
        sums += totals[draws].sum(axis=1)
        denominator += counts[draws].sum(axis=1)
    lower, upper = np.quantile(sums / denominator, [0.025, 0.975])
    deltas = [b - a for a, b in zip(first_rates, second_rates, strict=True)]
    baseline = statistics.mean(first_rates)
    estimate = statistics.mean(deltas)
    return {
        "n": len(keys),
        "seeds": len(pairs),
        "families": sum(map(len, strata.values())),
        "baseline": baseline,
        "candidate": statistics.mean(second_rates),
        "baseline_seed_sd": statistics.stdev(first_rates) if len(pairs) > 1 else None,
        "candidate_seed_sd": statistics.stdev(second_rates) if len(pairs) > 1 else None,
        "absolute_difference": estimate,
        "relative_difference": estimate / baseline if baseline else None,
        "seed_differences": deltas,
        "seed_sd": statistics.stdev(deltas) if len(deltas) > 1 else None,
        "cluster_bootstrap_95": [float(lower), float(upper)],
    }
