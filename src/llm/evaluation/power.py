from __future__ import annotations

import numpy as np

CONFIDENCE_TAIL = 0.025


def lower_bound(values: np.ndarray, repetitions: int, rng: np.random.Generator) -> float:
    draws = rng.integers(0, len(values), size=(repetitions, len(values)))
    return float(np.quantile(values[draws].mean(axis=1), CONFIDENCE_TAIL))


def floor_power(
    clusters: int,
    rate: float,
    floor: float,
    simulations: int = 400,
    repetitions: int = 1000,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(simulations):
        values = (rng.random(clusters) < rate).astype(float)
        hits += lower_bound(values, repetitions, rng) >= floor
    return hits / simulations


def difference_power(
    clusters: int,
    reference: float,
    candidate: float,
    simulations: int = 400,
    repetitions: int = 1000,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(simulations):
        gap = (rng.random(clusters) < candidate).astype(float) - (
            rng.random(clusters) < reference
        ).astype(float)
        hits += lower_bound(gap, repetitions, rng) > 0.0
    return hits / simulations
