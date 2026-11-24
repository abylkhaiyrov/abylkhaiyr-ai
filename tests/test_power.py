from __future__ import annotations

from llm.evaluation.power import difference_power, floor_power


def test_power_to_clear_a_floor_grows_with_the_true_rate_and_cluster_count() -> None:
    low = floor_power(120, 0.30, 0.30, simulations=150, repetitions=300)
    high = floor_power(120, 0.55, 0.30, simulations=150, repetitions=300)
    small = floor_power(30, 0.45, 0.30, simulations=150, repetitions=300)
    large = floor_power(240, 0.45, 0.30, simulations=150, repetitions=300)
    assert low < 0.1 and high > 0.95
    assert large > small


def test_power_to_separate_two_rates_depends_on_their_gap() -> None:
    assert difference_power(120, 0.05, 0.40, simulations=150, repetitions=300) > 0.95
    assert difference_power(120, 0.30, 0.31, simulations=150, repetitions=300) < 0.3
