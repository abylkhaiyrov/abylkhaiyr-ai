from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from llm.evaluation.capacity_metrics import Prediction, paired_summary
from llm.evaluation.diagnostic import encode_options, score_encoded
from llm.evaluation.diagnostic_bank import ITEMS_PER_CELL, LANGUAGES, TASKS, items
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM
from llm.training.capacity_study import CONTEXT, budget_step, capacity_config, matched_step
from llm.training.session import stop_at_deadline


def test_study_matches_examples_and_keeps_floated_milestones(tmp_path: Path) -> None:
    tiny, small = (
        capacity_config(name, 2, tmp_path, tmp_path, tmp_path, tmp_path)
        for name in ("tiny", "small")
    )
    assert tiny.train.seed == small.train.seed == 2
    assert tiny.train.sequence_length == small.train.sequence_length == CONTEXT
    assert tiny.model.max_position_embeddings == small.model.max_position_embeddings == CONTEXT
    assert tiny.train.init_from is small.train.init_from is None
    assert tiny.train.mixture == small.train.mixture == {}
    assert tiny.train.optimizer == small.train.optimizer
    assert budget_step(5_000_000) == 305
    assert matched_step("small", 5_000_000) == 98
    assert small.train.protected_steps[:2] == [98, 305]


def test_diagnostic_bank_has_balanced_keys_and_unique_localized_prefixes() -> None:
    bank = list(items())
    assert len(bank) == len(LANGUAGES) * len(TASKS) * ITEMS_PER_CELL
    assert len({item.prefix for item in bank}) == len(bank)
    assert all(len(set(item.options)) == 4 for item in bank)
    cells = Counter((item.language, item.task, item.answer) for item in bank)
    assert set(cells.values()) == {8}
    assert len(cells) == len(LANGUAGES) * len(TASKS) * 4


def test_option_variants_share_retained_prefix_and_batch_padding_is_invariant(
    model_config,
    tokenizer,
) -> None:
    engine = InferenceEngine(CausalLM(model_config), tokenizer, "cpu")
    long_prefix = "a word " * model_config.max_position_embeddings
    options = encode_options(engine, long_prefix, ["yes", "a much longer option"])
    assert options[0].context == options[1].context
    assert len(options[1].context) + len(options[1].continuation) == engine.max_positions
    scores = score_encoded(engine, options)
    for item, score in zip(options, scores, strict=True):
        assert score[0] == pytest.approx(score_encoded(engine, [item])[0][0], abs=1e-5)


def test_cluster_pairing_rejects_keys_and_preserves_seed_variation() -> None:
    template = Prediction(
        "a",
        "knowledge",
        "ru",
        "family-1",
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        (),
        (),
    )
    first = [template, replace(template, id="b", language="kk")]
    winning = [replace(row, normalized=0) for row in first]
    result = paired_summary([(first, winning), (first, first)], repetitions=20)
    assert result["seed_differences"] == [1.0, 0.0]
    assert result["cluster_bootstrap_95"] == [0.5, 0.5]
    assert result["families"] == 1
    with pytest.raises(ValueError, match="identical task IDs"):
        paired_summary([(first, winning[:1])], repetitions=20)


def test_expired_deadline_requests_graceful_stop() -> None:
    event = threading.Event()
    with stop_at_deadline(event, time.monotonic() - 1):
        assert event.is_set()
    future = threading.Event()
    with stop_at_deadline(future, time.monotonic() + 60):
        assert not future.is_set()
    assert not future.is_set()
