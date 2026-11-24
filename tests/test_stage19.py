from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from llm.evaluation.capacity_metrics import unigram_probabilities
from llm.evaluation.diagnostic_bank import Item
from llm.evaluation.diagnostic_controls import (
    METHODOLOGY,
    NEUTRAL_PREFIX,
    control_report,
    foreign_prompts,
    score_controls,
)
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM
from llm.optimization.schedule import learning_rate_at
from llm.training.spec import load_spec

SCRIPTS = Path("scripts")


def load_script(name: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def item(index: int, language: str = "en", task: str = "knowledge") -> Item:
    return Item(
        f"id-{language}-{task}-{index}",
        language,
        task,
        task,
        f"family-{index}",
        "test",
        f"Question number {index} about topic {index}?",
        ("red", "green", "blue", "yellow"),
        index % 4,
    )


def test_foreign_prompts_rotate_inside_language_and_task_cells() -> None:
    items = (
        [item(i) for i in range(3)] + [item(i, "ru") for i in range(3)] + [item(0, "en", "math")]
    )
    foreign = foreign_prompts(items)
    for entry in items:
        other = foreign[entry.id]
        same_cell = (other.language, other.task) == (entry.language, entry.task)
        singleton = entry.task == "math"
        assert same_cell and (other.id == entry.id) == singleton
    assert len({foreign[entry.id].id for entry in items if entry.task == "knowledge"}) == 6


def test_neutral_prefix_covers_every_language() -> None:
    assert set(NEUTRAL_PREFIX) == {"en", "ru", "kk"}


def test_controls_are_computed_for_each_item_and_reported(
    model_config, tokenizer, tmp_path
) -> None:
    engine = InferenceEngine(CausalLM(model_config), tokenizer, device=torch.device("cpu"))
    items = [item(i) for i in range(6)]
    tokens = tmp_path / "train.bin"
    (np.arange(4000) % (tokenizer.vocab_size - 1) + 1).astype(np.uint16).tofile(tokens)
    unigram = unigram_probabilities(tokens, tokenizer)
    rows = list(score_controls(engine, items, unigram, batch_size=4))
    assert [row.id for row in rows] == [entry.id for entry in items]
    for row in rows:
        assert len(row.lengths) == 4 and all(count > 0 for count in row.lengths)
        assert row.shortest == min(range(4), key=row.lengths.__getitem__)
        assert {row.normalized, row.total, row.calibrated, row.mismatched} <= {0, 1, 2, 3}
    report = control_report(rows, repetitions=50)
    assert report["methodology"] == METHODOLOGY == "v3"
    assert report["items"] == 6
    assert report["accuracy"]["normalized"]["n"] == 6
    assert report["question_use"]["n"] == 6


def test_stage19_arms_differ_only_in_registered_optimizer_settings() -> None:
    spec = load_spec("configs/experiments/stage19.yaml")
    reference = spec.job_configuration("L1").train
    registered = {
        "L1": (5e-5, 1.0, False),
        "L2": (5e-5, 1.0, True),
        "L3": (1e-4, 1.0, False),
        "L4": (2e-4, 0.25, False),
    }
    assert spec.target_steps == 305
    for arm, (rate, floor, restore) in registered.items():
        train = spec.job_configuration(arm).train
        assert (train.optimizer.learning_rate, train.optimizer.min_lr_ratio) == (rate, floor)
        assert train.restore_optimizer == restore
        assert train.optimizer.warmup_steps == 10
        assert (train.seed, train.data_dir, train.max_steps, train.init_from) == (
            reference.seed,
            reference.data_dir,
            reference.max_steps,
            reference.init_from,
        )
        assert train.checkpoint_dir.endswith(arm)


@pytest.mark.parametrize(("arm", "final"), [("L1", 5e-5), ("L2", 5e-5), ("L3", 1e-4), ("L4", 5e-5)])
def test_registered_schedules_end_at_the_registered_rate(arm: str, final: float) -> None:
    spec = load_spec("configs/experiments/stage19.yaml")
    optimizer = spec.job_configuration(arm).train.optimizer
    last = learning_rate_at(
        spec.target_steps,
        spec.target_steps,
        optimizer.learning_rate,
        optimizer.warmup_steps,
        optimizer.min_lr_ratio,
    )
    assert last == pytest.approx(final)


def test_corpus_audit_helpers_measure_diversity() -> None:
    module = load_script("audit_stage19_corpus")
    uniform = np.ones(8)
    assert module.entropy(uniform) == pytest.approx(3.0)
    assert module.entropy(np.asarray([5.0, 0.0])) == 0.0
    repeated = np.tile(np.arange(4), 50)
    assert module.distinct_ngrams(repeated, 4) < 0.05
    assert module.distinct_ngrams(np.arange(200), 4) == 1.0


def test_regression_audit_groups_every_parameter_name() -> None:
    module = load_script("audit_regression_stage19")
    names = {
        "embedding.weight": "embedding",
        "blocks.3.attention.q_proj.weight": "attention",
        "blocks.3.feed_forward.up_proj.weight": "ffn",
        "blocks.3.ffn_norm.weight": "norm",
        "final_norm.weight": "norm",
    }
    group: Callable[[str], str] = module.group_of
    assert {name: group(name) for name in names} == names


def moment_pair(model_config):
    from llm.core.config import TrainConfig
    from llm.training.engine import NO_DECAY_NDIM, build_optimizer

    model = CausalLM(model_config)
    parameters = list(model.parameters())
    decayed = [p for p in parameters if p.ndim > NO_DECAY_NDIM]
    plain = [p for p in parameters if p.ndim <= NO_DECAY_NDIM]
    previous = torch.optim.AdamW(
        [{"params": decayed, "weight_decay": 0.1}, {"params": plain, "weight_decay": 0.0}]
    )
    model(torch.randint(1, 100, (2, 16))).float().pow(2).mean().backward()
    previous.step()
    return model, previous, build_optimizer(model, TrainConfig())


def test_optimizer_state_saved_with_the_older_group_order_is_remapped(model_config) -> None:
    from llm.training.checkpoint import align_optimizer_state

    model, previous, current = moment_pair(model_config)
    saved = previous.state_dict()
    with pytest.raises(ValueError, match="doesn't match the size"):
        current.load_state_dict(saved)
    current.load_state_dict(align_optimizer_state(saved, current))
    for parameter in model.parameters():
        assert torch.equal(
            current.state[parameter]["exp_avg"], previous.state[parameter]["exp_avg"]
        )
        assert torch.equal(
            current.state[parameter]["exp_avg_sq"], previous.state[parameter]["exp_avg_sq"]
        )


def test_missing_hyperparameters_of_older_states_use_current_values(model_config) -> None:
    from llm.training.checkpoint import align_optimizer_state

    _, previous, current = moment_pair(model_config)
    saved = previous.state_dict()
    for group in saved["param_groups"]:
        group.pop("foreach", None)
    aligned = align_optimizer_state(saved, current)
    assert all("foreach" in group and "lr_scale" in group for group in aligned["param_groups"])
    current.load_state_dict(aligned)


def test_matching_optimizer_state_keeps_every_saved_value(model_config) -> None:
    from llm.training.checkpoint import align_optimizer_state

    _, _, current = moment_pair(model_config)
    saved = current.state_dict()
    assert align_optimizer_state(saved, current) == saved


def test_incompatible_optimizer_state_is_rejected(model_config) -> None:
    from llm.training.checkpoint import CheckpointError, align_optimizer_state

    _, previous, current = moment_pair(model_config)
    saved = previous.state_dict()
    saved["param_groups"][0] = {
        **saved["param_groups"][0],
        "params": saved["param_groups"][0]["params"][:-1],
    }
    with pytest.raises(CheckpointError):
        align_optimizer_state(saved, current)


def test_weight_decay_is_reapplied_by_parameter_rank_after_restore(model_config) -> None:
    from llm.training.engine import NO_DECAY_NDIM, apply_weight_decay

    _, _, current = moment_pair(model_config)
    for group in current.param_groups:
        group["weight_decay"] = 0.5
    apply_weight_decay(current, 0.1)
    for group in current.param_groups:
        expected = 0.1 if group["params"][0].ndim > NO_DECAY_NDIM else 0.0
        assert group["weight_decay"] == expected
