from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from llm.cli.main import main
from llm.training.checkpoint import CheckpointManager
from llm.training.spec import load_spec


def write_project(root: Path, tokenizer, model_config) -> Path:
    tokenizer.save(root / "tokenizer")
    tokens = root / "data" / "tokens"
    tokens.mkdir(parents=True)
    rng = np.random.default_rng(0)
    pattern = rng.integers(4, model_config.vocab_size, size=37)
    data = np.tile(pattern, 200)[:6000].astype(np.uint16)
    data.tofile(tokens / "train.bin")
    data.tofile(tokens / "validation.bin")
    base = {
        "name": "unit-base",
        "model": model_config.model_dump(),
        "tokenizer": {"vocab_size": model_config.vocab_size, "path": str(root / "tokenizer")},
        "train": {
            "stage": "pretrain",
            "data_dir": str(root / "data"),
            "sequence_length": 16,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "eval_interval": 2,
            "eval_batches": 2,
            "log_interval": 2,
            "checkpoint_interval": 2,
            "keep_checkpoints": 2,
            "device": "cpu",
            "precision": "fp32",
            "dynamic_batch_size": False,
            "max_ram_fraction": 1.0,
            "min_free_disk_gb": 0.0,
            "optimizer": {"learning_rate": 0.003, "warmup_steps": 1},
        },
    }
    (root / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    spec = {
        "name": "unit",
        "base_config": str(root / "base.yaml"),
        "seeds": [1],
        "arms": {"A": {}, "B": {"optimizer": {"learning_rate": 0.001}}},
        "token_budget": 256,
        "tokens_per_step": 64,
        "output_root": str(root / "out"),
        "checkpoint_root": str(root / "ckpt"),
        "resources": {
            "ram_margin_gib": 0.0,
            "disk_reserve_gib": 0.0,
            "disk_per_job_gib": 0.0,
            "disk_slack_gib": 0.0,
            "require_mps": False,
        },
        "limits": {"active_seconds": 600, "absolute_seconds": 1200},
    }
    path = root / "spec.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


def test_runner_trains_every_job_to_the_budget_and_never_repeats_finished_work(
    tmp_path, tokenizer, model_config, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    path = write_project(tmp_path, tokenizer, model_config)
    spec = load_spec(path)
    assert spec.target_steps == 4
    assert main(["experiment", "run", str(path)]) == 0
    for job in spec.job_names():
        latest = CheckpointManager(spec.checkpoint_dir(job), keep=2).latest_valid()
        assert latest is not None
        assert CheckpointManager.verify(latest)["step"] == 4
    state = json.loads(spec.state_path.read_text(encoding="utf-8"))
    assert set(state["plan"]["jobs"]) == {"A-1", "B-1"}
    ledger = json.loads(spec.ledger_path.read_text(encoding="utf-8"))
    assert ledger["active_seconds"] > 0 and ledger["idle_seconds"] == 0
    before = {
        job: (spec.checkpoint_dir(job) / "step-00000004").stat().st_mtime for job in ("A-1", "B-1")
    }
    assert main(["experiment", "run", str(path)]) == 0
    after = {
        job: (spec.checkpoint_dir(job) / "step-00000004").stat().st_mtime for job in ("A-1", "B-1")
    }
    assert before == after
    capsys.readouterr()
    assert main(["experiment", "status", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["jobs"] == {"A-1": 4, "B-1": 4}


def test_runner_rejects_a_changed_plan_instead_of_extending_the_budget(
    tmp_path, tokenizer, model_config, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    path = write_project(tmp_path, tokenizer, model_config)
    spec = load_spec(path)
    spec.state_path.parent.mkdir(parents=True)
    assert main(["experiment", "run", str(path)]) == 0
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["limits"]["active_seconds"] = 6000
    payload["token_budget"] = 320
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert main(["experiment", "run", str(path)]) != 0


def staged_spec(path: Path, stages: list[int], horizon: int) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["stopping"] = {"max_steps": horizon, "stages": stages}
    payload["train"] = {"protected_steps": stages}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_registered_stages_continue_one_trajectory_with_a_fixed_horizon(
    tmp_path, tokenizer, model_config, monkeypatch
):
    import pytest

    from llm.core.errors import ConfigError
    from llm.training import campaign

    monkeypatch.chdir(tmp_path)
    path = write_project(tmp_path, tokenizer, model_config)
    staged_spec(path, [2, 4], 6)
    spec = load_spec(path)
    assert spec.target_steps == 6 and spec.goal(4) == 4 and spec.goal() == 6
    assert all(spec.job_configuration(job).train.max_steps == 6 for job in spec.job_names())
    with pytest.raises(ConfigError, match="registered stage"):
        campaign.run_campaign(spec, path, 3)
    assert main(["experiment", "run", str(path), "--until", "2"]) == 0
    assert {campaign.completed_step(spec, job) for job in spec.job_names()} == {2}
    assert main(["experiment", "run", str(path), "--until", "4"]) == 0
    assert {campaign.completed_step(spec, job) for job in spec.job_names()} == {4}
    for job in spec.job_names():
        assert (spec.stage_checkpoint(job, 2) / "meta.json").is_file()
        assert (spec.stage_checkpoint(job, 4) / "meta.json").is_file()
    assert spec.stage_checkpoint("A-1", None) == spec.checkpoint_dir("A-1")
    assert main(["experiment", "run", str(path), "--until", "4"]) == 0
    assert {campaign.completed_step(spec, job) for job in spec.job_names()} == {4}


def test_stage_validation_rejects_unsorted_or_oversized_stages(tmp_path, tokenizer, model_config):
    import pytest

    path = write_project(tmp_path, tokenizer, model_config)
    for stages, horizon in (([4, 2], 6), ([2, 9], 6), ([0, 2], 6)):
        staged_spec(path, stages, horizon)
        with pytest.raises(Exception, match="stages"):
            load_spec(path)
