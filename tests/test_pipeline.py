from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm.core.config import ExperimentConfig, ModelConfig, TokenizerConfig, TrainConfig
from llm.core.errors import TrainingError
from llm.data.sft_builder import deduplicate_sft_splits
from llm.model.transformer import CausalLM
from llm.training.checkpoint import CheckpointManager, TrainingState
from llm.training.pipeline import STAGES, PipelineConfig, QualityGates, TrainingPipeline
from llm.training.release import export_release


def _write_yaml_config(path: Path, stage: str, ckpt_dir: Path, data_dir: Path) -> None:
    cfg = ExperimentConfig(
        name=f"test_{stage}",
        model=ModelConfig(
            vocab_size=256,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=64,
            max_position_embeddings=64,
        ),
        tokenizer=TokenizerConfig(vocab_size=256, path=str(data_dir)),
        train=TrainConfig(
            stage=stage,
            data_dir=str(data_dir),
            output_dir=str(ckpt_dir.parent / f"exp_{stage}"),
            checkpoint_dir=str(ckpt_dir),
            device="cpu",
            precision="fp32",
            max_steps=2,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            sequence_length=16,
            checkpoint_alias=stage,
        ),
    )
    import yaml

    path.write_text(yaml.safe_dump(cfg.model_dump()), encoding="utf-8")


def _make_fake_ckpt(directory: Path, model_config: ModelConfig, step: int) -> Path:
    model = CausalLM(model_config)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    manager = CheckpointManager(directory, keep=2, alias_prefix="test")
    return manager.save(
        model,
        model_config,
        optimizer,
        None,
        TrainingState(step=step, tokens_seen=step * 10),
        {"epoch": 0, "cursor": 0},
        "v1",
        torch.device("cpu"),
    )


def test_pipeline_runs_all_stages_and_resumes(tmp_path: Path, model_config: ModelConfig, tokenizer):
    data_dir = tmp_path / "data"
    (data_dir / "preference").mkdir(parents=True)
    (data_dir / "preference" / "train.jsonl").write_text('{"chosen":[],"rejected":[]}\n')
    tok_dir = tmp_path / "tok"
    tokenizer.save(tok_dir)
    base_ckpt = _make_fake_ckpt(tmp_path / "base", model_config, 10)
    pre_dir = tmp_path / "pre"
    sft_dir = tmp_path / "sft"
    dpo_dir = tmp_path / "dpo"
    pre_cfg = tmp_path / "pre.yaml"
    sft_cfg = tmp_path / "sft.yaml"
    dpo_cfg = tmp_path / "dpo.yaml"
    _write_yaml_config(pre_cfg, "pretrain", pre_dir, data_dir)
    _write_yaml_config(sft_cfg, "sft", sft_dir, data_dir)
    _write_yaml_config(dpo_cfg, "dpo", dpo_dir, data_dir)

    calls: list[str] = []

    def fake_train(cfg: ExperimentConfig, resume: bool) -> dict:
        calls.append(f"train:{cfg.train.stage}")
        _make_fake_ckpt(Path(cfg.train.checkpoint_dir), model_config, 100)
        return {"final_step": 100, "best_validation_loss": 2.5}

    def fake_eval(ckpt: str, tok: str, data: str, dev: str, off: bool, name: str):
        calls.append(f"eval:{name}")
        if name == "baseline_tiny_session":
            return {
                "test_perplexity": 80.0,
                "repeated_trigram_ratio_greedy": 0.2,
                "non_empty_response_rate": 1.0,
            }, []
        if name == "sft_tiny_v2":
            return {
                "test_perplexity": 50.0,
                "conv_pass_rate": 0.5,
                "repeated_trigram_ratio_greedy": 0.15,
                "non_empty_response_rate": 1.0,
                "instruction_following_accuracy": 0.5,
            }, []
        if name == "dpo_tiny_v2":
            return {
                "test_perplexity": 48.0,
                "conv_pass_rate": 0.6,
                "repeated_trigram_ratio_greedy": 0.1,
                "non_empty_response_rate": 1.0,
                "instruction_following_accuracy": 0.6,
            }, []
        return {
            "test_perplexity": 45.0,
            "conv_pass_rate": 0.2,
            "repeated_trigram_ratio_greedy": 0.15,
            "non_empty_response_rate": 1.0,
        }, []

    config = PipelineConfig(
        pretrain_config=str(pre_cfg),
        sft_config=str(sft_cfg),
        dpo_config=str(dpo_cfg),
        baseline_checkpoint=str(base_ckpt),
        tokenizer_path=str(tok_dir),
        data_dir=str(data_dir),
        state_dir=str(tmp_path / "state"),
        release_dir=str(tmp_path / "releases"),
        release_version="v0.1.0",
        device="cpu",
    )
    state = TrainingPipeline(config, train_fn=fake_train, eval_fn=fake_eval).run()
    assert all(state["stages"][stage]["status"] == "completed" for stage in STAGES)
    release_dir = Path(state["artifacts"]["release_path"])
    for filename in (
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "MODEL_CARD.md",
        "TRAINING_REPORT.md",
        "EVALUATION_REPORT.md",
        "DATASET_MANIFEST.json",
        "RUN_INSTRUCTIONS.md",
    ):
        assert (release_dir / filename).is_file()
    prev_calls = len(calls)
    TrainingPipeline(config, train_fn=fake_train, eval_fn=fake_eval).run()
    assert len(calls) == prev_calls


def test_pipeline_stops_when_pretrain_fails_quality_gate(
    tmp_path: Path, model_config: ModelConfig, tokenizer
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tok_dir = tmp_path / "tok"
    tokenizer.save(tok_dir)
    base_ckpt = _make_fake_ckpt(tmp_path / "base", model_config, 10)
    pre_cfg = tmp_path / "pre.yaml"
    sft_cfg = tmp_path / "sft.yaml"
    dpo_cfg = tmp_path / "dpo.yaml"
    _write_yaml_config(pre_cfg, "pretrain", tmp_path / "pre", data_dir)
    _write_yaml_config(sft_cfg, "sft", tmp_path / "sft", data_dir)
    _write_yaml_config(dpo_cfg, "dpo", tmp_path / "dpo", data_dir)

    def fake_train(cfg: ExperimentConfig, resume: bool) -> dict:
        _make_fake_ckpt(Path(cfg.train.checkpoint_dir), model_config, 10)
        return {"final_step": 10}

    def fake_eval(ckpt: str, tok: str, data: str, dev: str, off: bool, name: str):
        ppl = 50.0 if name == "baseline_tiny_session" else 90.0
        return {"test_perplexity": ppl, "repeated_trigram_ratio_greedy": 0.1}, []

    config = PipelineConfig(
        pretrain_config=str(pre_cfg),
        sft_config=str(sft_cfg),
        dpo_config=str(dpo_cfg),
        baseline_checkpoint=str(base_ckpt),
        tokenizer_path=str(tok_dir),
        data_dir=str(data_dir),
        state_dir=str(tmp_path / "state"),
        release_dir=str(tmp_path / "releases"),
        device="cpu",
        gates=QualityGates(require_pretrain_improvement=True),
    )
    pipeline = TrainingPipeline(config, train_fn=fake_train, eval_fn=fake_eval)
    with pytest.raises(TrainingError):
        pipeline.run()
    assert pipeline.state["stages"]["select_best_pretrain"]["status"] == "failed"
    assert pipeline.state["stages"]["sft"]["status"] == "pending"


def test_deduplicate_sft_splits_preserves_test_and_removes_overlaps(tmp_path: Path, tokenizer):
    sft_dir = tmp_path / "sft"
    sft_dir.mkdir(parents=True)
    test_rows = [
        {"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]}
    ]
    val_rows = [
        {
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1-alt"},
            ]
        },
        {"messages": [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]},
    ]
    train_rows = [
        {
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1-tr"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2-tr"},
            ]
        },
        {"messages": [{"role": "user", "content": "Q3"}, {"role": "assistant", "content": "A3"}]},
    ]
    for name, rows in (("test", test_rows), ("validation", val_rows), ("train", train_rows)):
        (sft_dir / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
    summary = deduplicate_sft_splits(tmp_path, tokenizer)
    assert summary == {
        "train": 1,
        "validation": 1,
        "test": 1,
        "removed_train": 2,
        "removed_validation": 1,
    }


def test_export_release_creates_bundle(tmp_path: Path, model_config: ModelConfig, tokenizer):
    tok_dir = tmp_path / "tok"
    tokenizer.save(tok_dir)
    ckpt = _make_fake_ckpt(tmp_path / "ckpt", model_config, 50)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(
        json.dumps({"sources": {"s": {"license": "MIT"}}}), encoding="utf-8"
    )
    out = export_release(
        ckpt,
        tok_dir,
        data_dir,
        tmp_path / "rel",
        "v0.1.0",
        eval_metrics={"test_perplexity": 42.0},
        baseline_metrics={"test_perplexity": 65.0},
    )
    assert (out / "model.safetensors").is_file()
    assert "v0.1.0" in (out / "MODEL_CARD.md").read_text(encoding="utf-8")


def _pipeline_fixture(tmp_path: Path, model_config: ModelConfig, tokenizer, with_preference: bool):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    if with_preference:
        (data_dir / "preference").mkdir()
        (data_dir / "preference" / "train.jsonl").write_text('{"chosen":[],"rejected":[]}\n')
    tok_dir = tmp_path / "tok"
    tokenizer.save(tok_dir)
    base_ckpt = _make_fake_ckpt(tmp_path / "base", model_config, 10)
    paths = {}
    for stage in ("pretrain", "sft", "dpo"):
        paths[stage] = tmp_path / f"{stage}.yaml"
        _write_yaml_config(paths[stage], stage, tmp_path / f"ckpt_{stage}", data_dir)
    config = PipelineConfig(
        pretrain_config=str(paths["pretrain"]),
        sft_config=str(paths["sft"]),
        dpo_config=str(paths["dpo"]),
        baseline_checkpoint=str(base_ckpt),
        tokenizer_path=str(tok_dir),
        data_dir=str(data_dir),
        state_dir=str(tmp_path / "state"),
        release_dir=str(tmp_path / "releases"),
        device="cpu",
    )
    return config, data_dir


def test_interrupted_training_is_not_reported_as_completed(
    tmp_path: Path, model_config: ModelConfig, tokenizer
):
    config, _ = _pipeline_fixture(tmp_path, model_config, tokenizer, False)

    def interrupted(cfg: ExperimentConfig, resume: bool) -> dict:
        _make_fake_ckpt(Path(cfg.train.checkpoint_dir), model_config, 10)
        return {"final_step": 5, "interrupted": True}

    pipeline = TrainingPipeline(config, train_fn=interrupted, eval_fn=lambda *a: ({}, []))
    with pytest.raises(TrainingError, match="interrupted"):
        pipeline.run()
    assert pipeline.state["stages"]["pretrain"]["status"] == "failed"


def _scripted_eval(sft_conv: float, pre_conv: float):
    def fake_eval(ckpt: str, tok: str, data: str, dev: str, off: bool, name: str):
        conv = sft_conv if name.startswith(("sft", "dpo")) else pre_conv
        ppl = 80.0 if name == "baseline_tiny_session" else 40.0
        return {
            "test_perplexity": ppl,
            "repeated_trigram_ratio_greedy": 0.1,
            "non_empty_response_rate": 1.0,
            "conv_pass_rate": conv,
        }, []

    return fake_eval


def test_dpo_is_skipped_when_sft_does_not_improve_conversations(
    tmp_path: Path, model_config: ModelConfig, tokenizer
):
    config, _ = _pipeline_fixture(tmp_path, model_config, tokenizer, True)

    def train(cfg: ExperimentConfig, resume: bool) -> dict:
        _make_fake_ckpt(Path(cfg.train.checkpoint_dir), model_config, 10)
        return {"final_step": 10}

    state = TrainingPipeline(config, train_fn=train, eval_fn=_scripted_eval(0.1, 0.2)).run()
    assert state["stages"]["dpo"]["status"] == "skipped"
    assert "conversational" in state["stages"]["dpo"]["reason"]
    assert state["artifacts"]["final_checkpoint"] == state["artifacts"]["sft_checkpoint"]


def test_dpo_stage_builds_preference_corpus_from_sft_policy(
    tmp_path: Path, model_config: ModelConfig, tokenizer
):
    config, data_dir = _pipeline_fixture(tmp_path, model_config, tokenizer, False)
    config.build_preference = True
    built: list[str] = []

    def train(cfg: ExperimentConfig, resume: bool) -> dict:
        _make_fake_ckpt(Path(cfg.train.checkpoint_dir), model_config, 10)
        return {"final_step": 10}

    def preference(cfg: ExperimentConfig, policy: str, tokenizer_path: str, device: str) -> dict:
        built.append(policy)
        target = Path(cfg.train.data_dir) / "preference"
        target.mkdir(exist_ok=True)
        (target / "train.jsonl").write_text('{"chosen":[],"rejected":[]}\n')
        return {"pairs": 1}

    pipeline = TrainingPipeline(
        config,
        train_fn=train,
        eval_fn=_scripted_eval(0.5, 0.1),
        preference_fn=preference,
    )
    state = pipeline.run()
    assert len(built) == 1 and built[0] == state["artifacts"]["sft_checkpoint"]
    assert state["artifacts"]["preference_corpus"] == {"pairs": 1}
    assert state["stages"]["dpo"]["status"] == "completed"
    assert (data_dir / "preference" / "train.jsonl").is_file()
