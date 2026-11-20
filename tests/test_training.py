import json

import numpy as np
import pytest
import torch

from llm.core.config import OptimizerConfig, TrainConfig
from llm.core.errors import CheckpointError, ResourceError
from llm.data.binary import PackedTokenLoader
from llm.model.transformer import CausalLM
from llm.optimization.memory import is_out_of_memory
from llm.training.checkpoint import CheckpointManager, load_model_for_inference
from llm.training.engine import Trainer, seed_everything
from llm.training.monitor import TrainingMonitor

CPU = torch.device("cpu")


def write_tokens(path, vocab, size=6000):
    rng = np.random.default_rng(0)
    pattern = rng.integers(4, vocab, size=37)
    np.tile(pattern, size // 37 + 1)[:size].astype(np.uint16).tofile(path)


def build_trainer(tmp_path, model_config, steps, accumulation=2, micro=4, name="run", **extra):
    # Windows cannot overwrite a token file while another loader has it memory-mapped.
    if not (tmp_path / "train.bin").exists():
        write_tokens(tmp_path / "train.bin", model_config.vocab_size)
    train = PackedTokenLoader(tmp_path / "train.bin", np.dtype(np.uint16), 16, micro, seed=0)
    valid = PackedTokenLoader(
        tmp_path / "train.bin", np.dtype(np.uint16), 16, micro, 0, shuffle=False
    )
    extra.setdefault("max_ram_fraction", 1.0)
    config = TrainConfig(
        max_steps=steps,
        micro_batch_size=micro,
        gradient_accumulation_steps=accumulation,
        sequence_length=16,
        eval_interval=5,
        eval_batches=2,
        log_interval=5,
        checkpoint_interval=5,
        keep_checkpoints=2,
        optimizer=OptimizerConfig(learning_rate=3e-3, warmup_steps=2),
        **extra,
    )
    seed_everything(0)
    model = CausalLM(model_config)
    checkpoints = CheckpointManager(tmp_path / f"ckpt_{name}", keep=2)
    monitor = TrainingMonitor(tmp_path / f"exp_{name}", CPU, steps)
    trainer = Trainer(
        model, model_config, config, train, valid, CPU, torch.float32, checkpoints, monitor, "v1"
    )
    return trainer


def test_loss_decreases_and_weights_update(tmp_path, model_config):
    trainer = build_trainer(tmp_path, model_config, steps=40)
    before = [p.detach().clone() for p in trainer.model.parameters()]
    initial = trainer.evaluate()
    result = trainer.train()
    assert result.last_validation_loss < initial * 0.7
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters()))
    records = [
        json.loads(line)
        for line in (tmp_path / "exp_run" / "metrics.jsonl").read_text().splitlines()
    ]
    assert {"train/loss", "train/grad_norm", "perf/tokens_per_second"} <= {
        k for r in records for k in r
    }


def test_resume_reproduces_uninterrupted_training(tmp_path, model_config):
    straight = build_trainer(tmp_path, model_config, steps=10, name="a")
    straight.train()
    first = build_trainer(tmp_path, model_config, steps=10, name="b")
    first.train(stop_step=5)
    second = build_trainer(tmp_path, model_config, steps=10, name="b")
    assert second.resume() and second.state.step == 5
    second.train()
    for a, b in zip(straight.model.parameters(), second.model.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6)
    assert second.state.tokens_seen == straight.state.tokens_seen


def test_gradient_accumulation_equals_large_batch(tmp_path, model_config):
    small = build_trainer(tmp_path, model_config, steps=1, accumulation=4, micro=2, name="s")
    large = build_trainer(tmp_path, model_config, steps=1, accumulation=1, micro=8, name="l")
    large.model.load_state_dict(small.model.state_dict())
    for t in (small, large):
        t.model.train()
        t._set_learning_rate()
        t._train_step()
    for a, b in zip(small.model.parameters(), large.model.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-5)


def test_checkpoint_rotation_and_corruption_fallback(tmp_path, model_config):
    trainer = build_trainer(tmp_path, model_config, steps=15)
    trainer.train()
    manager = trainer.checkpoints
    assert len(manager.list_steps()) == 2
    newest = manager.latest_valid()
    (newest / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(CheckpointError):
        manager.verify(newest)
    fallback = manager.latest_valid()
    assert fallback is not None and fallback != newest


def test_checkpoint_loads_for_inference(tmp_path, model_config):
    trainer = build_trainer(tmp_path, model_config, steps=5)
    trainer.train()
    model, config, meta = load_model_for_inference(trainer.checkpoints.latest_valid())
    assert config == model_config and meta["tokenizer_version"] == "v1"
    ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        assert torch.allclose(model.eval()(ids), trainer.model.eval()(ids), atol=1e-5)


def test_early_stopping(tmp_path, model_config):
    trainer = build_trainer(tmp_path, model_config, steps=50, early_stopping_patience=1)
    trainer.state.best_validation_loss = 0.0
    result = trainer.train()
    assert result.stopped_early and result.final_step < 50


def test_oom_reduces_micro_batch_and_preserves_effective_batch(tmp_path, model_config):
    trainer = build_trainer(tmp_path, model_config, steps=1, accumulation=2, micro=8)
    calls = {"count": 0}
    original = trainer._forward_loss

    def flaky(batch):
        calls["count"] += 1
        if batch[0].shape[0] > 4:
            raise RuntimeError("MPS backend out of memory")
        return original(batch)

    trainer._forward_loss = flaky
    trainer.model.train()
    trainer._train_step()
    assert trainer.state.micro_batch_size == 4
    assert trainer.state.gradient_accumulation_steps == 4


def test_oom_at_minimum_batch_raises_resource_error(tmp_path, model_config):
    trainer = build_trainer(tmp_path, model_config, steps=1, micro=1)
    trainer._forward_loss = lambda *a: (_ for _ in ()).throw(RuntimeError("out of memory"))
    trainer.model.train()
    with pytest.raises(ResourceError):
        trainer._train_step()
    assert not is_out_of_memory(RuntimeError("shape mismatch"))


def test_training_is_reproducible_with_same_seed(tmp_path, model_config):
    runs = []
    for name in ("x", "y"):
        trainer = build_trainer(tmp_path, model_config, steps=6, name=name)
        trainer.train()
        runs.append(torch.cat([p.flatten() for p in trainer.model.parameters()]))
    assert torch.allclose(runs[0], runs[1])


def test_no_memory_leak_across_steps(tmp_path, model_config):
    import gc

    trainer = build_trainer(tmp_path, model_config, steps=1)
    trainer.model.train()
    for _ in range(3):
        trainer._train_step()
    gc.collect()
    baseline = sum(1 for o in gc.get_objects() if isinstance(o, torch.Tensor))
    for _ in range(10):
        trainer._train_step()
    gc.collect()
    assert sum(1 for o in gc.get_objects() if isinstance(o, torch.Tensor)) <= baseline + 2


def test_init_from_restores_weights_and_optimizer_moments(tmp_path, model_config):
    from llm.core.config import ExperimentConfig, TokenizerConfig
    from llm.training.runner import build_trainer

    first = build_trainer_for_runner(tmp_path, model_config)
    first.train()
    config = ExperimentConfig(
        name="continue",
        model=model_config,
        tokenizer=TokenizerConfig(vocab_size=model_config.vocab_size, path=str(tmp_path / "tok")),
        train=first.config.model_copy(
            update={
                "init_from": str(tmp_path / "ckpt_runner"),
                "restore_optimizer": True,
                "checkpoint_dir": str(tmp_path / "ckpt_next"),
                "output_dir": str(tmp_path / "exp_next"),
                "max_steps": 20,
            }
        ),
    )
    second = build_trainer(config, resume=False)
    assert second.state.step == 0
    for a, b in zip(first.model.parameters(), second.model.parameters(), strict=True):
        assert torch.allclose(a.cpu(), b.cpu())
    moments = [s["exp_avg"].abs().sum().item() for s in second.optimizer.state.values()]
    assert moments and all(m > 0 for m in moments)


def build_trainer_for_runner(tmp_path, model_config):
    from llm.core.config import TokenizerConfig
    from llm.tokenizer.bpe import BPETokenizer
    from tests.conftest import SAMPLE_TEXTS

    tokenizer = BPETokenizer.train(
        SAMPLE_TEXTS, TokenizerConfig(vocab_size=model_config.vocab_size, min_frequency=1)
    )
    tokenizer.save(tmp_path / "tok")
    trainer = build_trainer(tmp_path, model_config, steps=6, name="runner")
    trainer.tokenizer_version = tokenizer.version
    return trainer


def test_training_result_and_registry_record_capture_run_metadata(tmp_path, model_config):
    from llm.core.config import ExperimentConfig
    from llm.evaluation.registry import load_registry, record_experiment, training_record

    trainer = build_trainer(tmp_path, model_config, steps=6)
    result = trainer.train()
    assert result.tokens_per_second > 0 and result.duration_seconds > 0
    assert result.final_train_loss == result.final_train_loss
    experiment = ExperimentConfig(name="meta", model=model_config, train=trainer.config)
    details, results = training_record(experiment, result, {"conv_pass_rate": 0.5})
    path = tmp_path / "registry.jsonl"
    record_experiment("meta", details, results, path)
    entry = load_registry(path)[0]
    for key in (
        "architecture",
        "parameters",
        "seed",
        "dataset_version",
        "tokens_processed",
        "duration_seconds",
        "train_config",
    ):
        assert key in entry
    for key in (
        "train_loss",
        "validation_loss",
        "perplexity",
        "tokens_per_second",
        "device_memory_gb",
        "conv_pass_rate",
    ):
        assert key in entry["results"]


def test_trainer_reports_real_progress_phases(tmp_path, model_config):
    from llm.progress import read_progress

    trainer = build_trainer(tmp_path, model_config, steps=10)
    trainer.train()
    stage = read_progress(tmp_path / "exp_run")["train"]
    assert stage["status"] == "completed"
    assert stage["total"] == 10
    assert stage["done"] == 10
    assert stage["results"]["final_step"] == 10


def test_recovery_skips_malformed_or_incomplete_checkpoint_metadata(tmp_path, model_config):
    trainer = build_trainer(tmp_path, model_config, steps=15)
    trainer.train()
    manager = trainer.checkpoints
    newest = manager.latest_valid()
    metadata = newest / "meta.json"
    metadata.write_text('{"step":', encoding="utf-8")
    fallback = manager.latest_valid()
    assert fallback is not None and fallback != newest
    metadata.write_text('{"step":15,"model_config":{},"files":{}}', encoding="utf-8")
    assert manager.latest_valid() == fallback
