import torch

from llm.core.config import ModelConfig
from llm.model.transformer import CausalLM
from llm.optimization.adamw import AdamW
from llm.training.checkpoint import CheckpointManager, TrainingState


def _save(manager: CheckpointManager, step: int, label: str | None = None) -> None:
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = CausalLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    manager.save(
        model,
        config,
        optimizer,
        None,
        TrainingState(step=step),
        {},
        "v",
        torch.device("cpu"),
        label,
    )


def test_protected_steps_survive_rotation(tmp_path):
    manager = CheckpointManager(tmp_path, keep=1, protected_steps=[1], alias_prefix="tiny")
    for step in (1, 2, 3):
        _save(manager, step)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert "step-00000001" in names
    assert "step-00000002" not in names
    assert manager.verify(tmp_path / "tiny_step_1")["step"] == 1
    assert manager.verify(tmp_path / "tiny_latest")["step"] == 3


def test_labeled_checkpoint_is_not_rotated(tmp_path):
    manager = CheckpointManager(tmp_path, keep=1, alias_prefix="tiny")
    _save(manager, 1, manager.best_label())
    for step in (2, 3):
        _save(manager, step)
    assert (tmp_path / "tiny_best").is_dir()
    assert [p.name for p in manager.list_steps()] == ["step-00000003"]


def test_rotation_ignores_non_checkpoint_directories(tmp_path):
    manager = CheckpointManager(tmp_path, keep=1)
    unrelated = tmp_path / "step-not-a-checkpoint"
    unrelated.mkdir()
    _save(manager, 1)
    _save(manager, 2)
    assert unrelated.is_dir()
    assert manager.latest_valid() == manager.step_dir(2)
