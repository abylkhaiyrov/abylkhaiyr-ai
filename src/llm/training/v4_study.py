from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from llm.core.config import ExperimentConfig
from llm.data.binary import IGNORE_INDEX
from llm.tokenizer.bpe import BPETokenizer
from llm.training.runner import _build_loaders

BALANCED_MIXTURE = {
    "dialogue_en": 0.24,
    "dialogue_ru": 0.23,
    "semantic_en": 0.14,
    "semantic_ru": 0.12,
    "semantic_kk": 0.21,
    "math": 0.03,
    "code": 0.03,
}
CURRICULUM_START = {
    "dialogue_en": 0.4,
    "dialogue_ru": 0.35,
    "semantic_en": 0.06,
    "semantic_ru": 0.05,
    "semantic_kk": 0.12,
    "math": 0.01,
    "code": 0.01,
}
VARIANTS = ("balanced", "curriculum", "replay", "dialogue_reading")


@dataclass
class SFTExposure:
    compute_slots: int = 0
    assistant_tokens: int = 0
    replay_target_tokens: int = 0
    input_tokens: int = 0
    assistant_micro_batches: int = 0
    replay_micro_batches: int = 0

    @property
    def objective_tokens(self) -> int:
        return self.assistant_tokens + self.replay_target_tokens

    def as_dict(self) -> dict[str, int | float]:
        return {**asdict(self), "objective_tokens": self.objective_tokens}


def study_config(
    base: ExperimentConfig,
    variant: str,
    steps: int,
    seed: int,
    init: Path,
    root: Path,
    checkpoint_dir: Path,
    device: str,
    mixture: Mapping[str, float] | None = None,
    learning_rate: float = 2e-4,
) -> ExperimentConfig:
    if variant not in VARIANTS:
        raise ValueError(f"unknown SFT variant: {variant}")
    if steps < 1:
        raise ValueError("SFT steps must be positive")
    if not init.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {init}")
    chosen = dict(BALANCED_MIXTURE if mixture is None else mixture)
    if variant == "dialogue_reading":
        chosen.pop("semantic_kk", None)
    groups = root / "sft_groups"
    for name in chosen:
        for split in ("train", "validation"):
            if not (groups / f"{split}.{name}.ids.bin").is_file():
                raise ValueError(f"missing SFT split {split}.{name}")
    optimizer = base.train.optimizer.model_copy(
        update={"learning_rate": learning_rate, "warmup_steps": min(10, steps // 10)}
    )
    has_curriculum = variant == "curriculum"
    grad_accum = 2
    train = base.train.model_copy(
        update={
            "stage": "sft",
            "data_dir": str(root),
            "init_from": str(init),
            "output_dir": str(checkpoint_dir / "progress"),
            "checkpoint_dir": str(checkpoint_dir),
            "device": device,
            "seed": seed,
            "max_steps": steps,
            "micro_batch_size": 4,
            "gradient_accumulation_steps": grad_accum,
            "pack_sequences": True,
            "fast_causal": True,
            "optimizer": optimizer,
            "eval_interval": min(steps, 40),
            "log_interval": min(steps, 10),
            "checkpoint_interval": min(steps, 40),
            "protected_steps": [steps],
            "sft_replay_fraction": 0.15 if variant == "replay" else 0.0,
            "sft_replay_dir": str(root.parent / "tokens"),
            "mixture": chosen,
            "curriculum_start": CURRICULUM_START if has_curriculum else {},
            "curriculum_batches": steps * grad_accum if has_curriculum else 0,
        }
    )
    return base.model_copy(update={"name": f"tiny_v4_sft_{variant}", "train": train})


def planned_exposure(config: ExperimentConfig, tokenizer: BPETokenizer, steps: int) -> SFTExposure:
    if not config.train.pack_sequences:
        raise ValueError("exposure accounting requires segmented sequence packing")
    source, _ = _build_loaders(config, tokenizer)
    exposure = SFTExposure()
    for _ in range(steps * config.train.gradient_accumulation_steps):
        inputs, targets, *segments = source.next_batch()
        target_count = int((targets != IGNORE_INDEX).sum().item())
        exposure.compute_slots += inputs.numel()
        if segments:
            exposure.input_tokens += int((segments[0] != 0).sum().item())
            exposure.assistant_tokens += target_count
            exposure.assistant_micro_batches += 1
        else:
            exposure.input_tokens += inputs.numel()
            exposure.replay_target_tokens += target_count
            exposure.replay_micro_batches += 1
    return exposure
