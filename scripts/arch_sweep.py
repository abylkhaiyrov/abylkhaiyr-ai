import itertools
import json
import shutil
import sys
import time
from pathlib import Path

from llm.core.config import ExperimentConfig, load_experiment_config
from llm.evaluation.registry import record_experiment
from llm.training.runner import build_trainer

BASE_CONFIG = sys.argv[1]
TOKEN_BUDGET = int(float(sys.argv[2]))
SWEEP_ROOT = Path("experiments/sweep")
MIXTURE = {"ru": 1.0, "kk": 1.0, "en": 1.0, "math": 0.5, "code": 0.5}
VARIANTS: dict[str, dict[str, float | int]] = {
    "base": {},
    "lr_5e-4": {"train.optimizer.learning_rate": 5e-4},
    "lr_2e-3": {"train.optimizer.learning_rate": 2e-3},
    "wd_0": {"train.optimizer.weight_decay": 0.0},
    "wd_0.3": {"train.optimizer.weight_decay": 0.3},
    "warmup_low": {"warmup_fraction": 0.03},
    "warmup_high": {"warmup_fraction": 0.3},
    "dropout_0.05": {"model.dropout": 0.05},
    "dropout_0.1": {"model.dropout": 0.1},
    "batch_32": {"train.gradient_accumulation_steps": 4},
    "batch_128": {"train.gradient_accumulation_steps": 16},
    "clip_0.5": {"train.optimizer.max_grad_norm": 0.5},
    "clip_2.0": {"train.optimizer.max_grad_norm": 2.0},
}
DEFAULT_WARMUP_FRACTION = 0.1


def apply(config: ExperimentConfig, overrides: dict[str, float | int]) -> None:
    for path, value in overrides.items():
        if "." not in path:
            continue
        target = config
        *parents, leaf = path.split(".")
        for name in parents:
            target = getattr(target, name)
        setattr(target, leaf, value)


def build_config(name: str, overrides: dict[str, float | int], root: Path) -> ExperimentConfig:
    config = load_experiment_config(BASE_CONFIG)
    config.train.mixture = MIXTURE
    apply(config, overrides)
    train = config.train
    tokens_per_step = (
        train.micro_batch_size * train.gradient_accumulation_steps * train.sequence_length
    )
    steps = max(TOKEN_BUDGET // tokens_per_step, 10)
    fraction = float(overrides.get("warmup_fraction", DEFAULT_WARMUP_FRACTION))
    train.max_steps = steps
    train.eval_interval = steps
    train.eval_batches = 40
    train.checkpoint_interval = steps * 2
    train.log_interval = max(steps // 10, 1)
    train.optimizer.warmup_steps = max(int(steps * fraction), 1)
    train.output_dir = str(root)
    train.checkpoint_dir = str(root / "ckpt")
    train.data_dir = "data/v2"
    train.restore_optimizer = False
    return config


def stability(root: Path) -> dict[str, float]:
    rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    norms = [r["train/grad_norm"] for r in rows if "train/grad_norm" in r]
    losses = [r["train/loss"] for r in rows if "train/loss" in r]
    spikes = sum(b > a + 0.3 for a, b in itertools.pairwise(losses))
    return {
        "max_grad_norm": max(norms),
        "mean_grad_norm": sum(norms) / len(norms),
        "spikes": spikes,
    }


def run(name: str, overrides: dict[str, float | int]) -> dict[str, float]:
    root = SWEEP_ROOT / name
    shutil.rmtree(root, ignore_errors=True)
    config = build_config(name, overrides, root)
    trainer = build_trainer(config, resume=False)
    started = time.monotonic()
    result = trainer.train()
    elapsed = time.monotonic() - started
    shutil.rmtree(root / "ckpt", ignore_errors=True)
    results = {
        "validation_loss": result.last_validation_loss,
        "tokens_per_second": result.tokens_seen / elapsed,
        **stability(root),
    }
    record_experiment(
        f"sweep_{name}",
        {
            "steps": result.final_step,
            "tokens": result.tokens_seen,
            "overrides": overrides,
            "dataset": "v2",
            "init_from": config.train.init_from,
        },
        results,
    )
    print(name, result.final_step, json.dumps(results), flush=True)
    return results


if __name__ == "__main__":
    selected = sys.argv[3:] or list(VARIANTS)
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {name: run(name, VARIANTS[name]) for name in selected}
    (SWEEP_ROOT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
