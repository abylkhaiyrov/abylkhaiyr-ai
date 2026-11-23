from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

from llm.evaluation.comparison import load_engine
from llm.evaluation.v4_language import language_losses
from llm.training.checkpoint import MODEL_FILE

CHECKPOINTS = {
    "v2_start": "checkpoints/tiny_v2/step-00006000",
    "v4_5m_lr1e-4": "checkpoints/tiny_v4_pretrain/step-00000305",
    "a1": "checkpoints/stage18/a-seed-1/step-00000305",
    "a2": "checkpoints/stage18/a-seed-2/step-00000305",
    "b1": "checkpoints/stage18/b-seed-1/step-00000305",
    "b2": "checkpoints/stage18/b-seed-2/step-00000305",
}
SETS = {
    "s18_validation": ("data/stage18/a/tokens", "validation", 10_000),
    "s18_test": ("data/stage18/a/tokens", "test", 10_000),
    "s18_train_a_head": ("data/stage18/a/tokens", "train", 200),
    "s18_train_b_head": ("data/stage18/b/tokens", "train", 200),
    "v4_validation": ("data/v4/tokens", "validation", 200),
    "v4_test": ("data/v4/tokens", "test", 200),
    "v2_validation": ("data/v2/tokens", "validation", 200),
    "v2_test": ("data/v2/tokens", "test", 200),
    "v2_train_head": ("data/v2/tokens", "train", 200),
}
GROUPS = (
    (re.compile(r"embedding"), "embedding"),
    (re.compile(r"attention\.(q|k|v|o)_proj"), "attention"),
    (re.compile(r"feed_forward"), "ffn"),
    (re.compile(r"norm"), "norm"),
)


def group_of(name: str) -> str:
    return next(label for pattern, label in GROUPS if pattern.search(name))


def perplexity_grid(device: str) -> dict:
    grid: dict = {}
    for label, path in CHECKPOINTS.items():
        engine = load_engine(path, "data/tokenizer", device)
        grid[label] = {
            name: language_losses(engine, Path(root), batches, split=split)
            for name, (root, split, batches) in SETS.items()
        }
        del engine
    return grid


def weight_changes() -> dict:
    base = load_file(Path(CHECKPOINTS["v2_start"]) / MODEL_FILE)
    result: dict = {}
    deltas: dict[str, dict[str, torch.Tensor]] = {}
    for label, path in CHECKPOINTS.items():
        if label == "v2_start":
            continue
        current = load_file(Path(path) / MODEL_FILE)
        deltas[label] = {key: current[key].float() - base[key].float() for key in base}
        buckets: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for key, delta in deltas[label].items():
            bucket = buckets[group_of(key)]
            bucket[0] += float(delta.pow(2).sum())
            bucket[1] += float(base[key].float().pow(2).sum())
        layers: dict[str, float] = {}
        for key, delta in deltas[label].items():
            match = re.match(r"blocks\.(\d+)\.", key)
            if match:
                layer = layers.setdefault(match.group(1), 0.0)
                layers[match.group(1)] = layer + float(delta.pow(2).sum())
        result[label] = {
            "relative_change_by_group": {k: (v[0] / v[1]) ** 0.5 for k, v in buckets.items()},
            "delta_norm_by_layer": {
                k: v**0.5 for k, v in sorted(layers.items(), key=lambda i: int(i[0]))
            },
            "total_delta_norm": sum(v[0] for v in buckets.values()) ** 0.5,
        }
    names = list(deltas)
    cosine: dict[str, float] = {}
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            flat_first = torch.cat([deltas[first][key].flatten() for key in base])
            flat_second = torch.cat([deltas[second][key].flatten() for key in base])
            cosine[f"{first}~{second}"] = float(
                torch.nn.functional.cosine_similarity(flat_first, flat_second, dim=0)
            )
    result["delta_cosine"] = cosine
    return result


def training_logs() -> dict:
    result = {}
    for label, run in (
        ("a1", "a-seed-1"),
        ("a2", "a-seed-2"),
        ("b1", "b-seed-1"),
        ("b2", "b-seed-2"),
    ):
        rows = [
            json.loads(line)
            for line in Path(f"experiments/stage18/{run}/metrics.jsonl").read_text().splitlines()
        ]
        train = [row for row in rows if "train/loss" in row]
        gradients = [row["train/grad_norm"] for row in train]
        losses = [row["train/loss"] for row in train]
        result[label] = {
            "first_5_logged_loss": sum(losses[:5]) / 5,
            "last_5_logged_loss": sum(losses[-5:]) / 5,
            "grad_norm_mean": sum(gradients) / len(gradients),
            "grad_norm_min": min(gradients),
            "grad_norm_share_at_or_above_clip": sum(g >= 0.999 for g in gradients) / len(gradients),
            "learning_rates": sorted({row["train/learning_rate"] for row in train}),
            "validation": [
                (row["step"], row["validation/loss"]) for row in rows if "validation/loss" in row
            ],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "perplexity": perplexity_grid(args.device),
        "weights": weight_changes(),
        "logs": training_logs(),
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"written {args.output}")


if __name__ == "__main__":
    main()
