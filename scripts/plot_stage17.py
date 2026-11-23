from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from llm.core.scaling import flops_per_token
from llm.data.binary import read_jsonl
from llm.training.capacity_study import CONTEXT, SEEDS, TOKENS_PER_STEP, model_profile

COLORS = {"tiny": "#14708d", "small": "#b74a21"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/stage17"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), layout="constrained")
    for profile, color in COLORS.items():
        cost = flops_per_token(model_profile(profile), CONTEXT)
        for seed in SEEDS:
            log = args.root / f"{profile}-seed-{seed}" / "metrics.jsonl"
            style = "-" if seed == SEEDS[0] else "--"
            if log.exists():
                metrics = list(read_jsonl(log))
                for index, name in enumerate(("train/loss", "validation/loss")):
                    points = {row["step"]: row[name] for row in metrics if name in row}
                    axes[index].plot(
                        [step * TOKENS_PER_STEP / 1e6 for step in points],
                        list(points.values()),
                        color=color,
                        linestyle=style,
                        linewidth=1.5,
                        label=f"{profile.title()} seed {seed}",
                    )
            markers = [(0, f"{profile}-init-{seed}"), (305, f"{profile}-5m-{seed}")]
            if profile == "small":
                markers.insert(1, (98, f"small-equivalent-{seed}"))
            steps, rates = [], []
            for step, key in markers:
                report = args.root / "diagnostic" / f"{key}.json"
                if report.is_file():
                    payload = json.loads(report.read_text(encoding="utf-8"))
                    steps.append(step)
                    rates.append(100 * payload["metrics"]["normalized"]["overall"]["rate"])
            axes[2].plot(
                [step * TOKENS_PER_STEP * cost / 1e15 for step in steps],
                rates,
                "o",
                color=color,
                linestyle=style,
                markersize=4,
            )
    for axis, title in zip(
        axes, ("Training loss", "Validation loss", "Диагностика, %"), strict=True
    ):
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlabel("Обработано токенов, млн")
    axes[1].set_xlabel("Обработано токенов, млн")
    axes[2].set_xlabel("Оценка FLOPs, ×10¹⁵")
    axes[2].axhline(25, color="#555555", linestyle=":", linewidth=1)
    axes[0].legend(frameon=False, fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
