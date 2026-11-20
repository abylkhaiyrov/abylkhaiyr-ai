from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

NOT_MEASURED = "_not measured_"
BASELINE_VARIANT = "baseline"


def load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def number(value: object, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def speed_section(rows: list[dict] | None) -> str:
    if not rows:
        return NOT_MEASURED
    base = next((r for r in rows if r["variant"] == BASELINE_VARIANT), rows[0])
    body = [
        (
            r["variant"],
            number(r["steps_per_second"], 3),
            f"{r['useful_tokens_per_second']:.0f}",
            f"{(r['useful_tokens_per_second'] / base['useful_tokens_per_second'] - 1) * 100:+.1f}%",
            number(r["useful_fraction"], 3),
            number(r["peak_device_gb"], 2),
            number(r["swap_gb"], 1),
        )
        for r in rows
    ]
    return table(
        [
            "variant",
            "steps/s",
            "useful tok/s",
            "vs baseline",
            "useful fraction",
            "device GB",
            "swap GB",
        ],
        body,
    )


def efficiency_section(rows: list[dict] | None) -> str:
    if not rows:
        return NOT_MEASURED
    keys = [
        "steps",
        "supervised_tokens",
        "duration_seconds",
        "useful_tokens_per_second",
        "fixed_validation_loss",
        "conv_pass_rate",
        "device_memory_gb",
    ]
    body = [(r["variant"], *[number(r.get(k), 3) for k in keys]) for r in rows]
    return table(["variant", *keys], body)


def inference_section(rows: list[dict] | None) -> str:
    if not rows:
        return NOT_MEASURED
    keys = [
        "prefill_ms",
        "first_token_ms",
        "decode_ms_per_token",
        "tokens_per_second",
        "peak_device_gb",
    ]
    body = [(r["cache"], r["prompt_tokens"], *[number(r[k], 2) for k in keys]) for r in rows]
    return table(["cache", "prompt tokens", *keys], body)


def comparison_section(payload: dict | None, keys: Sequence[str]) -> str:
    if not payload:
        return NOT_MEASURED
    body = [
        (name, *[number(data["metrics"].get(k), 3) for k in keys]) for name, data in payload.items()
    ]
    return table(["model", *keys], body)


def sweep_section(payload: dict | None) -> str:
    if not payload:
        return NOT_MEASURED
    body = [
        (
            group,
            row["variant"],
            number(row["pass_rate"]),
            number(row["completion_rate"]),
            number(row["repetition"]),
        )
        for group, row in payload["best_by_group"].items()
    ]
    return table(["task group", "best variant", "val F1", "completion", "repetition"], body)


SECTIONS: tuple[tuple[str, Callable[[Path], str]], ...] = (
    (
        "Training throughput (measured on this machine)",
        lambda root: speed_section(load_json(root / "profile_training.json")),
    ),
    (
        "Sample efficiency at an equal supervised-token budget",
        lambda root: efficiency_section(load_json(root / "efficiency.json")),
    ),
    (
        "Inference latency",
        lambda root: inference_section(load_json(root / "inference_benchmark.json")),
    ),
    (
        "Conversational benchmark per model",
        lambda root: comparison_section(
            load_json(root / "conversation_compare.json"),
            [
                "conv_pass_rate",
                "conv_completion_rate",
                "conv_language_stability",
                "conv_repetition_ratio",
                "conv_lang_en",
                "conv_lang_ru",
                "conv_lang_kk",
            ],
        ),
    ),
    (
        "Sampling parameters selected on the validation split",
        lambda root: sweep_section(load_json(root / "sampling_sweep.json")),
    ),
)


def build_report(root: Path, title: str) -> str:
    parts = [
        f"# {title}",
        "",
        "Every number below is read from a result file; missing files are marked.",
    ]
    for heading, render in SECTIONS:
        parts.extend(["", f"## {heading}", "", render(root)])
    return "\n".join(parts) + "\n"
