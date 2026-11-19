from __future__ import annotations

import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm.training.checkpoint import MODEL_FILE, CheckpointManager


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _model_card(version: str, meta: dict[str, Any], manifest: dict[str, Any]) -> str:
    cfg = meta.get("model_config", {})
    licenses = sorted(
        {info.get("license", "unknown") for info in manifest.get("sources", {}).values()}
    )
    return "\n".join(
        [
            f"# Abylkhaiyr AI Tiny ({version})",
            "",
            "## Overview",
            "",
            "- Architecture: Decoder-only Transformer (RMSNorm, RoPE, SwiGLU, GQA, tied embeddings)",
            f"- Vocabulary size: {cfg.get('vocab_size', 8192)}",
            f"- Hidden size: {cfg.get('hidden_size', 384)}",
            f"- Layers: {cfg.get('num_hidden_layers', 8)}",
            f"- Attention heads / KV heads: {cfg.get('num_attention_heads', 6)} / {cfg.get('num_key_value_heads', 2)}",
            f"- Context length: {cfg.get('max_position_embeddings', 512)}",
            "- Languages: Russian (ru), Kazakh (kk), English (en), plus math and code",
            "- Training origin: Trained from scratch on local Apple Silicon (MPS, FP32) without external LLM weights or external LLM APIs.",
            "",
            "## Data Provenance",
            "",
            f"- Dataset version: v2 ({manifest.get('tokenized', {}).get('train', {}).get('tokens', 0)} train tokens)",
            f"- Source licenses: {', '.join(licenses) if licenses else 'see DATASET_MANIFEST.json'}",
            "- Decontamination and split deduplication applied across pretraining and SFT splits.",
            "",
            "## Limitations",
            "",
            "- Small parameter count (~15.7M) limits factual recall, complex multi-step reasoning, and long-horizon code synthesis.",
            "- Context window is limited to 512 tokens.",
            "- May hallucinate facts or repeat phrases on out-of-distribution prompts.",
            "- This is an experimental local model and does not match frontier models such as GPT or Claude.",
            "",
        ]
    )


def _training_report(
    version: str,
    sweep_summary: dict[str, Any],
    stage_metrics: dict[str, Any],
) -> str:
    lines = [f"# Training Report — Abylkhaiyr AI Tiny ({version})", ""]
    if sweep_summary:
        rows = []
        for name, item in sweep_summary.items():
            loss = float(item.get("validation_loss", float("nan")))
            ppl = math.exp(loss) if math.isfinite(loss) else float("nan")
            rows.append(
                [
                    name,
                    f"{loss:.4f}",
                    f"{ppl:.2f}",
                    f"{float(item.get('mean_grad_norm', 0.0)):.3f}",
                    f"{float(item.get('max_grad_norm', 0.0)):.3f}",
                    f"{float(item.get('tokens_per_second', 0.0)):.1f}",
                    str(item.get("spikes", 0)),
                ]
            )
        lines.extend(
            [
                "## Hyperparameter Sweep (13 Experiments)",
                "",
                _format_table(
                    ["Variant", "Val Loss", "PPL", "Mean Grad", "Max Grad", "Tok/s", "Spikes"],
                    rows,
                ),
                "",
            ]
        )
    lines.append("## Training Stages")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(stage_metrics, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _evaluation_report(
    version: str,
    baseline_metrics: dict[str, float],
    final_metrics: dict[str, float],
    samples: list[dict[str, str]],
) -> str:
    all_keys = sorted(set(baseline_metrics) | set(final_metrics))
    rows = []
    for key in all_keys:
        base_val = baseline_metrics.get(key)
        new_val = final_metrics.get(key)
        rows.append(
            [
                key,
                f"{base_val:.4f}" if isinstance(base_val, (int, float)) else "-",
                f"{new_val:.4f}" if isinstance(new_val, (int, float)) else "-",
            ]
        )
    lines = [
        f"# Evaluation Report — Abylkhaiyr AI Tiny ({version})",
        "",
        "## Metrics (Baseline vs Released Model)",
        "",
        _format_table(["Metric", "Baseline (tiny_session)", f"Released ({version})"], rows),
        "",
        "## Qualitative Samples (Without Tools vs With Tools)",
        "",
    ]
    for sample in samples:
        lines.extend(
            [
                f"### {sample.get('category', 'sample')}",
                f"- Prompt: `{sample.get('prompt', '')}`",
                f"- Without tools: `{sample.get('without_tools', '')}`",
                f"- With tools: `{sample.get('with_tools', '')}`",
                "",
            ]
        )
    return "\n".join(lines)


def _run_instructions(version: str, release_dir: Path) -> str:
    return "\n".join(
        [
            f"# Run Instructions — Abylkhaiyr AI Tiny ({version})",
            "",
            "## Generate",
            "",
            "```bash",
            f'.venv/bin/llm generate --checkpoint {release_dir} --tokenizer {release_dir} "Привет, расскажи о Казахстане"',
            "```",
            "",
            "## Interactive Chat",
            "",
            "```bash",
            f".venv/bin/llm chat --checkpoint {release_dir} --tokenizer {release_dir}",
            "```",
            "",
            "## Local OpenAI-Compatible Server",
            "",
            "```bash",
            f".venv/bin/llm serve --checkpoint {release_dir} --tokenizer {release_dir} --model-name abylkhaiyr-tiny-{version}",
            "```",
            "",
            "## Local Studio UI",
            "",
            "```bash",
            f".venv/bin/llm studio --checkpoint {release_dir} --tokenizer {release_dir}",
            "```",
            "",
        ]
    )


def export_release(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    version: str,
    stage_metrics: dict[str, Any] | None = None,
    eval_metrics: dict[str, float] | None = None,
    baseline_metrics: dict[str, float] | None = None,
    samples: list[dict[str, str]] | None = None,
    sweep_summary_path: str | Path = "experiments/sweep/summary.json",
) -> Path:
    ckpt = Path(checkpoint_path)
    resolved = CheckpointManager(ckpt, keep=1).latest_valid() or ckpt
    meta = CheckpointManager.verify(resolved)
    target = Path(output_dir) / f"abylkhaiyr-ai-tiny-{version}"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved / MODEL_FILE, target / MODEL_FILE)
    config_payload = {
        "model_name": "abylkhaiyr-ai-tiny",
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "source_checkpoint": str(resolved),
        "step": meta.get("step"),
        "tokens_seen": meta.get("tokens_seen"),
        "tokenizer_version": meta.get("tokenizer_version"),
        "model_config": meta.get("model_config", {}),
    }
    (target / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tok_dir = Path(tokenizer_path)
    for item in tok_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, target / item.name)
    data_root = Path(data_dir)
    manifest = _read_json(data_root / "manifest.json")
    dataset_bundle = {
        "manifest": manifest,
        "audit": _read_json(data_root / "audit.json"),
        "corpus_report": _read_json(data_root / "corpus_report.json"),
    }
    (target / "DATASET_MANIFEST.json").write_text(
        json.dumps(dataset_bundle, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    sweep_summary = _read_json(Path(sweep_summary_path))
    (target / "MODEL_CARD.md").write_text(_model_card(version, meta, manifest), encoding="utf-8")
    (target / "TRAINING_REPORT.md").write_text(
        _training_report(version, sweep_summary, stage_metrics or {}), encoding="utf-8"
    )
    (target / "EVALUATION_REPORT.md").write_text(
        _evaluation_report(version, baseline_metrics or {}, eval_metrics or {}, samples or []),
        encoding="utf-8",
    )
    (target / "RUN_INSTRUCTIONS.md").write_text(
        _run_instructions(version, target), encoding="utf-8"
    )
    return target
