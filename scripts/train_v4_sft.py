from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from llm.core.checksums import file_sha256
from llm.core.config import load_experiment_config
from llm.tokenizer.bpe import BPETokenizer
from llm.training.runner import build_trainer
from llm.training.v4_study import VARIANTS, planned_exposure, study_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/tiny_v4_pretrain.yaml")
    parser.add_argument("--data", type=Path, default=Path("data/v4/sft_final"))
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    base = load_experiment_config(args.config)
    config = study_config(
        base,
        args.variant,
        args.steps,
        args.seed,
        args.init,
        args.data,
        args.checkpoint_dir,
        args.device,
    )
    tokenizer = BPETokenizer.load(config.tokenizer.path)
    exposure = planned_exposure(config, tokenizer, args.steps)
    contract = {
        "config": config.model_dump(mode="json"),
        "tokenizer": tokenizer.version,
        "data_manifest_sha256": file_sha256(args.data / "manifest.json"),
        "training_files": {
            path.name: file_sha256(path)
            for path in sorted((args.data / "sft_groups").glob("train.*"))
            if path.suffix in {".bin", ".npy"}
        },
        "replay_files": {
            path.name: file_sha256(path)
            for path in sorted((args.data.parent / "tokens").glob("train.*.bin"))
            if args.variant == "replay"
        },
        "init_sha256": file_sha256(args.init / "model.safetensors"),
        "planned_exposure": exposure.as_dict(),
    }
    print(json.dumps(contract), flush=True)
    if args.plan_only:
        return
    expected = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    run = args.checkpoint_dir / "study.json"
    if args.resume:
        if run.read_text(encoding="utf-8") != expected:
            parser.error("resume config, corpus or initialization changed")
    else:
        run.parent.mkdir(parents=True, exist_ok=False)
        run.write_text(expected, encoding="utf-8")
    trainer = build_trainer(config, resume=args.resume)
    if trainer.state.step >= args.steps:
        parser.error("training already reached the requested budget")
    result = trainer.train()
    observed = planned_exposure(config, tokenizer, result.final_step)
    if result.tokens_seen != observed.objective_tokens:
        raise RuntimeError("loader replay does not match measured objective token count")
    summary = {**asdict(result), "exposure": observed.as_dict()}
    if not result.interrupted:
        seconds = max(result.duration_seconds, 1e-9)
        if not args.resume:
            summary["assistant_tokens_per_second"] = observed.assistant_tokens / seconds
        output = json.dumps(summary, indent=2, allow_nan=False)
        (run.parent / "summary.json").write_text(output + "\n", encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
