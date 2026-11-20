from __future__ import annotations

import argparse
import json

from llm.training.pipeline import PipelineConfig, TrainingPipeline


def main() -> int:
    parser = argparse.ArgumentParser(prog="run_pipeline")
    parser.add_argument("--pretrain", default="configs/tiny_v2.yaml")
    parser.add_argument("--sft", default="configs/tiny_v2_sft.yaml")
    parser.add_argument("--dpo", default="configs/tiny_v2_dpo.yaml")
    parser.add_argument("--baseline", default="checkpoints/tiny_session")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--data-dir", default="data/v2")
    parser.add_argument("--state-dir", default="experiments/pipeline_tiny_v2")
    parser.add_argument("--release-dir", default="releases")
    parser.add_argument("--version", default="v0.1.0")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--online-eval", action="store_true")
    parser.add_argument("--build-preference", action="store_true")
    args = parser.parse_args()
    config = PipelineConfig(
        pretrain_config=args.pretrain,
        sft_config=args.sft,
        dpo_config=args.dpo,
        baseline_checkpoint=args.baseline,
        tokenizer_path=args.tokenizer,
        data_dir=args.data_dir,
        state_dir=args.state_dir,
        release_dir=args.release_dir,
        release_version=args.version,
        device=args.device,
        offline_eval=not args.online_eval,
        build_preference=args.build_preference,
    )
    state = TrainingPipeline(config).run()
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
