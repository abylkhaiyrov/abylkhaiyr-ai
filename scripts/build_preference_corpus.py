from __future__ import annotations

import argparse
import json

from llm.core.hardware import resolve_device
from llm.data.preference_builder import PreferenceBuildConfig, build_preference_corpus
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.tokenizer.bpe import BPETokenizer, Message
from llm.training.checkpoint import CheckpointManager, load_model_for_inference


def main() -> int:
    parser = argparse.ArgumentParser(prog="build_preference_corpus")
    parser.add_argument("--data-dir", default="data/v3")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--policy", help="SFT checkpoint used to sample rejected answers")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--pairs-per-group", type=int, default=1500)
    parser.add_argument("--human-pairs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    tokenizer = BPETokenizer.load(args.tokenizer)
    sampler = None
    if args.policy:
        path = CheckpointManager(args.policy, keep=1).latest_valid()
        model, _, _ = load_model_for_inference(path or args.policy)
        engine = InferenceEngine(model, tokenizer, resolve_device(args.device))
        params = SamplingParams(max_new_tokens=96, temperature=0.8, top_p=0.95, seed=args.seed)

        def sampler(messages: list[dict[str, str]]) -> str:
            return engine.chat([Message(m["role"], m["content"]) for m in messages], params).text

    config = PreferenceBuildConfig(
        data_dir=args.data_dir,
        pairs_per_group=args.pairs_per_group,
        human_pairs=args.human_pairs,
        seed=args.seed,
    )
    print(json.dumps(build_preference_corpus(config, tokenizer, sampler), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
