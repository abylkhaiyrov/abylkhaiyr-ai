from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from llm.core.checksums import file_sha256
from llm.core.hardware import resolve_device
from llm.data.binary import read_jsonl
from llm.evaluation.capacity_metrics import score_bank, unigram_probabilities
from llm.evaluation.comparison import load_engine
from llm.evaluation.diagnostic import summarize
from llm.evaluation.diagnostic_bank import Item
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM
from llm.tokenizer.bpe import BPETokenizer
from llm.training.capacity_study import model_profile
from llm.training.engine import seed_everything

METHODS = ("normalized", "total", "option_only", "unigram", "random", "shortest", "numeric_rank")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="+", help="name=CHECKPOINT or name=random:profile:seed")
    parser.add_argument("--root", type=Path, default=Path("experiments/stage17"))
    parser.add_argument("--corpus", type=Path, default=Path("data/v4_expanded"))
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    filename = args.root / "diagnostic_v2.jsonl"
    frozen = json.loads((args.root / "preflight_data.json").read_text(encoding="utf-8"))
    if file_sha256(filename) != frozen["bank_sha256"]:
        parser.error("diagnostic checksum mismatch")
    items = [Item(**row) for row in read_jsonl(filename)]
    tokenizer = BPETokenizer.load(args.tokenizer)
    train = args.corpus / "tokens" / "train.bin"
    if tokenizer.version != frozen["corpus"]["tokenizer"]:
        parser.error("tokenizer version mismatch")
    if file_sha256(train) != frozen["corpus"]["splits"]["train"]["sha256"]:
        parser.error("unigram training data checksum mismatch")
    frequencies = unigram_probabilities(train, tokenizer)
    device = resolve_device(args.device)
    directory = args.root / "diagnostic"
    directory.mkdir(exist_ok=True)
    for spec in args.model:
        name, model = spec.split("=", 1)
        if not name or Path(name).name != name:
            parser.error("model name must be a single filename")
        if model.startswith("random:"):
            _, profile, seed = model.split(":")
            seed_everything(int(seed))
            engine = InferenceEngine(CausalLM(model_profile(profile)), tokenizer, device)
            checkpoint_hash = None
        else:
            engine = load_engine(model, args.tokenizer, args.device)
            checkpoint_hash = file_sha256(Path(model) / "model.safetensors")
        predictions = list(score_bank(engine, items, frequencies))
        report = {
            "bank_sha256": frozen["bank_sha256"],
            "checkpoint": model,
            "checkpoint_sha256": checkpoint_hash,
            "tokenizer": tokenizer.version,
            "metrics": {
                method: summarize(row.as_scored(method) for row in predictions)
                for method in METHODS
            },
            "items": [asdict(row) for row in predictions],
        }
        path = directory / f"{name}.json"
        temporary = path.with_suffix(".part")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
        print(name, report["metrics"]["normalized"]["overall"], flush=True)
        del engine


if __name__ == "__main__":
    main()
