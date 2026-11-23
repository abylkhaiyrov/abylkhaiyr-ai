from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from llm.core.checksums import file_sha256
from llm.data.binary import read_jsonl
from llm.evaluation.capacity_metrics import score_bank, unigram_probabilities
from llm.evaluation.comparison import FIXED_PARAMS, load_engine
from llm.evaluation.conversation import engine_responder, run_conversation_benchmark
from llm.evaluation.diagnostic import summarize
from llm.evaluation.diagnostic_bank import Item
from llm.evaluation.v4_language import language_losses
from llm.evaluation.v4_review import metrics, predictions
from llm.inference.sampling import SamplingParams
from llm.training.checkpoint import CheckpointManager

ROOT = Path("experiments/stage18")
METHODS = ("normalized", "total", "option_only", "random", "shortest", "unigram")
LIMITS = {"free": 64, "natural_v4": 64}
EVALUATIONS = {
    "free": ROOT / "free_cases.jsonl",
    "natural_v4": Path("configs/natural_eval_v4.jsonl"),
}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("expected a single filename")
    torch.set_num_threads(2)
    lock = json.loads(Path("configs/stage18_pilot_lock.json").read_text(encoding="utf-8"))
    bank = ROOT / "diagnostic.jsonl"
    corpus = Path("data/stage18")
    for key, path in (
        ("diagnosis", bank),
        ("generation", EVALUATIONS["free"]),
        ("corpus", corpus / "manifest.json"),
        ("historical_diagnostic", Path("experiments/stage17/diagnostic_v2.jsonl")),
        ("historical_generation", EVALUATIONS["natural_v4"]),
        ("unigram", Path("data/v4_expanded/tokens/train.bin")),
    ):
        if file_sha256(path) != lock[key]:
            parser.error(f"changed evaluation input: {key}")
    metadata = CheckpointManager.verify(args.checkpoint)
    engine = load_engine(str(args.checkpoint), "data/tokenizer", "cpu")
    frequencies = unigram_probabilities(Path("data/v4_expanded/tokens/train.bin"), engine.tokenizer)
    reports = ROOT / "evaluation"
    reports.mkdir(exist_ok=True)
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": metadata["files"]["model.safetensors"],
        "diagnostic_sha256": lock["diagnosis"],
    }
    for label, filename in (
        ("diagnostic", bank),
        ("historical_v2", Path("experiments/stage17/diagnostic_v2.jsonl")),
    ):
        items = [Item(**row) for row in read_jsonl(filename)]
        rows = list(score_bank(engine, items, frequencies))
        result[label] = {
            method: summarize(row.as_scored(method) for row in rows) for method in METHODS
        }
        if label == "diagnostic":
            result["predictions"] = [asdict(row) for row in rows]
    result["language"] = {
        split: language_losses(engine, corpus / "a" / "tokens", 10_000, split=split)
        for split in ("validation", "test")
    }
    for label, filename in EVALUATIONS.items():
        cases = list(read_jsonl(filename))
        params = SamplingParams(greedy=True, max_new_tokens=LIMITS[label], repetition_penalty=1.1)
        generated = list(predictions(engine, cases, params, batch_size=8))
        with (reports / f"{args.name}.{label}.jsonl").open("w", encoding="utf-8") as handle:
            for row in generated:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        result[label] = metrics(generated)
        result[f"{label}_sha256"] = file_sha256(filename)
    legacy = run_conversation_benchmark(engine_responder(engine, FIXED_PARAMS))
    result["legacy_conv"] = legacy.metrics
    write_json(reports / f"{args.name}.json", result)
    print(
        f"{args.name}: diagnostic {result['diagnostic']['normalized']['overall']['rate']:.4f}, "
        f"legacy conv {legacy.metrics['conv_pass_rate']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
