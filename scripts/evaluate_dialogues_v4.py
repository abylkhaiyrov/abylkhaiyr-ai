from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.data.binary import read_jsonl
from llm.evaluation.comparison import load_engine
from llm.evaluation.v4_review import Review, metrics, predictions
from llm.inference.sampling import SamplingParams
from llm.training.checkpoint import CheckpointManager, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--cases", type=Path, default=Path("data/v4/sft_final/validation.jsonl"))
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--exclude-ids", type=Path)
    parser.add_argument("--groups", type=Path, default=Path("configs/ability_groups_v5.json"))
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch size and generation budget must be positive")
    checkpoint = CheckpointManager(args.checkpoint, keep=1).latest_valid() or Path(args.checkpoint)
    if not args.score_only:
        if args.predictions.exists():
            parser.error("predictions already exist; use --score-only to review them")
        engine = load_engine(str(checkpoint), args.tokenizer, args.device)
        cases = list(read_jsonl(args.cases))
        params = SamplingParams(
            greedy=True, max_new_tokens=args.max_new_tokens, repetition_penalty=1.1
        )
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.predictions.with_suffix(".part")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in predictions(engine, cases, params, args.batch_size):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with temporary.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "provenance": {
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
                            "cases_sha256": file_sha256(args.cases),
                            "tokenizer": engine.tokenizer.version,
                            "device": args.device,
                            "max_new_tokens": args.max_new_tokens,
                            "greedy": True,
                            "repetition_penalty": 1.1,
                        },
                    }
                )
                + "\n"
            )
        temporary.replace(args.predictions)
    rows = list(read_jsonl(args.predictions))
    provenance = rows.pop()["provenance"]
    if args.exclude_ids:
        excluded = set(json.loads(args.exclude_ids.read_text(encoding="utf-8")))
        rows = [row for row in rows if row["id"] not in excluded]
        provenance["excluded_cases"] = len(excluded)
    if provenance["checkpoint_sha256"] != file_sha256(checkpoint / "model.safetensors"):
        parser.error("predictions belong to a different checkpoint")
    if provenance["cases_sha256"] != file_sha256(args.cases):
        parser.error("prediction case file has changed")
    reviews = (
        [Review.model_validate(row) for row in read_jsonl(args.annotations)]
        if args.annotations
        else []
    )
    report = {"provenance": provenance, "overall": metrics(rows, reviews), "by_language": {}}
    for language in sorted({row["language"] for row in rows}):
        group = [row for row in rows if row["language"] == language]
        keys = {row["id"] for row in group}
        report["by_language"][language] = metrics(
            group, [review for review in reviews if review.id in keys]
        )
    membership = {
        category: group
        for group, categories in json.loads(args.groups.read_text(encoding="utf-8")).items()
        for category in categories
    }
    report["by_group"] = {
        group: metrics([row for row in rows if membership.get(row["category"]) == group])
        for group in sorted(set(membership.values()))
        if any(membership.get(row["category"]) == group for row in rows)
    }
    report["by_category"] = {
        category: metrics([row for row in rows if row["category"] == category])
        for category in sorted({row["category"] for row in rows})
    }
    result = json.dumps(report, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
