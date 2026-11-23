from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from llm.data.binary import read_jsonl
from llm.evaluation.v4_review import Review, rater_agreement


def load_predictions(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in read_jsonl(path) if "response" in row}


def export_packet(args: argparse.Namespace) -> None:
    models = {
        item.split("=", 1)[0]: load_predictions(Path(item.split("=", 1)[1]))
        for item in args.predictions
    }
    cases = [row for row in read_jsonl(args.cases) if row["kind"] == "free"]
    by_language: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        if all(case["id"] in predictions for predictions in models.values()):
            by_language[case["language"]].append(case)
    generator = random.Random(args.seed)
    packet: list[dict] = []
    key: dict[str, dict] = {}
    for language in sorted(by_language):
        chosen = generator.sample(
            by_language[language], min(args.per_language, len(by_language[language]))
        )
        for case in chosen:
            order = list(models)
            generator.shuffle(order)
            for model in order:
                item_id = f"item-{len(packet):05d}"
                row = models[model][case["id"]]
                packet.append(
                    {
                        "item_id": item_id,
                        "language": language,
                        "category": case["category"],
                        "messages": row["messages"],
                        "response": row["response"],
                    }
                )
                key[item_id] = {
                    "model": model,
                    "case_id": case["id"],
                    "response_sha256": row["response_sha256"],
                }
    args.output.mkdir(parents=True)
    (args.output / "packet.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet)
    )
    (args.output / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(packet)} items written to {args.output}")


def import_ratings(args: argparse.Namespace) -> None:
    key = json.loads(args.key.read_text())
    grouped: dict[str, list[Review]] = defaultdict(list)
    for row in read_jsonl(args.ratings):
        entry = key[row["item_id"]]
        grouped[entry["model"]].append(
            Review(
                id=entry["case_id"],
                response_sha256=entry["response_sha256"],
                ratings=row["ratings"],
            )
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for model, reviews in grouped.items():
        target = args.output / f"{model}.annotations.jsonl"
        target.write_text("".join(review.model_dump_json() + "\n" for review in reviews))


def agreement(args: argparse.Namespace) -> None:
    ratings = [
        {row["item_id"]: row["ratings"] for row in read_jsonl(path)}
        for path in (args.first, args.second)
    ]
    print(json.dumps(rater_agreement(*ratings), indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--cases", type=Path, required=True)
    export.add_argument("--predictions", nargs="+", required=True, help="model=predictions.jsonl")
    export.add_argument("--per-language", type=int, default=20)
    export.add_argument("--seed", type=int, default=7)
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(handler=export_packet)
    importer = commands.add_parser("import")
    importer.add_argument("--key", type=Path, required=True)
    importer.add_argument("--ratings", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    importer.set_defaults(handler=import_ratings)
    compare = commands.add_parser("agreement")
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    compare.set_defaults(handler=agreement)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
