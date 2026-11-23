from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from llm.core.checksums import file_sha256
from llm.data.v4_splits import normalized
from llm.evaluation.diagnostic_bank import Item
from llm.tokenizer.bpe import BPETokenizer

LANGUAGES = ("en", "ru", "kk")
ORDER_SEED = 1818


def read_bank(path: Path) -> list[Item]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or len(lines) % 4:
        raise ValueError("expected one header and three translations per family")
    counts: Counter[str] = Counter()
    result = []
    families: set[str] = set()
    for start in range(0, len(lines), 4):
        if not lines[start].startswith("@"):
            raise ValueError("expected family header")
        task, family = lines[start][1:].split()
        if family in families:
            raise ValueError(f"reused family: {family}")
        families.add(family)
        for index, language in enumerate(LANGUAGES):
            prefix, correct, *others = lines[start + 1 + index].split("|")
            if len(others) != 3 or len({correct, *others}) != 4:
                raise ValueError(f"{family}/{language}: expected four distinct answers")
            rng = random.Random(f"{ORDER_SEED}/{family}/{language}")
            rng.shuffle(others)
            position = (counts[task] + index) % 4
            others.insert(position, correct)
            result.append(
                Item(
                    f"stage18/{family}/{language}",
                    language,
                    task,
                    task,
                    family,
                    "stage18_original",
                    prefix,
                    tuple(others),
                    position,
                )
            )
        counts[task] += 1
    if any(count % 4 for count in counts.values()):
        raise ValueError("each language/task cell must balance all four positions")
    if len({normalized(item.prefix) for item in result}) != len(result):
        raise ValueError("duplicate question")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/stage18"))
    parser.add_argument("--tokenizer", default="data/tokenizer")
    args = parser.parse_args()
    tokenizer = BPETokenizer.load(args.tokenizer)
    path = args.root / "natural_bank.txt"
    output = args.root / "diagnostic.jsonl"
    generation = args.root / "free_cases.jsonl"
    cases = read_bank(path)
    shortest = longest = boundary_changes = 0
    for item in cases:
        ids = [tokenizer.encode(item.separator + option) for option in item.options]
        if len(set(map(tuple, ids))) != 4:
            raise ValueError(f"{item.id}: duplicate tokenized options")
        lengths = list(map(len, ids))
        n = lengths[item.answer]
        shortest += n == min(lengths) and lengths.count(n) == 1
        longest += n == max(lengths) and lengths.count(n) == 1
        prefix = tokenizer.encode(item.prefix)
        boundary_changes += sum(
            tokenizer.encode(item.prefix + item.separator + option) != prefix + ids[index]
            for index, option in enumerate(item.options)
        )
    serialized = "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in cases)
    if output.is_file():
        if output.read_text(encoding="utf-8") != serialized:
            raise ValueError("existing diagnostic differs")
    else:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    report = {
        "diagnostic_input_sha256": file_sha256(path),
        "diagnostic_sha256": file_sha256(output),
        "generation_sha256": file_sha256(generation),
        "items": len(cases),
        "families": len({item.family for item in cases}),
        "unique_shortest_gold": shortest,
        "unique_longest_gold": longest,
        "token_boundary_changes": boundary_changes,
    }
    (args.root / "evaluation_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
