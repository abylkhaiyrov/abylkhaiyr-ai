from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llm.data.copy_ladder import CopyCase
from llm.data.grounded import INVENTED
from llm.tokenizer.bpe import BPETokenizer

REPLACEMENT = "�"


@dataclass(frozen=True)
class Occurrence:
    start: int
    ids: tuple[int, ...]


def encoded(tokenizer: BPETokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    result = tokenizer.backend.encode(text, add_special_tokens=False)
    return list(result.ids), [(start, end) for start, end in result.offsets]


def occurrences(tokenizer: BPETokenizer, text: str, names: Iterable[str]) -> dict[str, Occurrence]:
    ids, spans = encoded(tokenizer, text)
    found: dict[str, Occurrence] = {}
    for name in names:
        start = text.find(name)
        if start < 0:
            continue
        end = start + len(name)
        inside = [i for i, (a, b) in enumerate(spans) if a < end and b > start]
        found[name] = Occurrence(start, tuple(ids[i] for i in inside))
    return found


def value_statistics(
    tokenizer: BPETokenizer, groups: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, values in groups.items():
        sizes = [len(tokenizer.encode(f" {value}")) for value in values]
        pieces = [
            tokenizer.decode([token]) for value in values for token in tokenizer.encode(f" {value}")
        ]
        report[name] = {
            "values": len(values),
            "tokens_mean": statistics.fmean(sizes),
            "tokens_min": min(sizes),
            "tokens_max": max(sizes),
            "single_token_share": sum(size == 1 for size in sizes) / len(sizes),
            "multi_token_share": sum(size > 1 for size in sizes) / len(sizes),
            "four_plus_share": sum(size >= 4 for size in sizes) / len(sizes),
            "single_character_piece_share": sum(len(p.strip()) <= 1 for p in pieces) / len(pieces),
            "partial_utf8_piece_share": sum(REPLACEMENT in p for p in pieces) / len(pieces),
        }
    return report


def roundtrip(tokenizer: BPETokenizer, texts: Iterable[str]) -> dict[str, int]:
    checked = failed = case_loss = invalid = 0
    for text in texts:
        checked += 1
        restored = tokenizer.decode(tokenizer.encode(text))
        failed += restored != text
        case_loss += restored.casefold() == text.casefold() and restored != text
        try:
            text.encode("utf-8").decode("utf-8")
            restored.encode("utf-8").decode("utf-8")
        except UnicodeError:
            invalid += 1
    return {
        "checked": checked,
        "mismatches": failed,
        "case_changes": case_loss,
        "invalid_utf8": invalid,
    }


def identifier_match(tokenizer: BPETokenizer, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        prompt, answer = row["messages"][0]["content"], row["messages"][-1]["content"]
        names = set(INVENTED.findall(answer)) & set(INVENTED.findall(prompt))
        source = occurrences(tokenizer, prompt, names)
        target = occurrences(tokenizer, answer, names)
        for name in names & source.keys() & target.keys():
            place = "answer_start" if target[name].start == 0 else "after_text"
            same = target[name].ids == source[name].ids
            stats[place].append(int(same))
            stats[f"{place}_first"].append(int(target[name].ids[0] == source[name].ids[0]))
    return {
        key: {"occurrences": len(values), "match_share": statistics.fmean(values)}
        for key, values in sorted(stats.items())
    }


def case_groups(cases: Sequence[CopyCase]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        groups[f"{case.language}/{case.category}"].append(case.target)
    return dict(groups)


def audit(
    tokenizer: BPETokenizer, cases: Sequence[CopyCase], rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    texts = [case.prompt for case in cases] + [case.target for case in cases]
    texts += [turn["content"] for row in rows[:2000] for turn in row["messages"]]
    names = sorted(
        {name for row in rows for name in INVENTED.findall(row["messages"][0]["content"])}
    )
    return {
        "vocabulary": tokenizer.vocab_size,
        "version": tokenizer.version,
        "ladder_values": value_statistics(tokenizer, case_groups(cases)),
        "training_names": value_statistics(tokenizer, {"invented": names}) if names else {},
        "roundtrip": roundtrip(tokenizer, texts),
        "identifier_match": identifier_match(tokenizer, rows),
    }
