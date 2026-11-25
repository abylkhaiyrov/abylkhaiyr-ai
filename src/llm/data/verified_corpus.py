from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from llm.data.binary import supervised_token_count, tokenize_sft_split
from llm.data.verified_common import Row, load_config, split_for
from llm.data.verified_knowledge import knowledge_rows
from llm.data.verified_tasks import task_rows
from llm.evaluation.acceptance import load_benchmark
from llm.tokenizer.bpe import BPETokenizer

KAZAKH_ONLY = frozenset("әғқңұһі")
KAZAKH_LETTERS = frozenset("әғқңөұүһі")
UKRAINIAN_ONLY = frozenset("їєґ")
UKRAINIAN_WORDS = frozenset(
    {
        "що",
        "це",
        "який",
        "яка",
        "які",
        "хто",
        "від",
        "також",
        "дуже",
        "після",
        "було",
        "була",
        "були",
        "його",
        "вона",
        "вони",
        "коли",
        "якщо",
        "відкрив",
        "відповідь",
    }
)
KYRGYZ_WORDS = frozenset({"жана", "үчүн", "эмне", "кандай", "жөнүндө", "болот", "менен"})
QUOTED = re.compile(r"«[^»]*»|\"[^\"]*\"|“[^”]*”")
PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
SHORT_TEXT = 40
OPEN_ANSWER_CATEGORIES = frozenset({"list_items"})
SPLITS = ("train", "validation", "test")
CURATED_PATH = Path("configs/curated_dialogues_v5.jsonl")
CODE_PATH = Path("data/v3_1/sft_groups/train.code.jsonl")
CODE_SOURCE = "synthetic_code_write"
CURATED_LICENSE = "MIT (project-authored dialogue)"
CODE_LICENSE = "MIT (project-generated, executable-verified tasks)"


def prompt_key(text: str) -> str:
    cleaned = PUNCTUATION.sub(" ", text.lower().replace("ё", "е"))
    return " ".join(cleaned.split())


def letters_of(text: str) -> list[str]:
    return [char for char in text.lower() if char.isalpha()]


def language_ok(text: str, language: str) -> bool:
    letters = letters_of(QUOTED.sub("", text))
    if not letters:
        return True
    cyrillic = sum("Ѐ" <= char <= "ӿ" for char in letters) / len(letters)
    if language == "en":
        return cyrillic < 0.1
    chars = set(letters)
    words = set(re.findall(r"\w+", text.lower()))
    if cyrillic < 0.5 or chars & UKRAINIAN_ONLY or words & UKRAINIAN_WORDS:
        return False
    if language == "ru":
        return not chars & KAZAKH_LETTERS
    if words & KYRGYZ_WORDS:
        return False
    return bool(chars & KAZAKH_ONLY) or len(letters) < SHORT_TEXT


def instruction_part(row: Row, text: str) -> str:
    return (
        text.split(":")[0]
        if row["category"].startswith(("words_", "sentences_", "phrases_"))
        else text
    )


def language_issues(row: Row) -> list[str]:
    issues: list[str] = []
    for message in row["messages"]:
        user = message["role"] == "user"
        language = row["language"] if user else row["answer_language"]
        if language is None:
            continue
        text = instruction_part(row, message["content"]) if user else message["content"]
        if not language_ok(text, language):
            issues.append(f"language_{message['role']}")
    return issues


def curated_rows() -> list[Row]:
    spec = load_config("knowledge.json")
    rows: list[Row] = []
    for line in CURATED_PATH.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        rows.append(
            {
                "source": "curated_v5",
                "group": "curated",
                "category": item["category"],
                "language": item["language"],
                "answer_language": None if item["category"] == "translate" else item["language"],
                "entity": f"curated:{item['id']}",
                "split": split_for(f"curated:{item['id']}"),
                "provenance": item["provenance"],
                "license": CURATED_LICENSE,
                "messages": item["messages"],
            }
        )
    assert spec
    return rows


def code_language(text: str) -> str:
    letters = letters_of(text)
    cyrillic = sum("Ѐ" <= char <= "ӿ" for char in letters)
    if letters and cyrillic / len(letters) > 0.3:
        return "kk" if "тілінде" in text or set(text.lower()) & KAZAKH_ONLY else "ru"
    return "en"


def code_rows() -> list[Row]:
    rows: list[Row] = []
    for line in CODE_PATH.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        user, answer = item["messages"][0]["content"], item["messages"][1]["content"]
        if item["source"] != CODE_SOURCE or not answer.startswith("```python"):
            continue
        language = code_language(user)
        rows.append(
            {
                "source": "executable_code_v3_1",
                "group": "code",
                "category": "python_function",
                "language": language,
                "answer_language": "en",
                "entity": f"code:{prompt_key(user)}",
                "split": split_for(f"code:{prompt_key(user)}"),
                "provenance": "generated_with_executed_tests",
                "license": CODE_LICENSE,
                "messages": item["messages"],
            }
        )
    return rows


def collect_rows() -> list[Row]:
    return [*knowledge_rows(), *task_rows(), *curated_rows(), *code_rows()]


def user_sequence_key(messages: list[dict[str, str]]) -> str:
    return " || ".join(prompt_key(m["content"]) for m in messages if m["role"] == "user")


def protected_keys(benchmark: Path) -> set[str]:
    items, _ = load_benchmark(benchmark)
    return {" || ".join(prompt_key(turn) for turn in item.turns) for item in items}


def verify(rows: list[Row], protected: set[str]) -> tuple[list[Row], dict[str, int]]:
    rejected: Counter[str] = Counter()
    kept: list[Row] = []
    answers: dict[str, str] = {}
    for row in rows:
        if user_sequence_key(row["messages"]) in protected:
            rejected["benchmark_overlap"] += 1
            continue
        issues = language_issues(row)
        if issues:
            rejected[issues[0]] += 1
            continue
        first = prompt_key(row["messages"][0]["content"])
        final = row["messages"][-1]["content"]
        if prompt_key(final) == first:
            rejected["answer_echoes_question"] += 1
            continue
        context = json.dumps(row["messages"][:-1], ensure_ascii=False)
        open_answer = row["category"] in OPEN_ANSWER_CATEGORIES
        if not open_answer and answers.setdefault(context, final) != final:
            rejected["conflicting_answer"] += 1
            continue
        if any(m["content"] != m["content"].strip() or not m["content"] for m in row["messages"]):
            rejected["malformed_message"] += 1
            continue
        kept.append(row)
    return kept, dict(rejected)


def enforce_split_integrity(rows: list[Row]) -> tuple[list[Row], int]:
    owner: dict[str, str] = {}
    kept: list[Row] = []
    removed = 0
    order = {"test": 0, "validation": 1, "train": 2}
    for row in sorted(rows, key=lambda item: order[item["split"]]):
        key = user_sequence_key(row["messages"])
        if owner.get(key, row["split"]) != row["split"]:
            removed += 1
            continue
        owner[key] = row["split"]
        kept.append(row)
    return kept, removed


def top_share(values: list[str]) -> float:
    return Counter(values).most_common(1)[0][1] / len(values) if values else 0.0


def group_statistics(rows: list[Row]) -> dict[str, Any]:
    buckets: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        buckets[row["group"]].append(row)
    report: dict[str, Any] = {}
    for group, items in sorted(buckets.items()):
        answers = [item["messages"][-1]["content"] for item in items]
        prompts = [prompt_key(item["messages"][0]["content"]) for item in items]
        report[group] = {
            "rows": len(items),
            "languages": dict(Counter(item["language"] for item in items)),
            "splits": dict(Counter(item["split"] for item in items)),
            "unique_prompts": len(set(prompts)),
            "unique_answers": len(set(answers)),
            "top_answer_share": round(top_share(answers), 4),
            "sources": dict(Counter(item["source"] for item in items)),
            "licenses": sorted({item["license"] for item in items}),
        }
    return report


def write_corpus(
    rows: list[Row], out_dir: Path, tokenizer: BPETokenizer
) -> dict[str, dict[str, dict[str, int]]]:
    groups = out_dir / "sft_groups"
    groups.mkdir(parents=True, exist_ok=True)
    buckets: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in rows:
        buckets[(row["group"], row["split"])].append(row)
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for (group, split), items in sorted(buckets.items()):
        path = groups / f"{split}.{group}.jsonl"
        path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
            encoding="utf-8",
        )
        tokenize_sft_split(path, groups / f"{split}.{group}", tokenizer)
        tokens = supervised_token_count(groups / f"{split}.{group}")
        counts.setdefault(group, {})[split] = {"rows": len(items), "supervised_tokens": tokens}
    return counts


def verified_rows(benchmark: Path) -> tuple[list[Row], dict[str, Any]]:
    collected = collect_rows()
    verified, rejected = verify(collected, protected_keys(benchmark))
    final, split_removed = enforce_split_integrity(verified)
    return final, {
        "collected": len(collected),
        "rejected": rejected,
        "removed_for_split_integrity": split_removed,
    }


def build_corpus(
    out_dir: Path, tokenizer: BPETokenizer, benchmark: Path, version: str
) -> dict[str, Any]:
    final, accounting = verified_rows(benchmark)
    manifest = {
        "version": version,
        "benchmark": str(benchmark),
        **accounting,
        "kept": len(final),
        "groups": group_statistics(final),
        "files": write_corpus(final, out_dir, tokenizer),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return manifest
