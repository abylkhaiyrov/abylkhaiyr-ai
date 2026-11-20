from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from llm.data.binary import read_jsonl
from llm.data.preference import PREFERENCE_DIR, tokenize_preference_split
from llm.data.sft_builder import SPLITS
from llm.data.sft_quality import normalize
from llm.reasoning.tools import extract_final_answer
from llm.tokenizer.bpe import BPETokenizer

PROMPT_GROUPS = (
    "skills",
    "dialog",
    "identity",
    "reading_en",
    "reading_ru",
    "reading_kk",
    "word_math",
)
FINAL_ANSWER_GROUPS = frozenset({"word_math"})
IMITATION_FOREIGN_IDENTITY = ("ChatGPT", "OpenAI GPT", "Gemini", "Claude")
NUMBER = re.compile(r"\d+")
SENTENCE = re.compile(r"(?<=[.!?])\s+")
MIN_TRUNCATE_CHARS = 40
MAX_PROMPT_CHARS = 700
Sampler = Callable[[list[dict[str, str]]], str]
Corruption = Callable[[random.Random, dict, list[dict]], str | None]


@dataclass
class PreferenceBuildConfig:
    data_dir: str = "data/v3"
    human_pairs_dir: str = "data/v2"
    pairs_per_group: int = 1500
    human_pairs: int = 600
    validation_fraction: float = 0.05
    seed: int = 1
    source_split: str = "train"


def last_answer(row: dict) -> str:
    return row["messages"][-1]["content"]


def prompt_messages(row: dict) -> list[dict[str, str]]:
    return row["messages"][:-1]


def swap_answer(rng: random.Random, row: dict, pool: list[dict]) -> str | None:
    other = rng.choice(pool)
    return (
        last_answer(other) if normalize(last_answer(other)) != normalize(last_answer(row)) else None
    )


def repeat_loop(rng: random.Random, row: dict, pool: list[dict]) -> str | None:
    words = last_answer(row).split()
    if len(words) < 2:
        return None
    unit = " ".join(words[: rng.randint(1, min(4, len(words)))])
    return " ".join([unit] * rng.randint(4, 8))


def truncate(rng: random.Random, row: dict, pool: list[dict]) -> str | None:
    answer = last_answer(row)
    if len(answer) < MIN_TRUNCATE_CHARS:
        return None
    cut = answer[: rng.randint(len(answer) // 4, len(answer) // 2)].rstrip()
    return cut or None


def wrong_number(rng: random.Random, row: dict, pool: list[dict]) -> str | None:
    answer = last_answer(row)
    matches = list(NUMBER.finditer(answer))
    if not matches:
        return None
    target = rng.choice(matches)
    value = int(target.group()) + rng.choice((-3, -2, -1, 1, 2, 5, 10))
    return answer[: target.start()] + str(max(value, 0)) + answer[target.end() :]


def wrong_language(rng: random.Random, row: dict, pool: list[dict]) -> str | None:
    language = row.get("language")
    candidates = [
        item
        for item in pool
        if item.get("category") == row.get("category") and item.get("language") != language
    ]
    return last_answer(rng.choice(candidates)) if candidates and language else None


def foreign_identity(rng: random.Random, row: dict, pool: list[dict]) -> str | None:
    answer = last_answer(row)
    if "Abylkhaiyr AI" not in answer:
        return None
    return answer.replace("Abylkhaiyr AI", rng.choice(IMITATION_FOREIGN_IDENTITY))


CORRUPTIONS: tuple[Corruption, ...] = (
    swap_answer,
    repeat_loop,
    truncate,
    wrong_number,
    wrong_language,
    foreign_identity,
)


def corrupted_answer(rng: random.Random, row: dict, pool: list[dict]) -> str | None:
    for corruption in rng.sample(CORRUPTIONS, len(CORRUPTIONS)):
        candidate = corruption(rng, row, pool)
        if candidate and normalize(candidate) != normalize(last_answer(row)):
            return candidate
    return None


def make_pair(row: dict, rejected: str, kind: str) -> dict:
    prompt = prompt_messages(row)
    return {
        "source": f"preference_{kind}",
        "kind": kind,
        "chosen": [*prompt, {"role": "assistant", "content": last_answer(row)}],
        "rejected": [*prompt, {"role": "assistant", "content": rejected}],
    }


def usable_prompt(row: dict) -> bool:
    return sum(len(m["content"]) for m in row["messages"][:-1]) <= MAX_PROMPT_CHARS


def final_answer(text: str) -> str | None:
    value = extract_final_answer(text)
    return None if value is None else normalize(value)


def is_verified_wrong(row: dict, sample: str) -> bool:
    if row.get("group") in FINAL_ANSWER_GROUPS:
        reference, produced = final_answer(last_answer(row)), final_answer(sample)
        return reference is not None and produced != reference
    return normalize(sample) != normalize(last_answer(row))


def synthetic_pairs(
    rows: list[dict], count: int, rng: random.Random, sampler: Sampler | None = None
) -> list[dict]:
    pool = [row for row in rows if usable_prompt(row)]
    chosen_rows = rng.sample(pool, min(count, len(pool)))
    pairs: list[dict] = []
    for row in chosen_rows:
        rejected, kind = None, "corruption"
        if sampler is not None:
            sample = sampler(prompt_messages(row)).strip()
            if sample and is_verified_wrong(row, sample):
                rejected, kind = sample, "on_policy"
        if rejected is None:
            rejected = corrupted_answer(rng, row, pool)
        if rejected is not None:
            pairs.append(make_pair(row, rejected, kind))
    return pairs


def human_pairs(directory: Path, count: int, rng: random.Random) -> list[dict]:
    path = directory / PREFERENCE_DIR / "train.jsonl"
    if not path.is_file() or count <= 0:
        return []
    rows = list(read_jsonl(path))
    picked = rng.sample(rows, min(count, len(rows)))
    return [
        {
            "source": "preference_human",
            "kind": "human",
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }
        for r in picked
    ]


def load_group_rows(data_dir: Path, group: str, split: str) -> list[dict]:
    path = data_dir / "sft_groups" / f"{split}.{group}.jsonl"
    return list(read_jsonl(path)) if path.is_file() else []


def write_pairs(root: Path, split: str, pairs: Iterable[dict]) -> int:
    target = root / PREFERENCE_DIR
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    with (target / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_preference_corpus(
    config: PreferenceBuildConfig, tokenizer: BPETokenizer, sampler: Sampler | None = None
) -> dict[str, object]:
    root = Path(config.data_dir)
    rng = random.Random(config.seed)
    pairs: list[dict] = []
    for group in PROMPT_GROUPS:
        rows = load_group_rows(root, group, config.source_split)
        if rows:
            pairs.extend(synthetic_pairs(rows, config.pairs_per_group, rng, sampler))
    pairs.extend(human_pairs(Path(config.human_pairs_dir), config.human_pairs, rng))
    rng.shuffle(pairs)
    held_out = max(int(len(pairs) * config.validation_fraction), 1)
    counts = {
        "validation": write_pairs(root, "validation", pairs[:held_out]),
        "train": write_pairs(root, "train", pairs[held_out:]),
    }
    for split in ("train", "validation"):
        tokenize_preference_split(root, split, tokenizer)
    kinds: dict[str, int] = {}
    for pair in pairs:
        kinds[pair["kind"]] = kinds.get(pair["kind"], 0) + 1
    return {"pairs": counts, "kinds": kinds, "splits": list(SPLITS)}
