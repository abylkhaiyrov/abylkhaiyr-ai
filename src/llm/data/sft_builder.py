from __future__ import annotations

import json
import random
from pathlib import Path

from llm.data.binary import read_jsonl, tokenize_sft_split
from llm.data.codegen import build_code_dataset
from llm.data.dedup import stable_hash
from llm.data.dialogues import (
    MessageTree,
    conversations,
    load_oasst_messages,
    preference_pairs,
    refusal_examples,
)
from llm.data.mathgen import (
    MATH_BENCH_FILE,
    MathProblem,
    build_dataset,
    curriculum_order,
    read_dataset,
    write_dataset,
)
from llm.tokenizer.bpe import BPETokenizer

SPLITS = ("train", "validation", "test")
MATH_WRAPPERS = (
    "{q}",
    "Solve step by step: {q}",
    "Реши задачу и покажи шаги: {q}",
    "Есепті қадамдап шешіп көрсет: {q}",
)
MATH_BENCH_SIZE = 150
BENCH_SEED = 777
TRAIN_SEED = 1
SFT_GROUPS = ("chat", "math", "code")
EVAL_FRACTION = 0.01


def split_of(key: str, seed: int, fraction: float = EVAL_FRACTION) -> str:
    bucket = stable_hash(f"{seed}:{key}") % 10_000 / 10_000
    if bucket < fraction:
        return "test"
    return "validation" if bucket < 2 * fraction else "train"


def build_math_benchmark(benchmark_dir: Path) -> list[MathProblem]:
    path = benchmark_dir / MATH_BENCH_FILE
    if path.is_file():
        return read_dataset(path)
    problems = build_dataset(MATH_BENCH_SIZE, BENCH_SEED)
    write_dataset(problems, path)
    return problems


def math_conversations(
    count: int, benchmark: list[MathProblem], seed: int = TRAIN_SEED
) -> list[dict]:
    rng = random.Random(seed)
    problems = build_dataset(count, seed, exclude={p.key for p in benchmark})
    rows: list[dict] = []
    for problem in curriculum_order(problems, seed):
        question = rng.choice(MATH_WRAPPERS).format(q=problem.question)
        rows.append(
            {
                "source": f"synthetic_math_{problem.category}",
                "group": "math",
                "language": "multi",
                "level": problem.level,
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": problem.solution},
                ],
            }
        )
    return rows


def _prompt_key(row: dict) -> str:
    return row["messages"][0]["content"].strip().lower()


def _dialogue_key(row: dict) -> str:
    return json.dumps(row["messages"], sort_keys=True, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def deduplicate_sft_splits(data_dir: str | Path, tokenizer: BPETokenizer) -> dict[str, int]:
    sft_dir = Path(data_dir) / "sft"
    test_rows = list(read_jsonl(sft_dir / "test.jsonl"))
    val_rows = list(read_jsonl(sft_dir / "validation.jsonl"))
    train_rows = list(read_jsonl(sft_dir / "train.jsonl"))
    test_prompts = {_prompt_key(row) for row in test_rows}
    seen_dialogues = {_dialogue_key(row) for row in test_rows}
    clean_val: list[dict] = []
    val_prompts: set[str] = set()
    for row in val_rows:
        prompt_key = _prompt_key(row)
        dialogue_key = _dialogue_key(row)
        if prompt_key in test_prompts or dialogue_key in seen_dialogues:
            continue
        seen_dialogues.add(dialogue_key)
        val_prompts.add(prompt_key)
        clean_val.append(row)
    blocked = test_prompts | val_prompts
    clean_train: list[dict] = []
    for row in train_rows:
        prompt_key = _prompt_key(row)
        dialogue_key = _dialogue_key(row)
        if prompt_key in blocked or dialogue_key in seen_dialogues:
            continue
        seen_dialogues.add(dialogue_key)
        clean_train.append(row)
    _write_jsonl(sft_dir / "validation.jsonl", clean_val)
    _write_jsonl(sft_dir / "train.jsonl", clean_train)
    for split in SPLITS:
        tokenize_sft_split(sft_dir / f"{split}.jsonl", sft_dir / split, tokenizer)
    return {
        "train": len(clean_train),
        "validation": len(clean_val),
        "test": len(test_rows),
        "removed_train": len(train_rows) - len(clean_train),
        "removed_validation": len(val_rows) - len(clean_val),
    }


def write_group_files(
    rows: list[dict],
    root: Path,
    seed: int,
    keep_order: set[str],
    pinned_test_prompts: set[str] | None = None,
) -> dict:
    counts: dict[str, dict[str, int]] = {}
    buckets: dict[tuple[str, str], list[dict]] = {}
    seen_dialogues: set[str] = set()
    pinned = pinned_test_prompts or set()
    for row in rows:
        dialogue_key = _dialogue_key(row)
        if dialogue_key in seen_dialogues:
            continue
        seen_dialogues.add(dialogue_key)
        prompt_key = _prompt_key(row)
        split = "test" if prompt_key in pinned else split_of(prompt_key, seed)
        buckets.setdefault((row["group"], split), []).append(row)
    for (group, split), items in buckets.items():
        if group not in keep_order:
            random.Random(seed).shuffle(items)
        _write_jsonl(root / f"{split}.{group}.jsonl", items)
        counts.setdefault(group, {})[split] = len(items)
    return counts


def read_pipeline_sft(data_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for split in SPLITS:
        path = data_dir / "sft" / f"{split}.jsonl"
        if path.is_file():
            rows.extend({**row, "group": "chat"} for row in read_jsonl(path))
    return rows


def build_sft_groups(
    data_dir: str | Path,
    tokenizer: BPETokenizer,
    benchmark_dir: str | Path,
    math_count: int,
    code_count: int,
    seed: int = TRAIN_SEED,
    include_oasst: bool = True,
) -> dict:
    root = Path(data_dir)
    out = root / "sft_groups"
    out.mkdir(parents=True, exist_ok=True)
    benchmark = build_math_benchmark(Path(benchmark_dir))
    test_sft_path = root / "sft" / "test.jsonl"
    pinned_test_prompts = (
        {_prompt_key(row) for row in read_jsonl(test_sft_path)}
        if test_sft_path.is_file()
        else set()
    )
    rows = read_pipeline_sft(root)
    tree = None
    if include_oasst:
        tree = MessageTree(load_oasst_messages())
        rows.extend(conversations(tree))
    rows.extend(refusal_examples(seed, repeats=1))
    rows.extend(math_conversations(math_count, benchmark, seed))
    rows.extend(build_code_dataset(code_count, seed))
    counts = write_group_files(
        rows, out, seed, keep_order={"math"}, pinned_test_prompts=pinned_test_prompts
    )
    if tree is not None:
        pairs = list(preference_pairs(tree))
        pref = root / "preference"
        pref.mkdir(exist_ok=True)
        buckets: dict[str, list[dict]] = {"train": [], "validation": []}
        seen_pairs: set[str] = set()
        for pair in pairs:
            prompt_key = pair["chosen"][0]["content"].strip().lower()
            if prompt_key in pinned_test_prompts:
                continue
            pair_key = json.dumps(
                {"chosen": pair["chosen"], "rejected": pair["rejected"]},
                sort_keys=True,
                ensure_ascii=False,
            )
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            split = "validation" if split_of(prompt_key, seed, 0.03) != "train" else "train"
            buckets[split].append(pair)
        for split, items in buckets.items():
            _write_jsonl(pref / f"{split}.jsonl", items)
        counts["preference"] = {split: len(items) for split, items in buckets.items()}
    for group in SFT_GROUPS:
        for split in SPLITS:
            source = out / f"{split}.{group}.jsonl"
            if source.is_file():
                tokenize_sft_split(source, out / f"{split}.{group}", tokenizer)
    return counts
