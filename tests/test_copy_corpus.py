from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from llm.core.config import ModelConfig
from llm.data.copy_corpus import (
    EVAL_ONLY,
    PRIMARY,
    WEIGHTS,
    CorpusCase,
    build_eval,
    build_train,
    category_plan,
    exclusions,
    manifest_of,
    training_rows,
    value_split,
    write_corpus,
)
from llm.data.copy_ladder import CATEGORIES, LANGUAGES, copy_cases, doc_tokens, training_sets
from llm.data.copy_text import (
    FRAMES_DEV,
    FRAMES_HOLDOUT,
    FRAMES_TRAIN,
    INSTRUCTIONS_DEV,
    INSTRUCTIONS_HOLDOUT,
    INSTRUCTIONS_TRAIN,
    QUESTIONS_DEV,
    QUESTIONS_HOLDOUT,
    QUESTIONS_TRAIN,
)
from llm.evaluation.copying import evaluate
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM
from llm.tokenizer.bpe import BPETokenizer

TOKENIZER = Path("data/tokenizer")


@pytest.fixture(scope="module")
def real() -> BPETokenizer:
    if not TOKENIZER.is_dir():
        pytest.skip("the project tokenizer is not available")
    return BPETokenizer.load(TOKENIZER)


@pytest.fixture(scope="module")
def ladder(real: BPETokenizer):
    return copy_cases(real)


@pytest.fixture(scope="module")
def earlier(real: BPETokenizer, ladder) -> list[str]:
    sets = training_sets(real, ladder)
    return [case.target for name in ("copy648", "validation") for case in sets[name]]


@pytest.fixture(scope="module")
def excluded(ladder, earlier):
    return exclusions(ladder, earlier)


@pytest.fixture(scope="module")
def corpus(real: BPETokenizer, excluded):
    return {
        "train": build_train(real, excluded, count=1200),
        "dev": build_eval(real, "dev", excluded),
        "holdout": build_eval(real, "holdout", excluded),
    }


def test_generation_is_deterministic(real, excluded) -> None:
    assert build_train(real, excluded, count=300) == build_train(real, excluded, count=300)
    assert build_eval(real, "dev", excluded) == build_eval(real, "dev", excluded)


def test_target_values_never_cross_the_train_dev_holdout_or_regression_boundaries(
    corpus, ladder, earlier
) -> None:
    targets = {name: {c.target for c in cases} for name, cases in corpus.items()}
    assert not targets["train"] & targets["dev"]
    assert not targets["train"] & targets["holdout"]
    assert not targets["dev"] & targets["holdout"]
    regression = {c.target for c in ladder} | set(earlier)
    for name in ("dev", "holdout"):
        assert not targets[name] & regression
    assert all(value_split(v) == "train" for v in targets["train"])
    assert all(value_split(v) == "dev" for v in targets["dev"])
    assert all(value_split(v) == "holdout" for v in targets["holdout"])


def test_training_corpus_covers_every_category_language_and_level(corpus) -> None:
    train = corpus["train"]
    counts = Counter(c.category for c in train)
    assert set(counts) == set(CATEGORIES) and not set(counts) & set(EVAL_ONLY)
    assert {c.language for c in train} == set(LANGUAGES) and {c.level for c in train} == {1, 2}
    assert len({c.prompt for c in train}) == len(train)
    for category in PRIMARY:
        assert counts[category] / len(train) >= 0.15
    plan = category_plan(1000, WEIGHTS)
    assert Counter(plan)["long"] == 260 and len(plan) == 1000


def test_prompts_vary_in_structure_not_only_in_the_value(corpus) -> None:
    train = [c for c in corpus["train"] if c.level == 2]
    skeletons = {c.prompt.replace(c.target, "_") for c in train}
    assert len(skeletons) > 0.8 * len(train)
    positions = {c.prompt.find(c.target) / len(c.prompt) < 0.4 for c in train}
    assert positions == {True, False}
    assert {c.noun for c in train} >= {"river", "mountain"} or len({c.noun for c in train}) >= 8


def test_evaluation_templates_are_disjoint_from_training_templates() -> None:
    for train, dev, holdout in (
        (INSTRUCTIONS_TRAIN, INSTRUCTIONS_DEV, INSTRUCTIONS_HOLDOUT),
        (FRAMES_TRAIN, FRAMES_DEV, FRAMES_HOLDOUT),
        (QUESTIONS_TRAIN, QUESTIONS_DEV, QUESTIONS_HOLDOUT),
    ):
        for language in LANGUAGES:
            assert not set(train[language]) & set(dev[language])
            assert not set(train[language]) & set(holdout[language])
            assert not set(dev[language]) & set(holdout[language])


def test_evaluation_cells_have_twins_new_categories_and_cross_script_values(corpus, real) -> None:
    dev = corpus["dev"]
    cells = Counter(c.cell for c in dev)
    assert set(cells) == {"seen", "wording", "category", "cross"}
    assert {c.category for c in dev if c.cell == "category"} == set(EVAL_ONLY)
    twins: dict[str, list[CorpusCase]] = {}
    for case in dev:
        if case.cell == "seen" and case.level == 2:
            twins.setdefault(case.family, []).append(case)
    assert twins and all(len(v) == 2 and v[0].target != v[1].target for v in twins.values())
    cyrillic = [
        c for c in dev if c.cell == "cross" and c.language == "en" and c.category != "number"
    ]
    assert cyrillic and all(any("\u0400" <= ch <= "\u04ff" for ch in c.target) for c in cyrillic)
    foreign = [c for c in dev if c.cell == "cross" and c.language in ("ru", "kk")]
    assert any(c.target.isascii() for c in foreign)
    assert all(doc_tokens(real, c.target) >= 1 for c in dev[:50])


def test_training_rows_follow_the_answer_formats(corpus) -> None:
    rows = training_rows(corpus["train"])
    assert len(rows) == len(corpus["train"])
    for row, case in zip(rows, corpus["train"], strict=True):
        answer = row["messages"][1]["content"]
        assert case.target in answer
        if case.level == 1:
            assert answer == case.target


def test_manifest_counts_overlap_and_token_sharing(real, corpus) -> None:
    manifest = manifest_of(corpus["train"], corpus["dev"], corpus["holdout"], real)
    assert manifest["target_overlap"] == {"dev": 0, "holdout": 0}
    assert manifest["dev_holdout_target_overlap"] == 0
    assert 0.3 < manifest["shared_bpe_token_share"]["dev"] <= 1.0
    assert manifest["train"]["examples"] == 1200


def test_written_corpus_is_tokenized_and_reproducible(tmp_path, real, ladder, earlier) -> None:
    first = write_corpus(tmp_path / "one", real, ladder, earlier, 200)
    second = write_corpus(tmp_path / "two", real, ladder, earlier, 200)
    assert first == second and first["train"]["examples"] > 150
    directory = tmp_path / "one" / "B" / "sft"
    assert (directory / "train.ids.bin").stat().st_size > 0
    assert (tmp_path / "one" / "dev.jsonl").read_bytes() == (
        tmp_path / "two" / "dev.jsonl"
    ).read_bytes()
    assert (tmp_path / "one" / "holdout.jsonl").is_file() and (
        tmp_path / "one" / "trained_B.jsonl"
    ).is_file()


def test_suite_reports_cells_first_token_accuracy_twins_and_chat_space(
    tmp_path, real, corpus
) -> None:
    subset = [c for c in corpus["dev"] if c.language == "ru" and c.category in ("long", "number")][
        :80
    ]
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in subset), encoding="utf-8"
    )
    config = ModelConfig(
        vocab_size=real.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=256,
    )
    engine = InferenceEngine(CausalLM(config), real, device=torch.device("cpu"))
    output = evaluate(
        engine, {"file": str(path), "formats": ["chat", "chat_space"], "framed": True}
    )
    names = set(output.metrics)
    assert "copy.chat.exact.L1.cell.seen.primary" in names
    assert "copy.chat.tf_first_acc.L1.cell.seen" in names
    assert "copy.chat.consistent.L2" in names
    assert "copy.chat_space.tf_seq.L1" in names and "copy.chat_space.exact.L1" not in names
    assert all(r["format"] in ("chat", "chat_space") for r in output.records)
