from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from llm.data.binary import IGNORE_INDEX
from llm.data.sft_profile import entropy, intent_of, profile_sft, script_of


def write_group(directory: Path, name: str, rows: list[dict], supervised: list[int]) -> None:
    (directory / f"train.{name}.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    lengths = [count + 2 for count in supervised]
    labels = np.concatenate(
        [np.r_[np.full(2, IGNORE_INDEX), np.ones(count)] for count in supervised]
    ).astype(np.int32)
    labels.tofile(directory / f"train.{name}.labels.bin")
    np.zeros(sum(lengths), dtype=np.uint16).tofile(directory / f"train.{name}.ids.bin")
    np.save(directory / f"train.{name}.lengths.npy", np.asarray(lengths, dtype=np.int64))


def row(prompt: str, answer: str, **extra: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        **extra,
    }


def test_script_detection_separates_three_languages() -> None:
    assert script_of("What is the capital?") == "en"
    assert script_of("Какая столица страны?") == "ru"
    assert script_of("Қазақстанның астанасы қандай?") == "kk"
    assert script_of("123") == "none"


def test_entropy_of_uniform_and_degenerate_counts() -> None:
    from collections import Counter

    assert entropy(Counter({"a": 1, "b": 1})) == pytest.approx(1.0)
    assert entropy(Counter({"a": 5})) == 0.0
    assert entropy(Counter()) == 0.0


def test_intent_prefers_category_over_leading_words() -> None:
    assert intent_of({**row("Hello there friend", "Hi"), "category": "greeting"}) == "greeting"
    assert intent_of(row("Hello there friend again", "Hi")) == "hello there friend"


def test_profile_counts_tokens_templates_and_conflicts(tmp_path: Path) -> None:
    write_group(
        tmp_path,
        "alpha",
        [
            row("What is two plus two?", "Four.", source="synthetic_math"),
            row("What is two plus two?", "Five.", source="synthetic_math"),
            row("Name a colour", "Red.", source="other"),
        ],
        [3, 4, 5],
    )
    write_group(tmp_path, "beta", [row("Привет, как дела", "Хорошо.")], [6])
    report = profile_sft(tmp_path)
    alpha = report["groups"]["alpha"]
    assert alpha["examples"] == 3 and alpha["supervised_tokens"] == 12
    assert alpha["unique_prompts"] == 2 and alpha["conflicting_prompts"] == 1
    assert alpha["verified_synthetic_tokens"] == 7
    assert alpha["languages"] == {"en": 3}
    assert report["groups"]["beta"]["languages"] == {"ru": 1}
    assert report["supervised_tokens"] == 18
    assert report["groups"]["alpha"]["token_share"] == pytest.approx(12 / 18)
