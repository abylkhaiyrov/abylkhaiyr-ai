from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from llm.data.dialogue_pretraining import StreamFormat, document_ids, write_dialogue_stream
from llm.evaluation.diagnostic import Scored, option_log_probs, score_item, summarize
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM
from llm.tokenizer.bpe import Message

CONVERSATION = [Message("user", "How are you today?"), Message("assistant", "I am fine, thanks.")]


def test_dialogue_format_keeps_roles_and_ends_every_message(tokenizer) -> None:
    ids = document_ids(CONVERSATION, tokenizer, StreamFormat.DIALOGUE)
    assert ids[0] == tokenizer.bos_id
    assert ids.count(tokenizer.eos_id) == len(CONVERSATION)
    assert tokenizer.role_ids["user"] in ids and tokenizer.role_ids["assistant"] in ids


def test_plain_format_has_same_content_without_role_tokens(tokenizer) -> None:
    ids = document_ids(CONVERSATION, tokenizer, StreamFormat.PLAIN)
    assert ids[0] == tokenizer.bos_id and ids[-1] == tokenizer.eos_id
    assert not set(tokenizer.role_ids.values()) & set(ids)
    assert ids.count(tokenizer.eos_id) == 1
    text = tokenizer.decode(ids, skip_special_tokens=True)
    assert "How are you today?" in text and "I am fine, thanks." in text


def test_written_stream_length_matches_documents(tmp_path, tokenizer) -> None:
    source = tmp_path / "rows.jsonl"
    rows = [{"messages": [{"role": m.role, "content": m.content} for m in CONVERSATION]}] * 3
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    target = tmp_path / "train.dialog.bin"
    count = write_dialogue_stream([source], target, tokenizer, StreamFormat.DIALOGUE)
    assert count == 3 * len(document_ids(CONVERSATION, tokenizer, StreamFormat.DIALOGUE))
    assert np.memmap(target, dtype=np.uint16, mode="r").shape[0] == count


def test_option_scores_cover_every_option_token(model_config, tokenizer) -> None:
    engine = InferenceEngine(CausalLM(model_config), tokenizer, device=torch.device("cpu"))
    options = ["a short answer", "another much longer answer here"]
    scores = option_log_probs(engine, "The answer is", options)
    assert [count for _, count in scores] == [len(tokenizer.encode(f" {o}")) for o in options]
    assert all(value < 0 for value, _ in scores)


def test_score_item_reports_chance_from_option_count(model_config, tokenizer) -> None:
    engine = InferenceEngine(CausalLM(model_config), tokenizer, device=torch.device("cpu"))
    item = {
        "id": "x",
        "language": "en",
        "task": "t",
        "group": "g",
        "prefix": "Water is",
        "options": ["wet", "dry", "loud", "blue"],
        "answer": 0,
    }
    assert score_item(engine, item).chance == 0.25


def test_summary_requires_interval_above_chance() -> None:
    wins = [Scored(f"a{i}", "en", "t", "g", 0.25, True, True) for i in range(15)]
    mixed = [Scored(f"b{i}", "en", "t", "g", 0.25, i < 5, i < 5) for i in range(15)]
    assert summarize(wins)["overall"]["above_chance"] is True
    assert summarize(mixed)["overall"]["above_chance"] is False


def test_diagnostic_file_is_balanced_and_well_formed() -> None:
    rows = [
        json.loads(line) for line in Path("configs/diagnostic_v1.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len({row["id"] for row in rows}) == 150
    assert set(Counter(row["language"] for row in rows).values()) == {50}
    assert all(0 <= row["answer"] < len(row["options"]) == len(set(row["options"])) for row in rows)
    position = Counter(row["answer"] for row in rows)
    assert min(position.values()) >= 20


@pytest.mark.parametrize("category", ["greeting", "context_multi", "unknown_entity", "support"])
def test_ability_groups_cover_every_validation_category(category) -> None:
    groups = json.loads(Path("configs/ability_groups_v5.json").read_text())
    assert sum(category in members for members in groups.values()) == 1
