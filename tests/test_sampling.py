from __future__ import annotations

import json

import numpy as np
import pytest

from llm.core.config import SamplerConfig, TrainConfig
from llm.data.binary import ConversationLoader, tokenize_sft_split
from llm.data.sampling import ExposureCap, capped_labels, family_labels, quotas
from llm.training.runner import exposure_cap

TEMPLATE = "I do not know of a country called {}. I would rather not invent an answer."
OTHER = "Paris is the capital of France."


def refusal_row(name: str) -> dict:
    return {
        "category": "unknown_entity",
        "messages": [
            {"role": "user", "content": f"Who rules {name}?"},
            {"role": "assistant", "content": TEMPLATE.format(name)},
        ],
    }


def answer_row(index: int) -> dict:
    return {
        "category": "qa",
        "messages": [
            {"role": "user", "content": f"Question {index}?"},
            {"role": "assistant", "content": f"{OTHER} Answer {index}."},
        ],
    }


def name_for(index: int) -> str:
    letters = "abcdefghij"
    return f"Zorvan{letters[index % 10]}{letters[index // 10 % 10]}"


def rows(refusals: int = 40, answers: int = 60) -> list[dict]:
    return [refusal_row(name_for(i)) for i in range(refusals)] + [
        answer_row(i) for i in range(answers)
    ]


def test_template_families_ignore_random_names_and_skip_ordinary_answers():
    labels = family_labels(rows(5, 3))
    assert len({label for label in labels if label}) == 1
    assert labels[:5] == [labels[0]] * 5 and labels[5:] == [None] * 3


def test_quotas_limit_every_family_to_the_share_of_the_epoch():
    result = quotas({"a": 400, "b": 300, "c": 5}, others=500, cap=0.1)
    epoch = 500 + sum(result.values())
    assert result["c"] == 5
    assert all(result[name] <= 0.1 * epoch + 1 for name in ("a", "b"))
    assert result["a"] == result["b"] < 300
    assert quotas({"a": 10}, others=500, cap=0.5) == {"a": 10}


def test_selection_rotates_so_every_example_is_used_and_repeats_are_bounded():
    labels = family_labels(rows())
    cap = ExposureCap(labels, 0.1, seed=3)
    assert cap.active and cap.epoch_size < len(labels)
    seen: dict[int, int] = {}
    epochs = 12
    for epoch in range(epochs):
        order = cap.order(epoch)
        assert len(order) == cap.epoch_size and len(set(order.tolist())) == len(order)
        for index in order.tolist():
            seen[index] = seen.get(index, 0) + 1
    assert set(seen) == set(range(len(labels)))
    refusal_counts = [seen[i] for i in range(40)]
    answer_counts = [seen[i] for i in range(40, 100)]
    assert max(refusal_counts) < min(answer_counts) == epochs
    assert max(refusal_counts) - min(refusal_counts) <= 1


def test_selection_is_reproducible_and_depends_on_seed_and_epoch():
    labels = family_labels(rows())
    first, second = ExposureCap(labels, 0.1, 5), ExposureCap(labels, 0.1, 5)
    assert np.array_equal(first.order(4), second.order(4))
    assert not np.array_equal(first.order(4), first.order(5))
    assert not np.array_equal(first.order(4), ExposureCap(labels, 0.1, 6).order(4))


def test_cap_is_inactive_when_no_family_exceeds_the_limit():
    labels = family_labels(rows(2, 100))
    assert not ExposureCap(labels, 0.2, 1).active


def write_group(tmp_path, tokenizer, data):
    source = tmp_path / "train.dialog.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")
    prefix = tmp_path / "train.dialog"
    tokenize_sft_split(source, prefix, tokenizer)
    return prefix


def loader(prefix, tokenizer, cap, batch=4):
    return ConversationLoader(
        prefix, np.dtype(np.uint16), 128, batch, tokenizer.pad_id, 0, True, exposure_cap=cap
    )


def test_inactive_cap_keeps_the_original_order_for_backward_compatibility(tmp_path, tokenizer):
    prefix = write_group(tmp_path, tokenizer, rows(2, 100))
    plain = loader(prefix, tokenizer, None)
    capped = loader(prefix, tokenizer, ExposureCap(capped_labels(prefix), 0.5, 0))
    assert np.array_equal(plain.order, capped.order)
    for _ in range(5):
        left, right = plain.next_batch(), capped.next_batch()
        assert all(a.equal(b) for a, b in zip(left, right, strict=True))


def test_capped_loader_shortens_epochs_and_resumes_exactly(tmp_path, tokenizer):
    prefix = write_group(tmp_path, tokenizer, rows())
    cap = ExposureCap(capped_labels(prefix), 0.1, 0)
    full = loader(prefix, tokenizer, cap)
    assert len(full.order) == cap.epoch_size < full.count
    expected = [full.next_batch() for _ in range(60)]
    first = loader(prefix, tokenizer, cap)
    head = [first.next_batch() for _ in range(23)]
    resumed = loader(prefix, tokenizer, cap)
    resumed.load_state_dict(first.state_dict())
    tail = [resumed.next_batch() for _ in range(37)]
    for left, right in zip(expected, head + tail, strict=True):
        assert all(a.equal(b) for a, b in zip(left, right, strict=True))
    assert full.epoch >= 2


def test_loader_prefix_with_dots_finds_its_label_file(tmp_path, tokenizer):
    prefix = write_group(tmp_path, tokenizer, rows(3, 5))
    assert len(capped_labels(prefix)) == 8


def test_runner_builds_a_cap_only_for_training_when_configured(tmp_path, tokenizer):
    prefix = write_group(tmp_path, tokenizer, rows())
    off = TrainConfig()
    on = TrainConfig(sampler=SamplerConfig(refusal_share_cap=0.1))
    assert exposure_cap(prefix, off, "train", 0) is None
    assert exposure_cap(prefix, on, "validation", 0) is None
    assert exposure_cap(prefix, None, "train", 0) is None
    built = exposure_cap(prefix, on, "train", 0)
    assert built is not None and built.active


def test_sampler_cap_must_be_a_proper_share():
    with pytest.raises(ValueError):
        SamplerConfig(refusal_share_cap=1.0)
    with pytest.raises(ValueError):
        SamplerConfig(refusal_share_cap=0.0)
