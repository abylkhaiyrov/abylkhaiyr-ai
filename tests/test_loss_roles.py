from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm.core.config import ModelConfig
from llm.data.grounded import REFUSAL, row_of, training_examples
from llm.data.tokenizer_audit import encoded
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM
from llm.tokenizer.bpe import BPETokenizer
from llm.training.loss_roles import ROLES, role_losses, row_roles, spans_of, token_roles

TOKENIZER = Path("data/tokenizer")


@pytest.fixture()
def real() -> BPETokenizer:
    if not TOKENIZER.is_dir():
        pytest.skip("the project tokenizer is not available")
    return BPETokenizer.load(TOKENIZER)


@pytest.fixture()
def engine(real: BPETokenizer) -> InferenceEngine:
    config = ModelConfig(
        vocab_size=real.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
    )
    return InferenceEngine(CausalLM(config), real, device=torch.device("cpu"))


def test_spans_find_every_occurrence_without_overlap() -> None:
    assert spans_of("ab ab ab", "ab") == [(0, 2), (3, 5), (6, 8)]
    assert spans_of("abc", "") == [] and spans_of("abc", "z") == []


def first_example(answerable: bool):
    for example in training_examples("en", 80, 1, missing_share=0.25):
        if (example.plan.expects == REFUSAL) != answerable and example.plan.task == "extract":
            return example
    raise AssertionError


def test_roles_separate_template_entity_value_and_refusal(real) -> None:
    example = first_example(True)
    answer = f"The river of {example.plan.entity} is {example.plan.answer}."
    _, offsets = encoded(real, answer)
    roles = token_roles(example, answer, offsets)
    assert {"template", "entity", "value"} <= set(roles)
    pieces = [answer[a:b] for (a, b), role in zip(offsets, roles, strict=True) if role == "value"]
    assert "".join(pieces).replace(" ", "") == example.plan.answer
    refusal = first_example(False)
    text = "The text does not say."
    assert set(token_roles(refusal, text, encoded(real, text)[1])) == {"refusal"}


def test_row_losses_are_aligned_with_roles_and_end_with_eos(engine) -> None:
    example = first_example(True)
    row = row_of(example, 0)
    roles, losses, hits = row_roles(engine, row, example)
    assert len(roles) == len(losses) == len(hits) and roles[-1] == "eos"
    assert all(value > 0 for value in losses) and set(roles) <= set(ROLES)


def test_role_losses_report_every_role_and_value_exactness(tmp_path, engine) -> None:
    examples = training_examples("en", 60, 1, missing_share=0.25)
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row_of(e, 0)) + "\n" for e in examples), encoding="utf-8")
    report = role_losses(engine, [path], 1, 0.25, 4)
    assert {"template", "value", "eos", "refusal"} <= set(report)
    assert report["value_first_loss"] > 0 and 0.0 <= report["value_exact_teacher_forced"] <= 1.0
