from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from llm.core.config import ModelConfig
from llm.data.copy_ladder import (
    CATEGORIES,
    LANGUAGES,
    LEVELS,
    LONG_TOKENS,
    copy_cases,
    doc_tokens,
    write_copy_cases,
)
from llm.evaluation.copying import evaluate, forced, generate, judged, probe_of
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM
from llm.tokenizer.bpe import BPETokenizer

TOKENIZER = Path("data/tokenizer")


def test_ladder_is_deterministic_balanced_and_independent(real) -> None:
    first = copy_cases(real, per_cell=3)
    assert first == copy_cases(real, per_cell=3)
    assert len(first) == len(LANGUAGES) * len(CATEGORIES) * len(LEVELS) * 3
    assert len({c.id for c in first}) == len(first)
    for case in first:
        assert case.target in case.prompt and case.target not in case.rivals
        assert len(set(case.rivals)) == len(case.rivals) <= 3


def test_value_categories_respect_their_token_counts(real) -> None:
    cases = copy_cases(real, per_cell=3)
    for case in cases:
        size = doc_tokens(real, case.target)
        if case.category == "double":
            assert size == 2
        if case.category == "long":
            assert size >= LONG_TOKENS
        if case.category == "single" and case.language == "en":
            assert size == 1


def test_levels_ask_for_the_right_value(real) -> None:
    for case in copy_cases(real, per_cell=4):
        if case.level == 3:
            assert len(case.rivals) == 1 and case.entity in ("A", "B")
        if case.level == 4:
            assert len(case.rivals) == 3 and case.relation in ("river", "mountain")


def test_judging_separates_exact_contains_and_rival_answers(real) -> None:
    case = next(c for c in copy_cases(real, per_cell=3) if c.level == 3)
    value, rival = case.target, case.rivals[0]
    assert judged(case, f" {value}.")["exact"]
    assert (
        judged(case, f"It is {value}")["contains"] and not judged(case, f"It is {value}")["exact"]
    )
    verdict = judged(case, f"{rival}")
    assert verdict["rival"] and not verdict["contains"]
    assert not judged(case, f"{value} {rival}")["contains"]


def test_chat_and_plain_forms_use_different_leading_space_variants(real, real_config) -> None:
    engine = InferenceEngine(CausalLM(real_config), real, device=torch.device("cpu"))
    case = next(
        c
        for c in copy_cases(real, per_cell=3)
        if c.level == 1 and c.language == "en" and c.category == "invented"
    )
    chat = probe_of(engine, case, "chat")
    plain = probe_of(engine, case, "plain")
    assert chat.target_ids == real.encode(case.target)
    assert plain.target_ids == real.encode(f" {case.target}")
    assert chat.target_ids[0] != plain.target_ids[0]
    assert chat.prompt_ids[-1] == real.role_ids["assistant"]


def test_teacher_forcing_matches_the_model_distribution(real, real_config) -> None:
    engine = InferenceEngine(CausalLM(real_config), real, device=torch.device("cpu"))
    engine.model.eval()
    case = copy_cases(real, per_cell=2)[0]
    probe = probe_of(engine, case, "plain")
    score = forced(engine, probe)
    sequence = torch.tensor([[*probe.prompt_ids, *probe.target_ids][:-1]])
    with torch.no_grad():
        logits = engine.model(sequence).float()[0, len(probe.prompt_ids) - 1]
    expected = torch.softmax(logits, -1)[probe.target_ids[0]].item()
    assert abs(score.first_probability - expected) < 1e-5
    assert 0.0 <= score.token_accuracy <= 1.0
    text, finish = generate(engine, case, probe, "plain")
    assert isinstance(text, str) and finish in ("length", "stop")


def test_suite_reports_free_and_forced_metrics_per_scope(tmp_path, real, real_config) -> None:
    cases = write_copy_cases(tmp_path / "cases.jsonl", real)
    subset = [c for c in cases if c.language == "en" and c.category in ("single", "invented")][:24]
    path = tmp_path / "subset.jsonl"
    path.write_text(
        "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in subset), encoding="utf-8"
    )
    engine = InferenceEngine(CausalLM(real_config), real, device=torch.device("cpu"))
    output = evaluate(engine, {"file": str(path)})
    names = set(output.metrics)
    for form in ("chat", "plain"):
        assert f"copy.{form}.exact.L1" in names and f"copy.{form}.tf_seq.L2.single" in names
        assert f"copy.{form}.tf_loss_first.L1" in names and f"copy.{form}.id_first.L1" in names
    assert "copy.chat.tf_eos.L1" in names and "copy.plain.tf_eos.L1" not in names
    assert output.metrics["copy.chat.id_first.L1"] <= output.metrics["copy.plain.id_first.L1"]
    assert all(0.0 <= value for value in output.metrics.values())
    assert {r["format"] for r in output.records} == {"chat", "plain"}


@pytest.fixture()
def real() -> BPETokenizer:
    if not TOKENIZER.is_dir():
        pytest.skip("the project tokenizer is not available")
    return BPETokenizer.load(TOKENIZER)


@pytest.fixture()
def real_config(real: BPETokenizer) -> ModelConfig:
    return ModelConfig(
        vocab_size=real.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=256,
    )


def test_tokenizer_audit_reports_value_statistics_roundtrip_and_identifier_match(real) -> None:
    from llm.data.tokenizer_audit import (
        audit,
        identifier_match,
        occurrences,
        roundtrip,
        value_statistics,
    )

    cases = copy_cases(real, per_cell=2)
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Pelikaon: river — Zabatuis. Pelikaon: river?"},
                {"role": "assistant", "content": "Zabatuis."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Pelikaon: river — Zabatuis. Pelikaon: river?"},
                {"role": "assistant", "content": "The river of Pelikaon is Zabatuis."},
            ]
        },
    ]
    match = identifier_match(real, rows)
    assert match["answer_start"]["occurrences"] == 1 and match["after_text"]["occurrences"] == 2
    assert match["after_text"]["match_share"] == 0.5
    assert match["answer_start"]["match_share"] == 0.0
    assert roundtrip(real, ["Мерилон қаласы 123", "Merilon"]) == {
        "checked": 2,
        "mismatches": 0,
        "case_changes": 0,
        "invalid_utf8": 0,
    }
    found = occurrences(real, "a Zabatuis b", ["Zabatuis"])
    assert found["Zabatuis"].start == 2 and len(found["Zabatuis"].ids) >= 2
    stats = value_statistics(real, {"names": ["Zabatuis", "Pelikaon"]})["names"]
    assert stats["multi_token_share"] == 1.0 and stats["single_token_share"] == 0.0
    report = audit(real, cases, rows)
    assert report["roundtrip"]["mismatches"] == 0 and report["vocabulary"] == real.vocab_size


def test_framed_probe_keeps_leading_space_and_matches_document_identifiers(
    real, real_config
) -> None:
    from llm.data.copy_ladder import answer_frame

    engine = InferenceEngine(CausalLM(real_config), real, device=torch.device("cpu"))
    case = next(
        c
        for c in copy_cases(real, per_cell=3)
        if c.level == 3 and c.language == "ru" and c.category == "invented"
    )
    assert answer_frame(case).startswith("Река города") and case.entity in answer_frame(case)
    chat = probe_of(engine, case, "chat", framed=True)
    plain = probe_of(engine, case, "plain", framed=True)
    assert chat.target_ids == plain.target_ids == real.encode(f" {case.target}")
    assert not chat.eos and not plain.eos
    assert forced(engine, chat).id_first and forced(engine, plain).id_first
    assert answer_frame(next(c for c in copy_cases(real, per_cell=3) if c.level == 1)) == ""


def test_suite_adds_framed_metrics_only_when_requested(tmp_path, real, real_config) -> None:
    cases = write_copy_cases(tmp_path / "cases.jsonl", real)
    scoped = [c for c in cases if c.language == "ru" and c.category == "invented"]
    subset = [c for level in (1, 2, 4) for c in scoped if c.level == level][::3]
    path = tmp_path / "subset.jsonl"
    path.write_text(
        "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in subset), encoding="utf-8"
    )
    engine = InferenceEngine(CausalLM(real_config), real, device=torch.device("cpu"))
    plain = evaluate(engine, {"file": str(path), "formats": ["plain"]})
    framed = evaluate(engine, {"file": str(path), "formats": ["plain"], "framed": True})
    assert not any(".fr_" in name for name in plain.metrics)
    assert (
        "copy.plain.fr_seq.L2" in framed.metrics and "copy.plain.fr_loss_first.L4" in framed.metrics
    )
    assert not any(name.endswith(".L1") and ".fr_" in name for name in framed.metrics)


def test_training_sets_are_nested_disjoint_from_evaluation_and_permute_trained_values(real) -> None:
    from llm.data.copy_ladder import TRAIN_CATEGORIES, training_rows, training_sets

    evaluation = copy_cases(real)
    sets = training_sets(real, evaluation)
    small = {c.target for c in sets["copy36"]}
    full = {c.target for c in sets["copy648"]}
    held = {c.target for c in evaluation} | {c.target for c in sets["validation"]}
    assert len(sets["copy36"]) == 36 and len(sets["copy648"]) == 648
    assert small <= full and not (full & {c.target for c in evaluation})
    assert not {c.target for c in sets["validation"]} & full
    assert held and {c.category for c in sets["copy648"]} == set(TRAIN_CATEGORIES)
    assert {c.level for c in sets["copy648"]} == {1, 2}
    assert {c.target for c in sets["permuted"]} <= small
    for source in sets["permuted"]:
        assert source.target in source.prompt and source.id.startswith("permuted/")
    rows = training_rows(sets["copy36"])
    assert all(r["messages"][0]["role"] == "user" and r["messages"][1]["content"] for r in rows)
    assert {r["category"] for r in rows} == {"copy_l1", "copy_l2"}


def test_written_training_sets_are_tokenized_and_consistent(tmp_path, real) -> None:
    from llm.data.copy_ladder import write_training_sets

    summary = write_training_sets(tmp_path, real, copy_cases(real))
    assert summary == {"copy36": 36, "copy648": 648, "validation": 18, "permuted": 36}
    for name, count in (("copy36", 36), ("copy648", 648)):
        directory = tmp_path / name / "sft"
        assert len((directory / "train.jsonl").read_text(encoding="utf-8").splitlines()) == count
        assert (directory / "train.ids.bin").stat().st_size > 0
    assert len((tmp_path / "trained.jsonl").read_text(encoding="utf-8").splitlines()) == 36
