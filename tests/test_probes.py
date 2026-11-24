from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch

from llm.data.probe_text import LANGUAGES, QUESTIONS, RELATIONS, TRAIN_QUESTIONS
from llm.data.probes import (
    ANSWER,
    REFUSAL,
    VARIANTS,
    build_probes,
    build_world,
    training_rows,
    validation_rows,
    write_diagnostics,
)
from llm.evaluation.probes import classify, evaluate, fabricates, load_probes
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM


def test_world_and_probes_are_deterministic_and_have_unique_identifiers():
    first, second = build_probes(build_world()), build_probes(build_world())
    assert first == second
    assert len({probe.id for probe in first}) == len(first)
    assert build_probes(build_world(1)) != first


def test_every_kind_covers_the_three_languages_in_balanced_numbers():
    probes = build_probes(build_world())
    cells = Counter((probe.kind, probe.language) for probe in probes)
    for kind in {probe.kind for probe in probes}:
        counts = [cells[(kind, language)] for language in LANGUAGES]
        assert min(counts) > 0
        if kind != "fact_paraphrase_cross":
            assert max(counts) - min(counts) <= max(counts) // 2
    categories = {probe.category for probe in probes}
    assert categories == {1, 2, 3, 4}


def test_each_fact_is_taught_in_exactly_one_language_and_tested_in_all():
    world = build_world()
    by_entity: dict[str, set[str]] = {}
    for fact in world.taught:
        by_entity.setdefault(fact.entity, set()).add(fact.language)
    assert all(len(languages) == 1 for languages in by_entity.values())
    probes = build_probes(world)
    taught = {fact.entity: fact.language for fact in world.taught}
    cross = [probe for probe in probes if probe.kind == "fact_paraphrase_cross"]
    assert cross and all(probe.language != probe.taught_in for probe in cross)
    assert {probe.language for probe in probes if probe.kind == "fact_recall"} == set(LANGUAGES)
    assert all(taught[probe.id.split("/")[2]] == probe.taught_in for probe in cross)


def test_training_never_contains_held_out_questions_reversals_or_fresh_entities():
    world = build_world()
    for variant, templates in VARIANTS.items():
        rows = training_rows(world, variant)
        users = [row["messages"][0]["content"] for row in rows if row["category"] == "probe_fact"]
        assert len(users) == len(world.taught) * templates
        text = " ".join(item["content"] for row in rows for item in row["messages"])
        assert not any(entity in text for entity in world.fresh)
        for fact in world.taught:
            for index in (3, 4):
                held = QUESTIONS[fact.language][fact.relation][index].format(E=fact.entity)
                assert held not in users
    allowed = {fact.entity for fact in world.taught}
    assert {
        probe.id.split("/")[2] for probe in build_probes(world) if probe.kind == "known_entity"
    } <= allowed
    assert TRAIN_QUESTIONS == VARIANTS["multi"] and VARIANTS["single"] == 1


def test_rule_probes_use_inputs_that_were_never_trained():
    world = build_world()
    trained = " ".join(
        row["messages"][0]["content"]
        for row in training_rows(world, "single")
        if row["category"] == "probe_rule"
    )
    probes = [probe for probe in build_probes(world) if probe.kind.startswith("rule_")]
    assert probes and all(probe.prompt not in trained for probe in probes)
    assert {probe.kind for probe in probes} == {"rule_inputs", "rule_phrasing"}


def test_context_pairs_share_a_document_and_the_missing_fact_is_really_absent():
    probes = build_probes(build_world())
    facts = {
        probe.id.split("/", 1)[1].rsplit("/", 1)[0]: probe
        for probe in probes
        if probe.kind == "ctx_fact"
    }
    missing = [probe for probe in probes if probe.kind == "ctx_missing"]
    assert len(missing) == len(facts)
    for probe in missing:
        assert probe.expects == REFUSAL and probe.accept == ()
        document = probe.prompt
        entity = document.split()[-1].rstrip("?.")
        assert document.count("гора") + document.count("mountain") + document.count("тау") <= 2
        assert entity


def test_negation_and_cause_probes_reject_the_distractor_value():
    for probe in build_probes(build_world()):
        if probe.kind in {"ctx_negation", "ctx_cause", "ctx_fact"}:
            assert probe.expects == ANSWER and probe.rejected and probe.accept
            assert probe.rejected[0] != probe.accept[0][0]
            assert probe.options[probe.answer] or probe.kind == "ctx_cause"


def test_fabrication_heuristic_ignores_sentence_starts_and_prompt_names():
    prompt = "Who founded Zelmar?"
    assert not fabricates("Zelmar was founded long ago.", prompt)
    assert not fabricates("I think Zelmar is old.", prompt)
    assert fabricates("I think Brenvolt founded it.", prompt)
    assert fabricates("It was founded in 1923.", prompt)
    assert not fabricates("I do not know.", prompt)


def case(kind: str = "ctx_fact", expects: str = ANSWER, accept="Kelmar", rejected="Doran"):
    from llm.data.probes import Probe

    groups = ((accept,),) if accept else ()
    return Probe(
        "id",
        kind,
        3,
        "en",
        "river",
        "The river of Zorvan is Kelmar. Which river?",
        expects,
        groups,
        (rejected,) if rejected else (),
        (),
        0,
        "",
    )


def test_answer_classification_separates_every_failure_mode():
    assert classify(case(), "It is the river Kelmar.") == "correct"
    assert (
        classify(
            case(), "I do not have information about that, and I would rather not invent an answer."
        )
        == "unjustified_refusal"
    )
    assert classify(case(), "The river is Doran.") == "copied_distractor"
    assert classify(case(), "It is the river Brenvolt.") == "fabricated_answer"
    assert classify(case(), "A river is a stream.") == "no_answer"
    missing = case("ctx_missing", REFUSAL, "", "")
    assert classify(missing, "В тексте об этом не говорится.") == "justified_refusal"
    assert classify(missing, "The mountain is Brenvolt.") == "fabricated_answer"
    assert classify(missing, "Well, that is a good question.") == "non_refusal_other"


def test_diagnostic_files_are_written_and_consistent(tmp_path, tokenizer):
    summary = write_diagnostics(tmp_path, tokenizer)
    probes = load_probes(str(tmp_path / "probes.jsonl"))
    assert summary["probes"] == len(probes)
    for variant in VARIANTS:
        rows = [
            json.loads(line)
            for line in (tmp_path / variant / "sft" / "train.jsonl").read_text().splitlines()
        ]
        assert len(rows) == len(training_rows(build_world(), variant))
        assert (tmp_path / variant / "sft" / "train.ids.bin").exists()
    assert len(validation_rows(build_world())) == len(build_world().taught)
    assert {relation for relation in RELATIONS} == {
        probe.family for probe in probes if probe.kind == "fact_recall"
    }


def test_probe_suite_reports_free_form_choice_and_answerability_metrics(
    tmp_path, tokenizer, model_config
):
    write_diagnostics(tmp_path, tokenizer)
    subset = [
        line
        for line in (tmp_path / "probes.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["kind"]
        in {"ctx_fact", "ctx_missing", "known_entity", "fictional_entity", "fact_recall"}
    ][::4]
    (tmp_path / "subset.jsonl").write_text("\n".join(subset) + "\n", encoding="utf-8")
    unigram = tmp_path / "unigram.bin"
    (np.arange(4000) % (tokenizer.vocab_size - 1) + 1).astype(np.uint16).tofile(unigram)
    engine = InferenceEngine(CausalLM(model_config), tokenizer, device=torch.device("cpu"))
    output = evaluate(engine, {"file": str(tmp_path / "subset.jsonl"), "unigram": str(unigram)})
    metrics = output.metrics
    assert 0.0 <= metrics["probe.free_form_correctness"] <= 1.0
    assert {
        "answerability.unjustified_refusal",
        "answerability.justified_refusal",
        "answerability.hallucination",
    } <= set(metrics)
    assert any(name.startswith("probe.ctx_fact.mcq.") for name in metrics)
    assert any(name.startswith("probe.ctx_fact.option_only.") for name in metrics)
    assert {record["check"] for record in output.records} <= {
        "correct",
        "unjustified_refusal",
        "copied_distractor",
        "fabricated_answer",
        "no_answer",
        "justified_refusal",
        "non_refusal_other",
    }
    assert all(
        set(record) >= {"prompt", "expects", "answer", "finish_reason", "check"}
        for record in output.records
    )
