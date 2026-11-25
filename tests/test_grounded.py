from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from llm.core.errors import ConfigError
from llm.data.binary import IGNORE_INDEX, with_extension
from llm.data.grounded import (
    ANSWER,
    INVENTED,
    REFUSAL,
    SEEDS,
    TASKS,
    Structure,
    changed,
    distracted,
    eval_cases,
    plans,
    prompt_of,
    row_of,
    split_of_name,
    train_structure,
    trained_cases,
    training_examples,
    without_target,
)
from llm.data.grounded_blocks import Target, fit_rows, write_reading_block
from llm.data.grounded_text import LANGUAGES, NOUNS
from llm.data.refusals import is_refusal
from llm.evaluation.grounded import evaluate, judge, leaked_names, load_cases
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM


def names_of(texts: list[str]) -> set[str]:
    return {name for text in texts for name in INVENTED.findall(text)}


def train_texts(language: str) -> list[str]:
    return [
        turn["content"]
        for example in training_examples(language, 400, 1)
        for turn in row_of(example, 0)["messages"]
    ]


def test_every_task_exists_in_every_language_with_a_balanced_answerability_split() -> None:
    for language in LANGUAGES:
        examples = training_examples(language, 400, 1)
        tasks = Counter(e.plan.task for e in examples if e.plan.expects == ANSWER)
        assert set(tasks) == set(TASKS)
        assert max(tasks.values()) - min(tasks.values()) <= 1
        refusals = sum(e.plan.expects == REFUSAL for e in examples)
        assert refusals == len(examples) - refusals


def test_every_unanswerable_example_has_an_answerable_twin_with_the_same_question() -> None:
    for language in LANGUAGES:
        examples = training_examples(language, 200, 1)
        for first, second in zip(examples[::2], examples[1::2], strict=True):
            assert first.plan.expects == ANSWER and second.plan.expects == REFUSAL
            assert prompt_of(first.plan, first.structure).split("\n\n")[0:1] != [""]
            assert first.structure == second.structure
            assert first.plan.entity == second.plan.entity
            assert first.plan.question_pick == second.plan.question_pick


def test_unanswerable_documents_do_not_contain_the_target_fact() -> None:
    for language in LANGUAGES:
        for plan in plans(language, "train", 1, 160):
            absent = without_target(plan)
            assert absent.expects == REFUSAL
            text = prompt_of(absent, Structure(False, "dfirst", " "))
            document = text.rsplit("\n\n", 1)[0]
            if plan.task != "match":
                assert plan.key not in document
            assert not any(unit.target for unit in absent.units)


def test_changed_documents_move_the_answer_with_the_source() -> None:
    for language in LANGUAGES:
        for plan in plans(language, "train", 1, 80):
            other = changed(plan)
            assert other.key != plan.key
            assert plan.key in other.rejected
            document = prompt_of(other, Structure(False, "dfirst", " ")).rsplit("\n\n", 1)[0]
            assert other.key in document
            if plan.task not in ("match", "cause"):
                assert plan.key not in document or plan.task in ("negation", "update", "priority")


def test_distractor_variant_puts_a_rival_value_in_the_document_but_not_the_answer() -> None:
    for plan in plans("en", "train", 1, 96):
        if plan.task in ("cause",):
            continue
        rival = distracted(plan)
        text = prompt_of(rival, Structure(False, "dfirst", " "))
        assert plan.spare_value in text
        assert plan.spare_value in rival.rejected
        assert rival.expects == REFUSAL


def test_splits_use_disjoint_entities_and_values() -> None:
    for language in LANGUAGES:
        train = names_of(train_texts(language))
        for split in ("dev", "test"):
            cases = eval_cases(split)
            scoped = [c.prompt for c in cases if c.language == language]
            assert not names_of(scoped) & train
    assert {split_of_name(name) for name in names_of(train_texts("en"))} == {"train"}


def test_held_out_structures_never_appear_in_training_prompts() -> None:
    train = {
        row_of(example, 0)["messages"][0]["content"]
        for language in LANGUAGES
        for example in training_examples(language, 300, 1)
    }
    held = [case for case in eval_cases("test") if case.cell in ("structure", "question")]
    assert held and not {case.prompt for case in held} & train
    assert {case.layout for case in held if case.cell == "structure"} == {"flat", "qfirst"}


def test_evaluation_cases_have_independent_families_and_every_variant() -> None:
    cases = eval_cases("test")
    links = Counter(case.link for case in cases)
    assert len({case.family for case in cases}) == len(links)
    for language in LANGUAGES:
        scoped = [c for c in cases if c.language == language]
        assert len({c.link for c in scoped}) == 92
    variants = {c.variant for c in cases}
    assert variants == {"original", "changed", "missing", "distractor", "restructured"}
    assert len({c.id for c in cases}) == len(cases)
    assert Counter(c.cell for c in cases)["question"] > 0
    first = next(c for c in cases if c.variant == "original")
    twin = next(c for c in cases if c.variant == "changed" and c.link == first.link)
    assert first.accept != twin.accept and first.accept[0][0] in twin.rejected


def test_generation_is_deterministic_and_seeded() -> None:
    assert eval_cases("test") == eval_cases("test")
    assert eval_cases("test") != eval_cases("test", SEEDS["test"] + 1)
    assert training_examples("kk", 50, 1) == training_examples("kk", 50, 1)
    assert training_examples("kk", 50, 1) != training_examples("kk", 50, 2)


def test_missing_answers_are_recognised_as_grounded_refusals_and_vary() -> None:
    for language in LANGUAGES:
        answers = set()
        for example in training_examples(language, 400, 1):
            if example.plan.expects == REFUSAL:
                for style in range(4):
                    answer = row_of(example, style)["messages"][1]["content"]
                    assert is_refusal(answer)
                    answers.add(answer)
        assert len(answers) > 40


def test_trained_cases_are_a_prefix_of_the_training_stream() -> None:
    sample = trained_cases(1)
    rows = {
        row_of(example, 0)["messages"][0]["content"]
        for language in LANGUAGES
        for example in training_examples(language, 2_000, 1)
    }
    assert sample and all(case.prompt in rows for case in sample)


def test_block_fitting_matches_the_supervised_token_budget(tokenizer) -> None:
    examples = training_examples("en", 600, 1)
    rows = fit_rows(examples, Target(40_000, 7_000), tokenizer, 1)
    tokens = supervised = 0
    for row in rows:
        ids, labels = tokenizer.encode_conversation_with_labels(
            [
                __import__("llm.tokenizer.bpe", fromlist=["Message"]).Message(
                    turn["role"], turn["content"], True
                )
                for turn in row["messages"]
            ],
            IGNORE_INDEX,
        )
        tokens += len(ids)
        supervised += sum(label != IGNORE_INDEX for label in labels)
    assert abs(supervised - 7_000) <= 7_000 * 0.01
    assert tokens >= 40_000 and tokens < 41_500


def test_written_block_keeps_other_groups_and_matches_the_reference(tmp_path, tokenizer) -> None:
    base = tmp_path / "base" / "sft_groups"
    base.mkdir(parents=True)
    for group in ("reading_en", "reading_ru", "reading_kk"):
        for split in ("train", "validation", "test"):
            prefix = base / f"{split}.{group}"
            lengths = np.full(40, 90, dtype=np.int64)
            np.save(with_extension(prefix, ".lengths.npy"), lengths)
            labels = np.tile(np.r_[np.full(80, IGNORE_INDEX), np.ones(10)], 40).astype(np.int32)
            labels.tofile(with_extension(prefix, ".labels.bin"))
    (base / "train.chat_en.jsonl").write_text("{}\n", encoding="utf-8")
    report = write_reading_block(tmp_path / "out", tmp_path / "base", tokenizer, 1)
    for entry in report.values():
        assert (
            abs(entry["supervised"] - entry["target_supervised"])
            <= 0.05 * entry["target_supervised"]
        )
    assert (tmp_path / "out" / "sft_groups" / "train.chat_en.jsonl").is_symlink()
    assert not (tmp_path / "out" / "sft_groups" / "train.reading_en.jsonl").is_symlink()


def write_cases(path: Path, cases) -> None:
    from dataclasses import asdict

    path.write_text("\n".join(json.dumps(asdict(c)) for c in cases) + "\n", encoding="utf-8")


def test_judge_separates_correct_refused_copied_and_fabricated_answers() -> None:
    cases = eval_cases("dev")
    answerable = next(c for c in cases if c.variant == "original" and c.language == "en")
    key = answerable.accept[0][0]
    assert judge(answerable, f"It is {key}.") == "correct"
    assert judge(answerable, "The text does not say.") == "unjustified_refusal"
    refusing = next(c for c in cases if c.variant == "missing" and c.language == "en")
    assert judge(refusing, "The text does not say.") == "justified_refusal"
    assert judge(refusing, "It is Zzzzzaaar.") == "fabricated_answer"
    rival = next(c for c in cases if c.variant == "distractor" and c.rejected)
    assert judge(rival, f"It is {rival.rejected[-1]}.") == "copied_distractor"


def test_leaking_names_between_evaluation_and_training_are_detected(tmp_path) -> None:
    cases = eval_cases("dev")[:5]
    leaked = tmp_path / "train.jsonl"
    name = next(iter(names_of([cases[0].prompt])))
    leaked.write_text(
        json.dumps({"messages": [{"role": "user", "content": f"{name} here"}]}) + "\n",
        encoding="utf-8",
    )
    assert name in leaked_names(cases, [str(leaked)])
    assert not leaked_names(cases, [str(tmp_path / "missing.jsonl")])


def test_grounded_suite_reports_intervention_and_answerability_metrics(
    tmp_path, tokenizer, model_config
) -> None:
    cases = [c for c in eval_cases("dev") if c.language == "en"][::5] + trained_cases(1)[:6]
    path = tmp_path / "cases.jsonl"
    write_cases(path, cases)
    engine = InferenceEngine(CausalLM(model_config), tokenizer, device=torch.device("cpu"))
    output = evaluate(engine, {"file": str(path)})
    metrics = output.metrics
    assert {
        "ctx.fact_accuracy",
        "ctx.intervention_accuracy",
        "ctx.distractor_resistance",
        "ctx.justified_refusal",
        "ctx.unjustified_refusal",
        "ctx.hallucination",
        "ctx.balanced_accuracy",
        "ctx.trained.accuracy",
    } <= set(metrics)
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    families = {item.cluster for run in output.items["ctx.fact_accuracy"] for item in run}
    assert families <= {c.family for c in cases}
    assert all(
        set(record) >= {"prompt", "expects", "answer", "finish_reason", "check", "kind"}
        for record in output.records
    )
    assert len(load_cases(str(path))) == len(cases)


def test_suite_refuses_evaluation_names_that_were_trained_on(tmp_path, tokenizer, model_config):
    cases = eval_cases("dev")[:4]
    path = tmp_path / "cases.jsonl"
    write_cases(path, cases)
    name = next(iter(names_of([cases[0].prompt])))
    training = tmp_path / "train.jsonl"
    training.write_text(
        json.dumps({"messages": [{"role": "user", "content": name}]}) + "\n", encoding="utf-8"
    )
    engine = InferenceEngine(CausalLM(model_config), tokenizer, device=torch.device("cpu"))
    with pytest.raises(ConfigError):
        evaluate(engine, {"file": str(path), "training": [str(training)]})


def test_noun_tables_cover_every_relation_in_every_language() -> None:
    assert len({frozenset(NOUNS[language]) for language in LANGUAGES}) == 1
    import random

    rng = random.Random(0)
    assert train_structure(rng).layout in ("qfirst", "dfirst")


def test_binding_documents_mix_entities_and_relations_with_rivals_on_both_axes() -> None:
    from llm.data.grounded import BINDING_TASKS, TASKS

    assert "binding" not in TASKS and BINDING_TASKS[-1] == "binding"
    found = [
        plan
        for language in LANGUAGES
        for plan in plans(language, "train", 1, 90, tasks=BINDING_TASKS)
        if plan.task == "binding"
    ]
    assert len(found) >= 30
    for plan in found:
        facts = [unit for unit in plan.units if unit.kind == "fact"]
        entities = {unit.entity for unit in facts}
        assert len(entities) >= 2 and sum(unit.target for unit in facts) == 1
        same_relation = [u for u in facts if u.relation == plan.relation and not u.target]
        same_entity = [u for u in facts if u.entity == plan.entity and not u.target]
        assert same_relation and same_entity
        assert {u.value for u in same_relation + same_entity} <= set(plan.rejected)
        absent = without_target(plan)
        assert not any(
            u.entity == plan.entity and u.relation == plan.relation for u in absent.units
        )
        assert changed(plan).key != plan.key
