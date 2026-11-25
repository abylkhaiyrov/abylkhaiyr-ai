from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict

import pytest

from llm.data import grounded_frames, probe_text
from llm.data.grounded import (
    BINDING_TASKS,
    PairCase,
    eval_cases,
    inventory,
    pair_cases,
    plans,
    prompt_of,
    row_of,
    trained_cases,
    training_examples,
    without_target,
)
from llm.data.grounded_frames import (
    BANKS,
    DIVERSE,
    LEGACY,
    LEGACY_TABLES,
    LEGACY_TRAIN_COUNTS,
    TEMPLATE_SPLITS,
    legacy_bank,
    split_ids,
)
from llm.data.grounded_text import LANGUAGES
from llm.data.holdout import holdout_cases, natural_cases
from llm.evaluation.grounded import fine_error, judge

GOLDEN = {
    "dev": "b11967b6e9f3122a711903e6af6f4d01e5154b6d8440f1c9468ca8033905c451",
    "test": "86e79768b3f72c53f8375626b1417e1a1772d75bf53c007402d111dd9458eec7",
    "holdout": "84ae350d700f2c01c1f29e95e79a1aa23fd5c8bfb95bb155cf1804af06ec7255",
    "natural": "6914763ed2b043c27301c23eb534472dde629f1bf2621ea5bd38c1aca2638564",
    "train50": "a26f6ad69163e5bad3171f6354d2f98951e02c3d9362164bccf0a55e67c8d6cb",
    "train25": "19f90c349539352b427b9e75b059a7da0686c7964fa464a24ec35c54a7c482ae",
    "bind25": "328f2e6ba6a083069a39fef18f1ed88ae8ace24d6b8b6a3c79172df795b92869",
    "trained": "b9ec3b07ef322d416cf4746cc3f05e0ff19d718b6efdf87601a587f5b55a9867",
}
PLACEHOLDER = re.compile(r"\{[A-Za-z]+\}")


def sha(lines) -> str:
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def dump(case) -> str:
    return json.dumps(asdict(case), ensure_ascii=False) + "\n"


def skeleton(text: str) -> str:
    return PLACEHOLDER.sub("_", text).casefold().strip()


def rows(**kwargs) -> str:
    return sha(
        json.dumps(row_of(e, s), ensure_ascii=False) + "\n"
        for language in LANGUAGES
        for e in training_examples(language, 1500, 1, **kwargs)
        for s in (0, 3)
    )


def test_historical_data_reproduces_byte_for_byte_with_the_new_mode_off() -> None:
    assert sha(dump(c) for c in eval_cases("dev")) == GOLDEN["dev"]
    assert sha(dump(c) for c in eval_cases("test")) == GOLDEN["test"]
    assert sha(dump(c) for c in holdout_cases()) == GOLDEN["holdout"]
    assert sha(dump(c) for c in natural_cases()) == GOLDEN["natural"]
    assert sha(dump(c) for c in trained_cases(1)) == GOLDEN["trained"]
    assert rows() == GOLDEN["train50"]
    assert rows(missing_share=0.25) == GOLDEN["train25"]
    assert rows(missing_share=0.25, tasks=BINDING_TASKS) == GOLDEN["bind25"]


def test_split_membership_is_explicit_and_does_not_move_when_a_table_grows(monkeypatch) -> None:
    before = {t: legacy_bank(t, "en") for t in LEGACY_TABLES}
    grown = {
        table: {**texts, "en": (*texts["en"], "An extra template {E} {X}.")}
        for table, texts in LEGACY_TABLES.items()
    }
    monkeypatch.setattr(grounded_frames, "LEGACY_TABLES", grown)
    for table, bank in before.items():
        after = grounded_frames.legacy_bank(table, "en")
        assert after.train == bank.train
        assert after.dev[: len(bank.dev)] == bank.dev
        assert len(after.train) == LEGACY_TRAIN_COUNTS[table]


@pytest.mark.parametrize("mode", [LEGACY, DIVERSE])
def test_train_templates_never_appear_in_dev_test_or_holdout(mode: str) -> None:
    for table, languages in BANKS[mode].items():
        for bank in languages.values():
            texts = {split: set(getattr(bank, split)) for split in TEMPLATE_SPLITS}
            for other in ("dev", "test", "holdout"):
                shared = texts["train"] & texts[other]
                assert not shared or table == "claims"
            assert not texts["holdout"] & (texts["dev"] | texts["test"])
    ids = split_ids(mode)
    assert ids["holdout"] and not ids["train"] & ids["holdout"]


def test_diverse_training_templates_are_about_three_times_the_legacy_set() -> None:
    for table in LEGACY_TABLES:
        for language in LANGUAGES:
            legacy = len(BANKS[LEGACY][table][language].train)
            diverse = len(BANKS[DIVERSE][table][language].train)
            assert diverse >= 2.9 * legacy
            assert set(BANKS[LEGACY][table][language].train) <= set(
                BANKS[DIVERSE][table][language].train
            )


def test_diverse_training_text_never_copies_a_template_used_for_evaluation() -> None:
    evaluation: set[str] = set()
    for table, languages in BANKS[DIVERSE].items():
        for bank in languages.values():
            if table != "claims":
                evaluation.update(skeleton(t) for t in (*bank.dev, *bank.test, *bank.holdout))
    for name in dir(probe_text):
        value = getattr(probe_text, name)
        values = []
        if isinstance(value, dict):
            for item in value.values():
                values.extend(item.values() if isinstance(item, dict) else [item])
        elif isinstance(value, str):
            values.append(value)
        for item in values:
            for text in (item if isinstance(item, tuple) else (item,)):
                if isinstance(text, str):
                    evaluation.add(skeleton(text))
    for table, languages in BANKS[DIVERSE].items():
        for bank in languages.values():
            if table == "claims":
                continue
            for text in bank.train[len(BANKS[LEGACY][table]["en"].train) :]:
                assert skeleton(text) not in evaluation


def test_diversity_changes_only_the_text_and_never_the_facts() -> None:
    for language in LANGUAGES:
        legacy = training_examples(language, 600, 1, missing_share=0.25)
        diverse = training_examples(language, 600, 1, missing_share=0.25, mode=DIVERSE)
        assert [e.plan for e in legacy] == [e.plan for e in diverse]
        for example in diverse:
            document = prompt_of(example.plan, example.structure)
            if example.plan.expects == "answer":
                assert example.plan.key in document
            else:
                assert not any(unit.target for unit in example.plan.units)
        prompts = [prompt_of(e.plan, e.structure) for e in diverse]
        assert len(set(prompts)) == len(prompts)


def test_diverse_mode_raises_templates_structures_and_covers_every_form() -> None:
    legacy = training_examples("en", 900, 1, missing_share=0.25)
    diverse = training_examples("en", 900, 1, missing_share=0.25, mode=DIVERSE)
    legacy_templates, legacy_structures = inventory(legacy)
    templates, structures = inventory(diverse)
    assert len(templates) >= 2.5 * len(legacy_templates)
    assert len(structures) >= 5 * len(legacy_structures)
    assert {form for _, form, _ in structures} == {"sentences", "profile", "table"}
    assert {layout for layout, _, _ in structures} >= {"labeled_dfirst", "labeled_qfirst"}
    assert all("structure" in row_of(e, 0) for e in diverse[:5])
    assert all("structure" not in row_of(e, 0) for e in legacy[:5])


def test_profile_and_table_forms_keep_every_fact_in_the_document() -> None:
    diverse = training_examples("ru", 400, 1, tasks=BINDING_TASKS, mode=DIVERSE)
    checked = 0
    for example in diverse:
        if example.structure.form == "sentences":
            continue
        document = prompt_of(example.plan, example.structure)
        for unit in example.plan.units:
            if unit.kind == "fact":
                assert unit.value in document and unit.entity in document
                checked += 1
    assert checked > 100


def test_pair_cases_label_rivals_and_separate_entity_from_relation_errors() -> None:
    cases = pair_cases(per_language=12)
    assert all(isinstance(c, PairCase) for c in cases)
    assert {c.variant for c in cases} == {
        "original",
        "changed",
        "missing",
        "other_entity",
        "other_relation",
        "requestion",
    }
    for case in cases:
        assert len(case.kinds) == len(case.rejected)
        if case.variant == "other_entity":
            assert set(case.kinds) == {"entity"}
        if case.variant == "other_relation":
            assert set(case.kinds) == {"relation"}
    by_link: dict[str, dict[str, PairCase]] = {}
    for case in cases:
        by_link.setdefault(case.link, {})[case.variant] = case
    for variants in by_link.values():
        question = lambda c: c.prompt.split("\n\n")[0 if c.layout == "qfirst" else -1]
        assert question(variants["original"]) == question(variants["changed"])
        assert question(variants["original"]) != question(variants["requestion"])


def test_fine_error_classes_separate_binding_mistakes_from_other_failures() -> None:
    case = next(
        c
        for c in pair_cases(per_language=12)
        if c.variant == "missing" and "entity" in c.kinds and "relation" in c.kinds
    )
    entity_value = case.rejected[case.kinds.index("entity")]
    relation_value = case.rejected[case.kinds.index("relation")]
    for text, expected in (
        (f"It is {entity_value}.", "other_entity"),
        (f"It is {relation_value}.", "other_relation"),
        ("The text does not say.", "justified_refusal"),
    ):
        assert fine_error(case, text, "stop", judge(case, text)) == expected
    answerable = next(c for c in pair_cases(per_language=12) if c.variant == "original")
    key = answerable.accept[0][0]
    assert fine_error(answerable, f"It is {key}.", "stop", "correct") == "correct"
    assert fine_error(answerable, "It is", "length", "no_answer") == "incomplete"
    assert fine_error(answerable, "The text does not say.", "stop", "unjustified_refusal") == (
        "unjustified_refusal"
    )
    question = answerable.prompt.split("\n\n")[0]
    assert fine_error(answerable, question, "stop", "no_answer") == "echo"
    assert fine_error(answerable, "It is Zzzzzaaar.", "stop", "fabricated_answer") == "fabricated"


def test_unanswerable_binding_variants_remove_the_target_but_keep_rivals() -> None:
    for plan in plans("en", "dev", 5, 60, tasks=BINDING_TASKS):
        if plan.task != "binding":
            continue
        absent = without_target(plan)
        assert not any(u.target for u in absent.units)
        assert any(u.kind == "fact" for u in absent.units)


def test_suite_prefix_renames_metrics_and_items(tmp_path, tokenizer, model_config) -> None:
    import torch

    from llm.evaluation.grounded import evaluate
    from llm.inference.engine import InferenceEngine
    from llm.model.transformer import CausalLM

    cases = pair_cases(per_language=3)[:12]
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in cases), encoding="utf-8"
    )
    engine = InferenceEngine(
        CausalLM(model_config.model_copy(update={"max_position_embeddings": 1024})),
        tokenizer,
        device=torch.device("cpu"),
    )
    output = evaluate(engine, {"file": str(path), "prefix": "pairs."})
    assert output.metrics and all(name.startswith("pairs.") for name in output.metrics)
    assert all(name.startswith("pairs.") for name in output.items)
    assert any(name.startswith("pairs.ctx.error.") for name in output.metrics)
    assert "pairs.ctx.binding_accuracy" in output.metrics
    assert all("fine" in record for record in output.records)
