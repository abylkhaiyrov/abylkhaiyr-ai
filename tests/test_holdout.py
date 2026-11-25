from __future__ import annotations

import json
from collections import Counter

from llm.data.grounded import (
    INVENTED,
    eval_cases,
    prompt_of,
    training_examples,
)
from llm.data.grounded_text import (
    HOLDOUT_QUESTIONS,
    HOLDOUT_RELATIONS,
    HOLDOUT_STATEMENTS,
    LANGUAGES,
    QUESTIONS,
    STATEMENTS,
    TRAIN_RELATIONS,
)
from llm.data.holdout import (
    HOLDOUT_SEED,
    holdout_cases,
    known_names,
    natural_cases,
    write_holdout,
)
from llm.data.natural_text import NATURAL
from llm.evaluation.grounded import judge, load_cases


def test_holdout_names_never_appear_in_training_dev_or_test() -> None:
    cases = holdout_cases()
    names = {name for case in cases for name in INVENTED.findall(case.prompt)}
    assert names and not names & known_names()
    assert known_names() >= {
        name for case in eval_cases("test") for name in INVENTED.findall(case.prompt)
    }


def test_holdout_prompts_are_new_and_deterministic() -> None:
    first = holdout_cases()
    assert first == holdout_cases()
    assert first != holdout_cases(HOLDOUT_SEED + 1)
    seen = {case.prompt for split in ("dev", "test") for case in eval_cases(split)}
    seen |= {
        prompt_of(example.plan, example.structure)
        for language in LANGUAGES
        for example in training_examples(language, 2_000, 1)
    }
    assert not {case.prompt for case in first} & seen


def test_holdout_frames_and_relations_are_absent_from_every_earlier_table() -> None:
    for language in LANGUAGES:
        assert not set(HOLDOUT_STATEMENTS[language]) & set(STATEMENTS[language])
        assert not set(HOLDOUT_QUESTIONS[language]) & set(QUESTIONS[language])
    assert not set(HOLDOUT_RELATIONS) & set(TRAIN_RELATIONS)
    cells = Counter(case.cell for case in holdout_cases())
    assert set(cells) == {"world", "structure", "relation", "combination"}
    relation_prompts = [c for c in holdout_cases() if c.cell == "relation" and c.language == "en"]
    assert any(word in c.prompt for c in relation_prompts for word in ("harbor", "librarian"))


def test_holdout_has_independent_families_and_every_intervention_variant() -> None:
    cases = holdout_cases()
    assert len({case.id for case in cases}) == len(cases)
    assert len({case.family for case in cases}) == len({case.link for case in cases})
    assert {case.variant for case in cases} == {
        "original",
        "changed",
        "missing",
        "distractor",
        "restructured",
    }
    for language in LANGUAGES:
        assert len({c.link for c in cases if c.language == language}) >= 70


def test_natural_passages_contain_their_answer_and_lack_the_missing_one() -> None:
    cases = natural_cases()
    assert Counter(case.language for case in cases) == {language: 18 for language in LANGUAGES}
    for passage in NATURAL:
        assert passage.original != passage.changed
        text = passage.text.format(V=passage.original)
        assert passage.original in text and passage.changed not in text
    originals = {c.link: c for c in cases if c.variant == "original"}
    changed = {c.link: c for c in cases if c.variant == "changed"}
    missing = {c.link: c for c in cases if c.variant == "missing"}
    for link, case in originals.items():
        assert case.accept[0][0] in case.prompt
        assert changed[link].accept[0][0] in changed[link].prompt
        assert (
            changed[link].rejected == (case.accept[0][0],)
            and case.accept[0][0]
            not in changed[link].prompt.split("\n\n")[0 if changed[link].layout == "dfirst" else 1]
        )
        assert missing[link].expects == "refusal" and not missing[link].accept
    sample = originals[next(iter(originals))]
    assert judge(sample, f"It is {sample.accept[0][0]}.") == "correct"


def test_written_holdout_matches_its_manifest(tmp_path) -> None:
    manifest = write_holdout(tmp_path)
    assert (tmp_path / "probes" / "probes.jsonl").is_file()
    assert manifest["grounded"]["cases"] == len(load_cases(str(tmp_path / "holdout.jsonl")))
    assert manifest["natural"]["cases"] == 54
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest
