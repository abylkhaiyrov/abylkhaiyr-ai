from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from llm.data.binary import IGNORE_INDEX
from llm.data.refusals import (
    ENTITY,
    GROUNDED,
    NO_ACCESS,
    is_refusal,
    phrasing_counts,
    refusal_kinds,
    template_of,
)
from llm.training.spec import load_spec

SCRIPTS = Path("scripts")


def load_script(name: str):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


ENTITY_ANSWERS = (
    "I do not know of a country called Zorvan. I may simply not have information about it, and I would rather not invent an answer.",
    "Я не знаю страны под названием Зорван. Зорван, у меня просто нет информации о ней, и я не хочу выдумывать ответ.",
    "Мен Зорван деп аталатын елді білмеймін. Зорван ол туралы ақпарат болмауы мүмкін, жауапты ойдан шығарғым келмейді.",
)


@pytest.mark.parametrize("answer", ENTITY_ANSWERS)
def test_entity_refusal_is_detected_in_every_language(answer):
    assert is_refusal(answer)
    assert refusal_kinds({"category": "unknown_entity"}, answer) == [ENTITY]


def test_ordinary_answers_are_not_refusals_and_grounded_refusals_are_distinguished():
    assert not is_refusal("Paris is the capital of France.")
    assert not is_refusal("Я могу отвечать на вопросы и переводить фразы.")
    assert refusal_kinds({"category": "context_qa"}, "В тексте об этом не говорится.") == [GROUNDED]
    assert refusal_kinds({"category": "refusal"}, "Anything") == [NO_ACCESS]
    assert is_refusal("I do not have access to the internet or real-time data.")


def test_phrasings_collapse_random_names_into_one_template_per_language():
    rows = [
        {
            "category": "unknown_entity",
            "messages": [
                {
                    "role": "assistant",
                    "content": answer.replace("Zorvan", name).replace("Зорван", name),
                }
            ],
        }
        for answer in ENTITY_ANSWERS
        for name in ("Zorvan", "Kelmar", "Dunita")
    ]
    assert len(phrasing_counts(rows, ENTITY)) == 3
    assert template_of("I do not know of a country called Zorvan.") == template_of(
        "I do not know of a country called Kelmar."
    )


def test_masking_hides_only_the_refusal_turn_and_keeps_token_ids(tokenizer):
    audit = load_script("audit_sft_refusals")
    row = {
        "category": "unknown_entity",
        "messages": [
            {"role": "user", "content": "Who is the president of Zorvan?"},
            {"role": "assistant", "content": ENTITY_ANSWERS[0]},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "You are welcome"},
        ],
    }
    full_ids, full = audit.encode(row, tokenizer, None)
    hidden_ids, hidden = audit.encode(row, tokenizer, ENTITY)
    assert full_ids == hidden_ids
    removed = sum(a != IGNORE_INDEX for a in full) - sum(a != IGNORE_INDEX for a in hidden)
    assert removed == len(tokenizer.encode(ENTITY_ANSWERS[0])) + 1
    assert sum(a != IGNORE_INDEX for a in hidden) == len(tokenizer.encode("You are welcome")) + 1


def test_stage20_configurations_differ_only_in_the_registered_factors():
    spec = load_spec("configs/experiments/stage20.yaml")
    jobs = {job: spec.job_configuration(job).train for job in spec.job_names()}
    assert spec.runnable_jobs() == ["B-1", "R-1", "A-2", "B-2", "R-2", "A-3", "B-3", "R-3"]
    assert spec.checkpoint_dir("A-1") == Path("checkpoints/stage19/L1")
    for job, train in jobs.items():
        arm, seed = job.split("-")
        assert train.seed == int(seed)
        assert (
            train.optimizer.learning_rate,
            train.optimizer.min_lr_ratio,
            train.optimizer.warmup_steps,
        ) == (5e-5, 1.0, 10)
        assert (train.max_steps, train.restore_optimizer, train.init_from) == (
            305,
            False,
            "checkpoints/tiny_v2/step-00006000",
        )
        assert bool(train.mixture) == (arm == "R")
    assert jobs["R-1"].mixture == {"b_en": 0.25, "b_ru": 0.25, "b_kk": 0.25, "v2": 0.25}
    assert jobs["B-1"].data_dir == "data/stage18/b" and jobs["A-2"].data_dir == "data/stage18/a"
