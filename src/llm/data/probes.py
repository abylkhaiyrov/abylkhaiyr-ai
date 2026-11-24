from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from llm.data.binary import tokenize_sft_split
from llm.data.probe_text import (
    ANSWER_SUFFIX,
    CAUSE_QUESTION,
    CAUSE_STATEMENT,
    CAUSES,
    ENTITY_SYLLABLES,
    LANGUAGES,
    MACHINE_NAMES,
    NEGATED_STATEMENT,
    PERSON_SYLLABLES,
    QUESTIONS,
    RELATIONS,
    REVERSE_QUESTIONS,
    RULE_ALT,
    RULE_ANSWER,
    RULE_TRAIN,
    STATEMENTS,
    SUFFIXES,
    TRAIN_QUESTIONS,
)
from llm.tokenizer.bpe import BPETokenizer

TAUGHT_ENTITIES = 24
FRESH_ENTITIES = 36
CONTEXT_DOCS = 12
MACHINES = 3
RULE_INPUTS = 8
OPTIONS = 4
HELD_OUT_QUESTION = 3
ALTERNATE_QUESTION = 4
BUILD_SEED = 2718
VARIANTS = {"single": 1, "multi": TRAIN_QUESTIONS}
ANSWER = "answer"
REFUSAL = "refusal"


@dataclass(frozen=True)
class Probe:
    id: str
    kind: str
    category: int
    language: str
    family: str
    prompt: str
    expects: str
    accept: tuple[tuple[str, ...], ...]
    rejected: tuple[str, ...]
    options: tuple[str, ...]
    answer: int
    taught_in: str


@dataclass(frozen=True)
class Fact:
    entity: str
    relation: str
    value: str
    language: str


@dataclass(frozen=True)
class World:
    taught: tuple[Fact, ...]
    fresh: tuple[str, ...]
    values: dict[str, tuple[str, ...]]
    machines: tuple[tuple[str, int], ...]


def names(rng: random.Random, syllables: Sequence[str], count: int, used: set[str]) -> list[str]:
    found: list[str] = []
    while len(found) < count:
        parts = rng.sample(list(syllables), 2)
        name = ("".join(parts) + rng.choice(SUFFIXES)).capitalize()
        if name not in used:
            used.add(name)
            found.append(name)
    return found


def build_world(seed: int = BUILD_SEED) -> World:
    rng = random.Random(seed)
    used: set[str] = set()
    entities = names(rng, ENTITY_SYLLABLES, TAUGHT_ENTITIES + FRESH_ENTITIES, used)
    pools = {
        "river": names(rng, ENTITY_SYLLABLES, 3 * len(entities), used),
        "founder": names(rng, PERSON_SYLLABLES, 3 * len(entities), used),
        "mountain": names(rng, ENTITY_SYLLABLES, 3 * len(entities), used),
    }
    taught = tuple(
        Fact(entity, relation, pools[relation][index], LANGUAGES[index % len(LANGUAGES)])
        for index, entity in enumerate(entities[:TAUGHT_ENTITIES])
        for relation in RELATIONS
    )
    machines = tuple(
        (name, rng.randint(3, 9)) for name in rng.sample(list(MACHINE_NAMES), MACHINES)
    )
    return World(
        taught,
        tuple(entities[TAUGHT_ENTITIES:]),
        {relation: tuple(pool) for relation, pool in pools.items()},
        machines,
    )


def row(language: str, question: str, answer: str, category: str) -> dict:
    return {
        "category": category,
        "language": language,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


def rule_inputs(machine: int, held_out: bool) -> list[int]:
    start = 10 + machine * 3
    return [start + 2 * step + (1 if held_out else 0) for step in range(RULE_INPUTS)]


def training_rows(world: World, variant: str) -> list[dict]:
    rows = []
    for fact in world.taught:
        for index in range(VARIANTS[variant]):
            question = QUESTIONS[fact.language][fact.relation][index].format(E=fact.entity)
            answer = STATEMENTS[fact.language][fact.relation][index].format(
                E=fact.entity, X=fact.value
            )
            rows.append(row(fact.language, question, answer, "probe_fact"))
    for language in LANGUAGES:
        for number, (machine, step) in enumerate(world.machines):
            for value in rule_inputs(number, False):
                question = RULE_TRAIN[language].format(M=machine, K=step, n=value)
                answer = RULE_ANSWER[language].format(n=value, out=value + step)
                rows.append(row(language, question, answer, "probe_rule"))
    return rows


def validation_rows(world: World) -> list[dict]:
    return [
        row(
            fact.language,
            QUESTIONS[fact.language][fact.relation][HELD_OUT_QUESTION].format(E=fact.entity),
            STATEMENTS[fact.language][fact.relation][0].format(E=fact.entity, X=fact.value),
            "probe_validation",
        )
        for fact in world.taught
    ]


def choices(rng: random.Random, truth: str, pool: Sequence[str]) -> tuple[tuple[str, ...], int]:
    wrong = rng.sample([value for value in pool if value != truth], OPTIONS - 1)
    options = [truth, *wrong]
    rng.shuffle(options)
    return tuple(options), options.index(truth)


def probe(
    identifier: str,
    kind: str,
    category: int,
    language: str,
    family: str,
    prompt: str,
    expects: str,
    truth: str = "",
    options: tuple[str, ...] = (),
    answer: int = 0,
    rejected: tuple[str, ...] = (),
    taught_in: str = "",
) -> Probe:
    accept = ((truth,),) if truth else ()
    return Probe(
        identifier,
        kind,
        category,
        language,
        family,
        prompt,
        expects,
        accept,
        rejected,
        options,
        answer,
        taught_in,
    )


def fact_probes(world: World, rng: random.Random) -> list[Probe]:
    probes = []
    for fact in world.taught:
        pool = world.values[fact.relation]
        options, answer = choices(rng, fact.value, pool)
        for language in LANGUAGES:
            templates = QUESTIONS[language][fact.relation]
            same = language == fact.language
            variants = [("fact_paraphrase_same" if same else "fact_paraphrase_cross", 3)]
            if same:
                variants += [("fact_recall", 0), ("fact_paraphrase_alt", 4)]
            for kind, index in variants:
                prompt = templates[index].format(E=fact.entity)
                probes.append(
                    probe(
                        f"{kind}/{language}/{fact.entity}/{fact.relation}",
                        kind,
                        1 if kind == "fact_recall" else 2,
                        language,
                        fact.relation,
                        prompt,
                        ANSWER,
                        fact.value,
                        options,
                        answer,
                        taught_in=fact.language,
                    )
                )
        reverse = REVERSE_QUESTIONS[fact.language][fact.relation].format(X=fact.value)
        entities = [other.entity for other in world.taught if other.relation == fact.relation]
        reverse_options, reverse_answer = choices(rng, fact.entity, entities)
        probes.append(
            probe(
                f"fact_reverse/{fact.language}/{fact.entity}/{fact.relation}",
                "fact_reverse",
                2,
                fact.language,
                fact.relation,
                reverse,
                ANSWER,
                fact.entity,
                reverse_options,
                reverse_answer,
                taught_in=fact.language,
            )
        )
    return probes


def with_suffix(language: str, prompt: str) -> str:
    return f"{prompt} {ANSWER_SUFFIX[language]}"


def document(language: str, facts: Sequence[Fact]) -> str:
    return " ".join(
        STATEMENTS[language][fact.relation][0].format(E=fact.entity, X=fact.value) for fact in facts
    )


def context_facts(world: World, rng: random.Random, index: int) -> tuple[Fact, ...]:
    first, second = world.fresh[2 * index], world.fresh[2 * index + 1]
    return tuple(
        Fact(entity, relation, rng.choice(world.values[relation]), "")
        for entity in (first, second)
        for relation in RELATIONS
    )


def context_probes(world: World, rng: random.Random) -> list[Probe]:
    probes = []
    for language in LANGUAGES:
        for index in range(CONTEXT_DOCS):
            facts = context_facts(world, rng, index)
            shown = [
                fact
                for fact in facts
                if (fact.entity, fact.relation) != (facts[0].entity, "mountain")
            ]
            rng.shuffle(shown)
            text = document(language, shown)
            probes += context_pair(language, index, text, facts, world, rng)
            probes += context_semantics(language, index, facts, world, rng)
    return probes


def context_pair(
    language: str, index: int, text: str, facts: Sequence[Fact], world: World, rng: random.Random
) -> list[Probe]:
    first = facts[0]
    river, founder = facts[0], facts[1]
    target = river if index % 2 == 0 else founder
    rival = facts[3] if index % 2 == 0 else facts[4]
    question = QUESTIONS[language][target.relation][HELD_OUT_QUESTION].format(E=first.entity)
    options, answer = choices(rng, target.value, world.values[target.relation])
    missing = QUESTIONS[language]["mountain"][HELD_OUT_QUESTION].format(E=first.entity)
    return [
        probe(
            f"ctx_fact/{language}/{index}/{target.relation}",
            "ctx_fact",
            3,
            language,
            target.relation,
            f"{text} {question}",
            ANSWER,
            target.value,
            options,
            answer,
            rejected=(rival.value,),
        ),
        probe(
            f"ctx_missing/{language}/{index}/mountain",
            "ctx_missing",
            4,
            language,
            "mountain",
            f"{text} {missing}",
            REFUSAL,
        ),
    ]


def context_semantics(
    language: str, index: int, facts: Sequence[Fact], world: World, rng: random.Random
) -> list[Probe]:
    entity = facts[0].entity
    right, wrong = rng.sample(world.values["river"], 2)
    negated = NEGATED_STATEMENT[language].format(E=entity, W=wrong, X=right)
    question = QUESTIONS[language]["river"][HELD_OUT_QUESTION].format(E=entity)
    options, answer = choices(rng, right, world.values["river"])
    negation = probe(
        f"ctx_negation/{language}/{index}",
        "ctx_negation",
        3,
        language,
        "negation",
        f"{negated} {question}",
        ANSWER,
        right,
        options,
        answer,
        rejected=(wrong,),
    )
    causes = CAUSES[language]
    true_cause, other_cause = rng.sample(range(len(causes)), 2)
    first, second = facts[0].entity, facts[3].entity
    text = " ".join(
        [
            CAUSE_STATEMENT[language].format(E=first, C=causes[true_cause][0]),
            CAUSE_STATEMENT[language].format(E=second, C=causes[other_cause][0]),
        ]
    )
    cause_options = tuple(cause for cause, _ in causes)
    cause = probe(
        f"ctx_cause/{language}/{index}",
        "ctx_cause",
        3,
        language,
        "cause",
        f"{text} {CAUSE_QUESTION[language].format(E=first)}",
        ANSWER,
        causes[true_cause][1],
        cause_options,
        true_cause,
        rejected=(causes[other_cause][1],),
    )
    return [negation, cause]


def entity_probes(world: World) -> list[Probe]:
    probes = []
    for fact in world.taught:
        if fact.relation == "river":
            question = QUESTIONS[fact.language]["river"][HELD_OUT_QUESTION]
            probes.append(
                probe(
                    f"known_entity/{fact.language}/{fact.entity}",
                    "known_entity",
                    4,
                    fact.language,
                    "river",
                    question.format(E=fact.entity),
                    ANSWER,
                    fact.value,
                    taught_in=fact.language,
                )
            )
    for index, entity in enumerate(world.fresh):
        language = LANGUAGES[index % len(LANGUAGES)]
        question = QUESTIONS[language]["river"][HELD_OUT_QUESTION].format(E=entity)
        probes.append(
            probe(
                f"fictional_entity/{language}/{entity}",
                "fictional_entity",
                4,
                language,
                "river",
                question,
                REFUSAL,
            )
        )
    return probes


def rule_probes(world: World, rng: random.Random) -> list[Probe]:
    probes = []
    for language in LANGUAGES:
        for number, (machine, step) in enumerate(world.machines):
            for held, kind, template in (
                (True, "rule_inputs", RULE_TRAIN[language]),
                (True, "rule_phrasing", RULE_ALT[language]),
            ):
                for value in rule_inputs(number, held):
                    out = value + step
                    options, answer = choices(
                        rng,
                        str(out),
                        [str(out + delta) for delta in (-2, -1, 1, 2, 3)] + [str(out)],
                    )
                    probes.append(
                        probe(
                            f"{kind}/{language}/{machine}/{value}",
                            kind,
                            2,
                            language,
                            "rule",
                            template.format(M=machine, K=step, n=value),
                            ANSWER,
                            str(out),
                            options,
                            answer,
                            taught_in=language,
                        )
                    )
    return probes


def build_probes(world: World) -> list[Probe]:
    rng = random.Random(BUILD_SEED + 1)
    return [
        *fact_probes(world, rng),
        *context_probes(world, rng),
        *entity_probes(world),
        *rule_probes(world, rng),
    ]


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8"
    )


def write_diagnostics(output: Path, tokenizer: BPETokenizer, seed: int = BUILD_SEED) -> dict:
    world = build_world(seed)
    probes = build_probes(world)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "probes.jsonl", [asdict(item) for item in probes])
    for variant in VARIANTS:
        directory = output / variant / "sft"
        directory.mkdir(parents=True, exist_ok=True)
        for split, rows in (
            ("train", training_rows(world, variant)),
            ("validation", validation_rows(world)),
        ):
            write_jsonl(directory / f"{split}.jsonl", rows)
            tokenize_sft_split(directory / f"{split}.jsonl", directory / split, tokenizer)
    kinds: dict[str, int] = {}
    for item in probes:
        kinds[item.kind] = kinds.get(item.kind, 0) + 1
    return {"probes": len(probes), "kinds": kinds, "seed": seed}
