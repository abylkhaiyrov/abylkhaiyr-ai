from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from llm.data.dedup import stable_hash
from llm.data.grounded_text import (
    ALTERNATIVES,
    ANSWERS,
    CAUSE_ANSWERS,
    CAUSE_QUESTIONS,
    CAUSE_SPLITS,
    CAUSE_STATEMENTS,
    CAUSES,
    CONSONANTS,
    FILLERS,
    GENERIC_MISSING,
    HELD_OUT_RELATIONS,
    KINDS,
    LANGUAGES,
    MISSING,
    NEGATIONS,
    NOUNS,
    NUMBER,
    PERSON,
    PLACE,
    PRIORITY_STATEMENTS,
    QUESTIONS,
    REVERSE_ANSWERS,
    REVERSE_QUESTIONS,
    SOURCE_CLAIMS,
    SOURCES,
    STATEMENTS,
    SUFFIXES,
    TRAIN_RELATIONS,
    UPDATES,
    VOWELS,
)

INVENTED = re.compile(
    r"\b[BDGKLMNPRSTVZ][aeiou](?:[bdgklmnprstvz][aeiou]){2}(?:a|en|is|ar|on|ith|ova)\b"
)
SPLITS = ("train", "dev", "test")
TASKS = (
    "extract",
    "distractor",
    "relation",
    "match",
    "negation",
    "cause",
    "update",
    "priority",
)
LAYOUTS = {"qfirst": "{q}\n\n{d}", "dfirst": "{d}\n\n{q}", "flat": "{d} {q}"}
TRAIN_LAYOUTS = ("qfirst", "dfirst")
HELD_LAYOUTS = ("flat", "qfirst")
ANSWER, REFUSAL = "answer", "refusal"
ENTITY = "entity"
SYLLABLES = {
    kind: [c + v for c in CONSONANTS for v in VOWELS][offset::3]
    for offset, kind in enumerate((ENTITY, PLACE, PERSON))
}
SHARES = {"train": 80, "dev": 90}
SEEDS = {"train": 11, "dev": 22, "test": 33}
NAME_PARTS = 3
YEARS = (1100, 1899)
FILLER_LIMIT = 2
TRAINED_PER_LANGUAGE = 30


def split_of_name(name: str) -> str:
    bucket = stable_hash(f"grounded:{name}") % 100
    return "train" if bucket < SHARES["train"] else "dev" if bucket < SHARES["dev"] else "test"


@dataclass(frozen=True)
class Unit:
    kind: str
    entity: str
    relation: str
    value: str
    other: str = ""
    source: int = 0
    reliable: bool = False
    target: bool = False
    pick: int = 0


@dataclass(frozen=True)
class Plan:
    task: str
    language: str
    relation: str
    entity: str
    value: str
    answer: str
    key: str
    rejected: tuple[str, ...]
    units: tuple[Unit, ...]
    question: str
    question_pick: int
    options: tuple[str, ...]
    spare_entity: str
    spare_value: str
    expects: str = ANSWER


@dataclass(frozen=True)
class Structure:
    held: bool
    layout: str
    joiner: str


@dataclass(frozen=True)
class Case:
    id: str
    link: str
    family: str
    variant: str
    task: str
    cell: str
    layout: str
    language: str
    prompt: str
    expects: str
    accept: tuple[tuple[str, ...], ...]
    rejected: tuple[str, ...]


class Namer:
    def __init__(self, rng: random.Random, split: str) -> None:
        self.rng = rng
        self.split = split
        self.used: set[str] = set()

    def draw(self, kind: str) -> str:
        while True:
            if kind == NUMBER:
                candidate = str(self.rng.randint(*YEARS))
            else:
                parts = self.rng.choices(SYLLABLES[kind], k=NAME_PARTS)
                candidate = ("".join(parts) + self.rng.choice(SUFFIXES)).capitalize()
            if candidate not in self.used and split_of_name(candidate) == self.split:
                self.used.add(candidate)
                return candidate

    def entity(self) -> str:
        return self.draw(ENTITY)

    def value(self, relation: str) -> str:
        return self.draw(KINDS[relation])


def frame_index(size: int, held: bool, pick: int) -> int:
    held_count = 2 if size == 6 else 1
    return size - held_count + pick % held_count if held else pick % (size - held_count)


def cause_key(language: str, phrase: str) -> str:
    return dict(CAUSES[language])[phrase]


def interleave(rng: random.Random, core: list[Unit], extras: list[Unit]) -> list[Unit]:
    result = list(core)
    for extra in extras:
        result.insert(rng.randint(0, len(result)), extra)
    return result


def fillers(rng: random.Random, language: str) -> list[Unit]:
    count = rng.randint(0, FILLER_LIMIT)
    pool = list(range(len(FILLERS[language])))
    return [Unit("filler", "", "", "", pick=index) for index in rng.sample(pool, count)]


def fact_unit(
    rng: random.Random, entity: str, relation: str, value: str, target: bool = False
) -> Unit:
    return Unit("fact", entity, relation, value, target=target, pick=rng.randrange(1_000))


class Builder:
    def __init__(self, language: str, split: str, rng: random.Random) -> None:
        self.language = language
        self.split = split
        self.rng = rng
        self.namer = Namer(rng, split)

    def relation_values(self, relation: str, count: int) -> list[str]:
        return [self.namer.value(relation) for _ in range(count)]

    def finish(
        self,
        task: str,
        relation: str,
        entity: str,
        value: str,
        rejected: Sequence[str],
        units: list[Unit],
        question: str = "ask",
        answer: str | None = None,
        key: str | None = None,
    ) -> Plan:
        spare_entity = self.namer.entity()
        spare_value = self.namer.value(relation) if relation in KINDS else ""
        options: tuple[str, ...] = ()
        if question == "alt":
            options = (value, spare_value) if self.rng.random() < 0.5 else (spare_value, value)
        return Plan(
            task,
            self.language,
            relation,
            entity,
            value,
            value if answer is None else answer,
            value if key is None else key,
            tuple(rejected),
            tuple(interleave(self.rng, units, fillers(self.rng, self.language))),
            question,
            self.rng.randrange(1_000),
            options,
            spare_entity,
            spare_value,
        )

    def extract(self, relation: str, question: str = "ask") -> Plan:
        entity = self.namer.entity()
        value = self.namer.value(relation)
        unit = fact_unit(self.rng, entity, relation, value, target=True)
        return self.finish("extract", relation, entity, value, (), [unit], question)

    def distractor(self, relation: str, question: str = "ask") -> Plan:
        entity = self.namer.entity()
        value = self.namer.value(relation)
        count = self.rng.randint(2, 4)
        others = [self.namer.entity() for _ in range(count)]
        values = self.relation_values(relation, count)
        units = [fact_unit(self.rng, entity, relation, value, target=True)] + [
            fact_unit(self.rng, other, relation, other_value)
            for other, other_value in zip(others, values, strict=True)
        ]
        self.rng.shuffle(units)
        return self.finish("distractor", relation, entity, value, values, units, question)

    def relation(self, relation: str, question: str = "ask") -> Plan:
        entity = self.namer.entity()
        pool = [name for name in TRAIN_RELATIONS if name != relation]
        if relation not in TRAIN_RELATIONS:
            pool = list(TRAIN_RELATIONS)
        extra = self.rng.sample(pool, self.rng.randint(1, 2))
        value = self.namer.value(relation)
        units = [fact_unit(self.rng, entity, relation, value, target=True)]
        rejected = []
        for name in extra:
            other_value = self.namer.value(name)
            rejected.append(other_value)
            units.append(fact_unit(self.rng, entity, name, other_value))
        self.rng.shuffle(units)
        return self.finish("relation", relation, entity, value, rejected, units, question)

    def match(self, relation: str) -> Plan:
        entities = [self.namer.entity() for _ in range(3)]
        values = self.relation_values(relation, 3)
        units = [
            fact_unit(self.rng, entity, relation, value, target=index == 0)
            for index, (entity, value) in enumerate(zip(entities, values, strict=True))
        ]
        self.rng.shuffle(units)
        return self.finish(
            "match",
            relation,
            entities[0],
            values[0],
            entities[1:],
            units,
            "reverse",
            answer=entities[0],
            key=entities[0],
        )

    def negation(self, relation: str) -> Plan:
        entity = self.namer.entity()
        right, wrong = self.relation_values(relation, 2)
        unit = Unit(
            "negation", entity, relation, right, wrong, target=True, pick=self.rng.randrange(1_000)
        )
        extra = [
            fact_unit(self.rng, self.namer.entity(), relation, self.namer.value(relation))
            for _ in range(self.rng.randint(0, 2))
        ]
        units = interleave(self.rng, [unit], extra)
        return self.finish("negation", relation, entity, right, (wrong,), units)

    def cause(self, relation: str) -> Plan:
        indices = self.rng.sample(CAUSE_SPLITS[self.split], 2)
        phrases = [CAUSES[self.language][index] for index in indices]
        entities = [self.namer.entity() for _ in range(2)]
        units = [
            Unit(
                "cause",
                entity,
                "cause",
                phrase,
                target=position == 0,
                pick=self.rng.randrange(1_000),
            )
            for position, (entity, (phrase, _)) in enumerate(zip(entities, phrases, strict=True))
        ]
        self.rng.shuffle(units)
        return self.finish(
            "cause",
            "cause",
            entities[0],
            phrases[0][0],
            (phrases[1][1],),
            units,
            "cause",
            answer=phrases[0][0],
            key=phrases[0][1],
        )

    def update(self, relation: str) -> Plan:
        entity = self.namer.entity()
        old, new = self.relation_values(relation, 2)
        core = [
            fact_unit(self.rng, entity, relation, old),
            Unit(
                "update",
                entity,
                relation,
                new,
                old,
                target=True,
                pick=self.rng.randrange(1_000),
            ),
        ]
        extra = [
            fact_unit(self.rng, self.namer.entity(), relation, self.namer.value(relation))
            for _ in range(self.rng.randint(0, 2))
        ]
        units = interleave(self.rng, core, extra)
        return self.finish("update", relation, entity, new, (old,), units)

    def priority(self, relation: str) -> Plan:
        entity = self.namer.entity()
        low, high = self.relation_values(relation, 2)
        source = self.rng.randrange(len(SOURCES[self.language]))
        pick = self.rng.randrange(1_000)
        claims = [
            Unit("claim", entity, relation, low, source=source, pick=pick),
            Unit("claim", entity, relation, high, source=source, reliable=True, target=True),
        ]
        self.rng.shuffle(claims)
        trust = Unit("trust", "", relation, "", source=source, pick=pick)
        return self.finish("priority", relation, entity, high, (low,), [*claims, trust])

    def plan(self, task: str, relation: str, question: str = "ask") -> Plan:
        if task == "extract":
            return self.extract(relation, question)
        if task == "distractor":
            return self.distractor(relation, question)
        if task == "relation":
            return self.relation(relation, question)
        simple = {
            "match": self.match,
            "negation": self.negation,
            "cause": self.cause,
            "update": self.update,
            "priority": self.priority,
        }
        return simple[task](relation)


def plans(language: str, split: str, seed: int, count: int, offset: int = 0) -> list[Plan]:
    rng = random.Random(f"{seed}:{language}:{split}:{offset}")
    return [
        Builder(language, split, rng).plan(
            TASKS[index % len(TASKS)], TRAIN_RELATIONS[(index // len(TASKS)) % len(TRAIN_RELATIONS)]
        )
        for index in range(count)
    ]


def render_unit(unit: Unit, plan: Plan, held: bool) -> str:
    language = plan.language
    noun = NOUNS[language].get(unit.relation, "")
    if unit.kind == "filler":
        return FILLERS[language][unit.pick]
    if unit.kind == "fact":
        table = STATEMENTS[language]
        return table[frame_index(len(table), held, unit.pick)].format(
            E=unit.entity, noun=noun, X=unit.value
        )
    if unit.kind == "negation":
        table = NEGATIONS[language]
        return table[frame_index(len(table), held, unit.pick)].format(
            E=unit.entity, noun=noun, X=unit.value, W=unit.other
        )
    if unit.kind == "update":
        table = UPDATES[language]
        return table[frame_index(len(table), held, unit.pick)].format(
            E=unit.entity, noun=noun, X=unit.value
        )
    if unit.kind == "cause":
        table = CAUSE_STATEMENTS[language]
        return table[frame_index(len(table), held, unit.pick)].format(E=unit.entity, C=unit.value)
    low_dative, low_nominative, high_dative, high_nominative = SOURCES[language][unit.source]
    if unit.kind == "claim":
        source = high_dative if unit.reliable else low_dative
        return SOURCE_CLAIMS[language].format(S=source, E=unit.entity, noun=noun, V=unit.value)
    table = PRIORITY_STATEMENTS[language]
    return table[frame_index(len(table), held, unit.pick)].format(
        High=high_nominative, Low=low_nominative
    )


def render_document(plan: Plan, structure: Structure) -> str:
    return structure.joiner.join(render_unit(unit, plan, structure.held) for unit in plan.units)


def render_question(plan: Plan, held: bool) -> str:
    language = plan.language
    noun = NOUNS[language].get(plan.relation, "")
    if plan.question == "reverse":
        table = REVERSE_QUESTIONS[language]
        return table[frame_index(len(table), held, plan.question_pick)].format(
            noun=noun, X=plan.value
        )
    if plan.question == "cause":
        table = CAUSE_QUESTIONS[language]
        return table[frame_index(len(table), held, plan.question_pick)].format(E=plan.entity)
    if plan.question == "alt":
        first, second = plan.options
        return ALTERNATIVES[language].format(noun=noun, E=plan.entity, A=first, B=second)
    table = QUESTIONS[language]
    return table[frame_index(len(table), held, plan.question_pick)].format(E=plan.entity, noun=noun)


def prompt_of(plan: Plan, structure: Structure) -> str:
    return LAYOUTS[structure.layout].format(
        d=render_document(plan, structure), q=render_question(plan, structure.held)
    )


def answer_options(plan: Plan) -> list[str]:
    language = plan.language
    if plan.expects == REFUSAL:
        table = MISSING[language]
        indices = GENERIC_MISSING if plan.task == "match" else range(len(table))
        noun = NOUNS[language].get(plan.relation, "")
        return [table[i].format(E=plan.entity, noun=noun) for i in indices]
    noun = NOUNS[language].get(plan.relation, "")
    if plan.task == "match":
        return [text.format(E=plan.answer) for text in REVERSE_ANSWERS[language]]
    if plan.task == "cause":
        return [text.format(E=plan.entity, C=plan.answer) for text in CAUSE_ANSWERS[language]]
    return [text.format(E=plan.entity, noun=noun, X=plan.answer) for text in ANSWERS[language]]


def changed(plan: Plan) -> Plan:
    if plan.task in ("cause", "match"):
        target = next(unit for unit in plan.units if unit.target)
        other = next(
            unit for unit in plan.units if unit.kind in ("fact", "cause") and not unit.target
        )
        swapped = tuple(
            (
                replace(unit, value=other.value)
                if unit is target
                else replace(unit, value=target.value) if unit is other else unit
            )
            for unit in plan.units
        )
        if plan.task == "cause":
            return replace(
                plan,
                units=swapped,
                answer=other.value,
                key=cause_key(plan.language, other.value),
                rejected=(plan.key,),
            )
        return replace(
            plan,
            units=swapped,
            answer=other.entity,
            key=other.entity,
            rejected=(plan.entity, *[e for e in plan.rejected if e != other.entity]),
        )
    units = tuple(
        replace(unit, value=plan.spare_value) if unit.target else unit for unit in plan.units
    )
    return replace(
        plan,
        units=units,
        value=plan.spare_value,
        answer=plan.spare_value,
        key=plan.spare_value,
        rejected=(plan.key, *plan.rejected),
    )


def without_target(plan: Plan) -> Plan:
    units = tuple(
        unit
        for unit in plan.units
        if unit.kind != "trust"
        and not (unit.entity == plan.entity and unit.relation == plan.relation)
        and not (plan.task == "cause" and unit.target)
    )
    if not any(unit.kind != "filler" for unit in units):
        units = (Unit("filler", "", "", "", pick=plan.question_pick % len(FILLERS[plan.language])),)
    remaining = tuple(
        item
        for unit in units
        if unit.kind not in ("filler", "trust")
        for item in (
            (unit.entity, unit.value)
            if plan.task == "match"
            else (cause_key(plan.language, unit.value),) if unit.kind == "cause" else (unit.value,)
        )
    )
    return replace(plan, units=units, expects=REFUSAL, rejected=remaining)


def distracted(plan: Plan) -> Plan:
    base = without_target(plan)
    unit = Unit("fact", plan.spare_entity, plan.relation, plan.spare_value, pick=plan.question_pick)
    return replace(
        base,
        units=(*base.units[:1], unit, *base.units[1:]),
        rejected=(*base.rejected, plan.spare_value),
    )


def accept_of(plan: Plan) -> tuple[tuple[str, ...], ...]:
    return () if plan.expects == REFUSAL else ((plan.key,),)


def train_structure(rng: random.Random) -> Structure:
    return Structure(False, rng.choice(TRAIN_LAYOUTS), rng.choice((" ", "\n")))


def held_structure(rng: random.Random) -> Structure:
    return Structure(True, rng.choice(HELD_LAYOUTS), rng.choice((" ", "\n")))


@dataclass(frozen=True)
class Example:
    plan: Plan
    structure: Structure


def training_examples(language: str, count: int, seed: int, split: str = "train") -> list[Example]:
    rng = random.Random(f"{seed}:{language}:{split}:structures")
    examples: list[Example] = []
    for index, plan in enumerate(plans(language, split, seed, (count + 1) // 2)):
        structure = train_structure(rng)
        partner = without_target(plan) if index % 2 == 0 else distracted(plan)
        if plan.task == "cause" and index % 2:
            partner = without_target(plan)
        examples.extend([Example(plan, structure), Example(partner, structure)])
    return examples[:count]


def row_of(example: Example, style: int) -> dict:
    plan = example.plan
    options = answer_options(plan)
    category = "grounded_missing" if plan.expects == REFUSAL else f"grounded_{plan.task}"
    return {
        "source": "synthetic_grounded",
        "group": f"reading_{plan.language}",
        "category": category,
        "language": plan.language,
        "task": plan.task,
        "messages": [
            {"role": "user", "content": prompt_of(plan, example.structure)},
            {"role": "assistant", "content": options[style % len(options)]},
        ],
    }


def eval_links(language: str, split: str, seed: int) -> list[tuple[str, str, Plan, Structure]]:
    rng = random.Random(f"{seed}:{language}:{split}:eval")
    links: list[tuple[str, str, Plan, Structure]] = []
    sizes = {"seen": 6, "structure": 4}
    for cell, size in sizes.items():
        for task in TASKS:
            for number in range(size):
                relation = rng.choice(TRAIN_RELATIONS)
                plan = Builder(language, split, rng).plan(task, relation)
                if cell == "seen":
                    layout = TRAIN_LAYOUTS[number % 2]
                    structure = Structure(False, layout, rng.choice((" ", "\n")))
                else:
                    layout = HELD_LAYOUTS[number % 2]
                    structure = Structure(True, layout, rng.choice((" ", "\n")))
                links.append((f"{cell}/{task}/{number}", cell, plan, structure))
    for number in range(6):
        task = ("extract", "distractor", "relation")[number % 3]
        relation = rng.choice(TRAIN_RELATIONS)
        plan = Builder(language, split, rng).plan(task, relation, "alt")
        structure = Structure(False, TRAIN_LAYOUTS[number % 2], rng.choice((" ", "\n")))
        links.append((f"question/alt-{task}/{number}", "question", plan, structure))
    for number in range(6):
        task = ("extract", "distractor", "relation")[number % 3]
        relation = rng.choice(HELD_OUT_RELATIONS)
        plan = Builder(language, split, rng).plan(task, relation)
        structure = Structure(False, TRAIN_LAYOUTS[number % 2], rng.choice((" ", "\n")))
        links.append((f"question/relation-{task}/{number}", "question", plan, structure))
    return links


def case_of(
    link: str, cell: str, variant: str, plan: Plan, structure: Structure, split: str
) -> Case:
    return Case(
        f"{split}/{plan.language}/{link}/{variant}",
        f"{split}/{plan.language}/{link}",
        f"{split}/{plan.language}/{link}",
        variant,
        plan.task,
        cell,
        structure.layout,
        plan.language,
        prompt_of(plan, structure),
        plan.expects,
        accept_of(plan),
        plan.rejected,
    )


def eval_cases(split: str, seed: int | None = None) -> list[Case]:
    base = SEEDS[split] if seed is None else seed
    cases: list[Case] = []
    for language in LANGUAGES:
        rng = random.Random(f"{base}:{language}:{split}:restructure")
        for link, cell, plan, structure in eval_links(language, split, base):
            variants = [
                ("original", plan),
                ("changed", changed(plan)),
                ("missing", without_target(plan)),
            ]
            if plan.task not in ("cause", "match"):
                variants.append(("distractor", distracted(plan)))
            cases.extend(
                case_of(link, cell, name, item, structure, split) for name, item in variants
            )
            if cell == "seen":
                cases.append(case_of(link, cell, "restructured", plan, held_structure(rng), split))
    return cases


def trained_cases(seed: int, per_language: int = TRAINED_PER_LANGUAGE) -> list[Case]:
    cases: list[Case] = []
    for language in LANGUAGES:
        examples = training_examples(language, 2 * per_language, seed)
        for index, example in enumerate(examples):
            variant = "missing" if example.plan.expects == REFUSAL else "original"
            cases.append(
                case_of(
                    f"trained/{index // 2}",
                    "trained",
                    variant,
                    example.plan,
                    example.structure,
                    "train",
                )
            )
    return cases
