from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from llm.data.grounded import (
    ANSWER,
    BINDING_TASKS,
    HELD_LAYOUTS,
    INVENTED,
    REFUSAL,
    TASKS,
    TRAIN_LAYOUTS,
    Builder,
    Case,
    Structure,
    case_of,
    changed,
    distracted,
    eval_cases,
    prompt_of,
    training_examples,
    without_target,
)
from llm.data.grounded_text import HOLDOUT_RELATIONS, LANGUAGES, TRAIN_RELATIONS
from llm.data.natural_text import NATURAL
from llm.data.probes import write_context_probes

HOLDOUT_SEED = 9091
HOLDOUT_PROBE_SEED = 7919
HOLDOUT_PROBE_DOCUMENTS = 40
SPLIT = "test"
LABEL = "holdout"
FRAME_TASKS = ("extract", "distractor", "relation", "binding")
KNOWN_POOL = 7_000
LAYOUT_JOINERS = (" ", "\n")
CELL_SIZES = {"world": 4, "structure": 6}


@lru_cache(maxsize=1)
def known_names() -> frozenset[str]:
    names: set[str] = set()
    for language in LANGUAGES:
        for tasks in (TASKS, BINDING_TASKS):
            for example in training_examples(language, KNOWN_POOL, 1, tasks=tasks):
                names.update(INVENTED.findall(prompt_of(example.plan, example.structure)))
    for split in ("dev", "test"):
        for case in eval_cases(split):
            names.update(INVENTED.findall(case.prompt))
    return frozenset(names)


def structure_of(rng: random.Random, number: int, kind: str) -> Structure:
    joiner = rng.choice(LAYOUT_JOINERS)
    if kind == "train":
        return Structure(False, TRAIN_LAYOUTS[number % 2], joiner)
    return Structure(True, HELD_LAYOUTS[number % 2], joiner, True)


def holdout_links(
    language: str, seed: int, exclude: frozenset[str]
) -> list[tuple[str, str, Builder, str, Structure, str]]:
    rng = random.Random(f"{seed}:{language}:holdout")
    links = []

    def add(
        name: str,
        cell: str,
        task: str,
        relation: str,
        number: int,
        kind: str,
        question: str = "ask",
    ) -> None:
        builder = Builder(language, SPLIT, rng, exclude)
        links.append(
            (
                f"{cell}/{name}/{number}",
                cell,
                builder,
                task,
                structure_of(rng, number, kind),
                relation + "|" + question,
            )
        )

    for task in (*TASKS, "binding"):
        for number in range(CELL_SIZES["world"]):
            add(task, "world", task, rng.choice(TRAIN_RELATIONS), number, "train")
    for task in FRAME_TASKS:
        for number in range(CELL_SIZES["structure"]):
            add(task, "structure", task, rng.choice(TRAIN_RELATIONS), number, "holdout")
    for relation in HOLDOUT_RELATIONS:
        for task in FRAME_TASKS:
            add(f"{relation}-{task}", "relation", task, relation, 0, "train")
    for number in range(6):
        task = ("extract", "distractor", "relation")[number % 3]
        add(f"alt-{task}", "combination", task, rng.choice(TRAIN_RELATIONS), number, "train", "alt")
    return links


def holdout_cases(seed: int = HOLDOUT_SEED) -> list[Case]:
    exclude = known_names()
    cases: list[Case] = []
    for language in LANGUAGES:
        rng = random.Random(f"{seed}:{language}:restructure")
        for link, cell, builder, task, structure, spec in holdout_links(language, seed, exclude):
            relation, question = spec.split("|")
            plan = builder.plan(task, relation, question)
            variants = [
                ("original", plan),
                ("changed", changed(plan)),
                ("missing", without_target(plan)),
            ]
            if task not in ("cause", "match"):
                variants.append(("distractor", distracted(plan)))
            cases.extend(
                case_of(link, cell, name, item, structure, LABEL) for name, item in variants
            )
            if cell == "world":
                cases.append(
                    case_of(
                        link, cell, "restructured", plan, structure_of(rng, 1, "holdout"), LABEL
                    )
                )
    return cases


def natural_cases() -> list[Case]:
    cases: list[Case] = []
    counters = dict.fromkeys(LANGUAGES, 0)
    for passage in NATURAL:
        number = counters[passage.language]
        counters[passage.language] += 1
        layout = TRAIN_LAYOUTS[number % 2]
        link = f"{LABEL}/{passage.language}/natural/{number}"
        for variant, value, question, expects, accept, rejected in (
            ("original", passage.original, passage.question, ANSWER, passage.original, ()),
            (
                "changed",
                passage.changed,
                passage.question,
                ANSWER,
                passage.changed,
                (passage.original,),
            ),
            ("missing", passage.original, passage.missing, REFUSAL, "", ()),
        ):
            text = passage.text.format(V=value)
            prompt = text + "\n\n" + question if layout == "dfirst" else question + "\n\n" + text
            cases.append(
                Case(
                    f"{link}/{variant}",
                    link,
                    link,
                    variant,
                    "natural",
                    "natural",
                    layout,
                    passage.language,
                    prompt,
                    expects,
                    ((accept,),) if accept else (),
                    rejected,
                )
            )
    return cases


def write_jsonl(path: Path, cases: list[Case]) -> str:
    text = "".join(json.dumps(asdict(case), ensure_ascii=False) + "\n" for case in cases)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_holdout(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    grounded = holdout_cases()
    natural = natural_cases()
    probe_summary = write_context_probes(
        output / "probes", HOLDOUT_PROBE_DOCUMENTS, HOLDOUT_PROBE_SEED
    )
    probes_hash = hashlib.sha256((output / "probes" / "probes.jsonl").read_bytes()).hexdigest()
    manifest = {
        "grounded": {
            "file": "holdout.jsonl",
            "sha256": write_jsonl(output / "holdout.jsonl", grounded),
            "cases": len(grounded),
            "families": len({case.family for case in grounded}),
            "seed": HOLDOUT_SEED,
        },
        "natural": {
            "file": "natural.jsonl",
            "sha256": write_jsonl(output / "natural.jsonl", natural),
            "cases": len(natural),
            "families": len({case.family for case in natural}),
        },
        "probes": {**probe_summary, "sha256": probes_hash, "file": "probes/probes.jsonl"},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
