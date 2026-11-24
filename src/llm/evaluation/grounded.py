from __future__ import annotations

import glob
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from llm.core.checksums import file_sha256
from llm.core.errors import ConfigError
from llm.data.binary import read_jsonl
from llm.data.grounded import INVENTED, REFUSAL, Case
from llm.data.refusals import is_refusal
from llm.evaluation.outcomes import Outcome, SuiteOutput
from llm.evaluation.probes import fabricates, generate
from llm.evaluation.v4_review import accepted
from llm.inference.engine import InferenceEngine

ANSWERABLE = ("original", "changed", "restructured")
UNANSWERABLE = ("missing", "distractor")
RIVAL_TASKS = ("distractor", "relation", "negation", "update", "priority")
CONTRADICTION_TASKS = ("update", "priority")
TRAINED = "trained"
Check = Callable[[Case], bool]


def load_cases(path: str) -> list[Case]:
    return [
        Case(
            **{
                **row,
                "accept": tuple(tuple(group) for group in row["accept"]),
                "rejected": tuple(row["rejected"]),
            }
        )
        for row in read_jsonl(Path(path))
    ]


def judge(case: Case, text: str) -> str:
    refused = is_refusal(text)
    copied = any(item.casefold() in text.casefold() for item in case.rejected if item)
    if case.expects == REFUSAL:
        if refused:
            return "justified_refusal"
        if copied:
            return "copied_distractor"
        return "fabricated_answer" if fabricates(text, case.prompt) else "non_refusal_other"
    if accepted(text, case.accept, case.rejected):
        return "correct"
    if refused:
        return "unjustified_refusal"
    if copied:
        return "copied_distractor"
    return "fabricated_answer" if fabricates(text, case.prompt) else "no_answer"


def leaked_names(cases: Sequence[Case], training: Sequence[str]) -> set[str]:
    seen: set[str] = set()
    for pattern in training:
        for path in sorted(glob.glob(pattern)):
            for row in read_jsonl(Path(path)):
                for turn in row["messages"]:
                    seen.update(INVENTED.findall(turn["content"]))
    return {name for case in cases for name in INVENTED.findall(case.prompt)} & seen


def outcome(case: Case, value: bool) -> Outcome:
    return Outcome(case.id, case.family, case.language, int(value))


def link_outcome(link: str, language: str, value: bool) -> Outcome:
    return Outcome(link, link, language, int(value))


def add_rate(output: SuiteOutput, name: str, outcomes: list[Outcome], by_language: bool) -> None:
    output.add_rate(name, [outcomes])
    if by_language:
        languages: dict[str, list[Outcome]] = defaultdict(list)
        for item in outcomes:
            languages[item.group].append(item)
        for language, scoped in languages.items():
            output.add_rate(f"{name}.{language}", [scoped])


def evaluate(engine: InferenceEngine, options: dict[str, Any]) -> SuiteOutput:
    path = options["file"]
    cases = load_cases(path)
    output = SuiteOutput(inputs={path: file_sha256(Path(path))})
    training = options.get("training", [])
    leaks = leaked_names(cases, training)
    if leaks:
        raise ConfigError(f"evaluation names appear in training rows: {sorted(leaks)[:5]}")
    for pattern in training:
        for item in sorted(glob.glob(pattern)):
            output.inputs[item] = file_sha256(Path(item))
    checks: dict[str, str] = {}
    for case in cases:
        text, finish = generate(engine, case.prompt)
        checks[case.id] = judge(case, text)
        output.records.append(
            {
                "id": case.id,
                "kind": case.variant,
                "language": case.language,
                "cell": case.cell,
                "task": case.task,
                "layout": case.layout,
                "prompt": case.prompt,
                "expects": case.expects,
                "answer": text,
                "finish_reason": finish,
                "check": checks[case.id],
            }
        )
    held = [case for case in cases if case.cell != TRAINED]
    add_metrics(output, held, checks)
    add_trained(output, [case for case in cases if case.cell == TRAINED], checks)
    return output


def correct(case: Case, checks: dict[str, str]) -> bool:
    return checks[case.id] == "correct"


def refused(case: Case, checks: dict[str, str]) -> bool:
    return checks[case.id] == "justified_refusal" or checks[case.id] == "unjustified_refusal"


def add_trained(output: SuiteOutput, cases: list[Case], checks: dict[str, str]) -> None:
    for variant, label in (("original", "accuracy"), ("missing", "refusal")):
        scoped = [case for case in cases if case.variant == variant]
        add_rate(
            output,
            f"ctx.trained.{label}",
            [
                outcome(
                    case,
                    correct(case, checks) if variant == "original" else refused(case, checks),
                )
                for case in scoped
            ],
            True,
        )


def add_metrics(output: SuiteOutput, cases: list[Case], checks: dict[str, str]) -> None:
    by_variant: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        by_variant[case.variant].append(case)
    original = by_variant["original"]
    add_rate(output, "ctx.fact_accuracy", [outcome(c, correct(c, checks)) for c in original], True)
    for cell in sorted({c.cell for c in original}):
        add_rate(
            output,
            f"ctx.fact_accuracy.cell.{cell}",
            [outcome(c, correct(c, checks)) for c in original if c.cell == cell],
            True,
        )
    for task in sorted({c.task for c in original}):
        add_rate(
            output,
            f"ctx.fact_accuracy.task.{task}",
            [outcome(c, correct(c, checks)) for c in original if c.task == task],
            False,
        )
    for layout in sorted({c.layout for c in original}):
        add_rate(
            output,
            f"ctx.fact_accuracy.layout.{layout}",
            [outcome(c, correct(c, checks)) for c in original if c.layout == layout],
            False,
        )
    add_rate(
        output,
        "ctx.changed_accuracy",
        [outcome(c, correct(c, checks)) for c in by_variant["changed"]],
        True,
    )
    add_rate(
        output,
        "ctx.restructured_accuracy",
        [outcome(c, correct(c, checks)) for c in by_variant["restructured"]],
        True,
    )
    add_intervention(output, by_variant, checks)
    add_resistance(output, by_variant, checks)
    add_answerability(output, by_variant, checks)
    answerable = [c for name in ANSWERABLE for c in by_variant[name]]
    add_rate(
        output,
        "ctx.free_form_correctness",
        [outcome(c, correct(c, checks)) for c in answerable],
        True,
    )
    add_rate(
        output,
        "ctx.contradiction_handling",
        [outcome(c, correct(c, checks)) for c in original if c.task in CONTRADICTION_TASKS],
        True,
    )


def add_intervention(
    output: SuiteOutput, by_variant: dict[str, list[Case]], checks: dict[str, str]
) -> None:
    changed = {c.link: c for c in by_variant["changed"]}
    both = [
        link_outcome(c.link, c.language, correct(c, checks) and correct(changed[c.link], checks))
        for c in by_variant["original"]
    ]
    add_rate(output, "ctx.intervention_accuracy", both, True)
    right = [c for c in by_variant["original"] if correct(c, checks)]
    if right:
        output.metrics["ctx.sensitivity"] = sum(
            correct(changed[c.link], checks) for c in right
        ) / len(right)


def add_resistance(
    output: SuiteOutput, by_variant: dict[str, list[Case]], checks: dict[str, str]
) -> None:
    exposed = [c for c in by_variant["original"] if c.task in RIVAL_TASKS] + by_variant[
        "distractor"
    ]
    add_rate(
        output,
        "ctx.distractor_resistance",
        [outcome(c, checks[c.id] != "copied_distractor") for c in exposed],
        True,
    )


def add_answerability(
    output: SuiteOutput, by_variant: dict[str, list[Case]], checks: dict[str, str]
) -> None:
    answerable = [c for name in ANSWERABLE for c in by_variant[name]]
    unanswerable = [c for name in UNANSWERABLE for c in by_variant[name]]
    for variant in UNANSWERABLE:
        add_rate(
            output,
            f"ctx.missing_detection.{variant}",
            [outcome(c, checks[c.id] == "justified_refusal") for c in by_variant[variant]],
            True,
        )
    add_rate(
        output,
        "ctx.justified_refusal",
        [outcome(c, checks[c.id] == "justified_refusal") for c in unanswerable],
        True,
    )
    add_rate(
        output,
        "ctx.unjustified_refusal",
        [outcome(c, checks[c.id] == "unjustified_refusal") for c in answerable],
        True,
    )
    add_rate(
        output,
        "ctx.hallucination",
        [
            outcome(c, checks[c.id] in ("fabricated_answer", "copied_distractor"))
            for c in unanswerable
        ],
        True,
    )
    output.metrics["ctx.balanced_accuracy"] = (
        output.metrics["ctx.justified_refusal"] + 1 - output.metrics["ctx.unjustified_refusal"]
    ) / 2
