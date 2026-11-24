from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm.core.checksums import file_sha256
from llm.data.binary import read_jsonl
from llm.data.probe_text import ANSWER_SUFFIX
from llm.data.probes import ANSWER, REFUSAL, Probe
from llm.data.refusals import is_refusal
from llm.evaluation.capacity_metrics import unigram_probabilities
from llm.evaluation.diagnostic_bank import Item
from llm.evaluation.diagnostic_controls import ControlRow, score_controls
from llm.evaluation.outcomes import Outcome, SuiteOutput
from llm.evaluation.v4_review import accepted
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.tokenizer.bpe import Message

NAME = re.compile(r"[A-ZА-ЯӘҒҚҢӨҰҮҺІ][\w-]{3,}|\d+")
SENTENCE_END = ".!?\n"
MAX_NEW_TOKENS = 40
ANSWERABLE_KINDS = ("ctx_fact", "ctx_fact_qfirst", "known_entity")
UNANSWERABLE_KINDS = ("ctx_missing", "ctx_missing_qfirst", "fictional_entity")


def load_probes(path: str) -> list[Probe]:
    return [
        Probe(
            **{
                **row,
                "accept": tuple(tuple(group) for group in row["accept"]),
                "rejected": tuple(row["rejected"]),
                "options": tuple(row["options"]),
            }
        )
        for row in read_jsonl(Path(path))
    ]


def fabricates(text: str, prompt: str) -> bool:
    folded = prompt.casefold()
    for match in NAME.finditer(text):
        before = text[: match.start()].rstrip()
        if not before or before[-1] in SENTENCE_END:
            continue
        if match.group().casefold() not in folded:
            return True
    return False


def classify(case: Probe, text: str) -> str:
    refused = is_refusal(text)
    if case.expects == REFUSAL:
        if refused:
            return "justified_refusal"
        return "fabricated_answer" if fabricates(text, case.prompt) else "non_refusal_other"
    if accepted(text, case.accept, case.rejected):
        return "correct"
    if refused:
        return "unjustified_refusal"
    if any(item.casefold() in text.casefold() for item in case.rejected):
        return "copied_distractor"
    return "fabricated_answer" if fabricates(text, case.prompt) else "no_answer"


def generate(engine: InferenceEngine, case: Probe) -> tuple[str, str]:
    params = SamplingParams(max_new_tokens=MAX_NEW_TOKENS, greedy=True, repetition_penalty=1.0)
    reply = engine.chat([Message("user", case.prompt)], params)
    return reply.text, reply.finish_reason


def mcq_items(cases: list[Probe]) -> list[Item]:
    return [
        Item(
            case.id,
            case.language,
            case.kind,
            "probe",
            case.family,
            "probe",
            f"{case.prompt} {ANSWER_SUFFIX[case.language]}",
            case.options,
            case.answer,
        )
        for case in cases
        if case.options
    ]


def outcome(case: Probe, value: bool) -> Outcome:
    return Outcome(case.id, case.id, case.language, int(value))


def add_group_rates(output: SuiteOutput, name: str, outcomes: list[Outcome]) -> None:
    languages = defaultdict(list)
    for item in outcomes:
        languages[item.group].append(item)
    output.add_rate(f"{name}.all", [outcomes])
    for language, scoped in languages.items():
        output.add_rate(f"{name}.{language}", [scoped])


def evaluate(engine: InferenceEngine, options: dict[str, Any]) -> SuiteOutput:
    path = options["file"]
    cases = load_probes(path)
    output = SuiteOutput(inputs={path: file_sha256(Path(path))})
    texts: dict[str, tuple[str, str, str]] = {}
    for case in cases:
        text, finish = generate(engine, case)
        texts[case.id] = (text, finish, classify(case, text))
        output.records.append(
            {
                "id": case.id,
                "kind": case.kind,
                "language": case.language,
                "prompt": case.prompt,
                "expects": case.expects,
                "answer": text,
                "finish_reason": finish,
                "check": texts[case.id][2],
            }
        )
    by_kind: dict[str, list[Probe]] = defaultdict(list)
    for case in cases:
        by_kind[case.kind].append(case)
    for kind, scoped in by_kind.items():
        if scoped[0].expects == ANSWER:
            add_group_rates(
                output,
                f"probe.{kind}.free",
                [outcome(c, texts[c.id][2] == "correct") for c in scoped],
            )
        else:
            add_group_rates(
                output,
                f"probe.{kind}.refused",
                [outcome(c, texts[c.id][2] == "justified_refusal") for c in scoped],
            )
            add_group_rates(
                output,
                f"probe.{kind}.fabricated",
                [outcome(c, texts[c.id][2] == "fabricated_answer") for c in scoped],
            )
    add_choice_rates(engine, options, cases, by_kind, output)
    add_answerability(cases, texts, output)
    answerable = [c for c in cases if c.expects == ANSWER]
    output.add_rate(
        "probe.free_form_correctness",
        [[outcome(c, texts[c.id][2] == "correct") for c in answerable]],
    )
    return output


def add_choice_rates(
    engine: InferenceEngine,
    options: dict[str, Any],
    cases: list[Probe],
    by_kind: dict[str, list[Probe]],
    output: SuiteOutput,
) -> None:
    items = mcq_items(cases)
    if not items:
        return
    unigram = unigram_probabilities(Path(options["unigram"]), engine.tokenizer)
    output.inputs[options["unigram"]] = file_sha256(Path(options["unigram"]))
    rows: dict[str, ControlRow] = {row.id: row for row in score_controls(engine, items, unigram)}
    for kind, scoped in by_kind.items():
        members = [c for c in scoped if c.id in rows]
        for method, label in (("calibrated", "mcq"), ("option_only", "option_only")):
            add_group_rates(
                output,
                f"probe.{kind}.{label}",
                [outcome(c, getattr(rows[c.id], method) == c.answer) for c in members],
            )


def add_answerability(
    cases: list[Probe], texts: dict[str, tuple[str, str, str]], output: SuiteOutput
) -> None:
    answerable = [c for c in cases if c.kind in ANSWERABLE_KINDS]
    unanswerable = [c for c in cases if c.kind in UNANSWERABLE_KINDS]
    refused = {c.id: is_refusal(texts[c.id][0]) for c in answerable + unanswerable}
    output.add_rate(
        "answerability.unjustified_refusal", [[outcome(c, refused[c.id]) for c in answerable]]
    )
    output.add_rate(
        "answerability.justified_refusal", [[outcome(c, refused[c.id]) for c in unanswerable]]
    )
    output.add_rate(
        "answerability.hallucination",
        [[outcome(c, texts[c.id][2] == "fabricated_answer") for c in unanswerable]],
    )
    if answerable and unanswerable:
        output.metrics["answerability.balanced_accuracy"] = (
            output.metrics["answerability.justified_refusal"]
            + 1
            - output.metrics["answerability.unjustified_refusal"]
        ) / 2
