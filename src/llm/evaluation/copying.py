from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from llm.core.checksums import file_sha256
from llm.data.binary import read_jsonl
from llm.data.copy_ladder import ANSWER_LABEL, CopyCase, answer_frame
from llm.evaluation.outcomes import Outcome, SuiteOutput
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.tokenizer.bpe import Message

MAX_NEW_TOKENS = 24
STRIP = ' .!,;:"«»\n'
CHAT, PLAIN = "chat", "plain"


@dataclass(frozen=True)
class Probe:
    prompt_ids: list[int]
    target_ids: list[int]
    text: str
    eos: bool = False


@dataclass(frozen=True)
class Forced:
    first_probability: float
    first_loss: float
    rest_loss: float | None
    token_accuracy: float
    exact: bool
    eos_probability: float | None
    id_first: bool
    id_all: bool


def load_cases(path: str) -> list[CopyCase]:
    return [CopyCase(**{**row, "rivals": tuple(row["rivals"])}) for row in read_jsonl(Path(path))]


def probe_of(engine: InferenceEngine, case: CopyCase, form: str, framed: bool = False) -> Probe:
    tokenizer = engine.tokenizer
    frame = answer_frame(case) if framed else ""
    target = f" {case.target}" if frame else case.target
    if form == CHAT:
        prompt = tokenizer.encode_conversation(
            [Message("user", case.prompt)], add_generation_prompt=True
        )
        framed_ids = tokenizer.encode(frame) if frame else []
        return Probe(
            prompt + framed_ids,
            tokenizer.encode(target),
            case.prompt,
            eos=not framed,
        )
    text = f"{case.prompt}\n{ANSWER_LABEL[case.language]}"
    if frame:
        text = f"{text} {frame}"
        return Probe(tokenizer.encode(text, add_bos=True), tokenizer.encode(target), text)
    return Probe(tokenizer.encode(text, add_bos=True), tokenizer.encode(f" {case.target}"), text)


def forced(engine: InferenceEngine, probe: Probe) -> Forced:
    target = probe.target_ids + ([engine.tokenizer.eos_id] if probe.eos else [])
    sequence = [*probe.prompt_ids, *target]
    with torch.no_grad():
        logits = engine.model(torch.tensor([sequence[:-1]], device=engine.device)).float()
    logprobs = torch.log_softmax(logits[0], dim=-1)
    start = len(probe.prompt_ids) - 1
    chosen = [logprobs[start + i, token].item() for i, token in enumerate(target)]
    hits = [int(logprobs[start + i].argmax().item() == token) for i, token in enumerate(target)]
    count = len(probe.target_ids)
    prompt_set = set(probe.prompt_ids)
    return Forced(
        first_probability=float(torch.tensor(chosen[0]).exp()),
        first_loss=-chosen[0],
        rest_loss=-statistics.fmean(chosen[1:count]) if count > 1 else None,
        token_accuracy=statistics.fmean(hits[:count]),
        exact=all(hits[:count]),
        eos_probability=float(torch.tensor(chosen[-1]).exp()) if probe.eos else None,
        id_first=probe.target_ids[0] in prompt_set,
        id_all=set(probe.target_ids) <= prompt_set,
    )


def generate(engine: InferenceEngine, case: CopyCase, probe: Probe, form: str) -> tuple[str, str]:
    params = SamplingParams(max_new_tokens=MAX_NEW_TOKENS, greedy=True, repetition_penalty=1.0)
    if form == CHAT:
        reply = engine.chat([Message("user", case.prompt)], params)
        return reply.text, reply.finish_reason
    reply = engine.generate(probe.text, params)
    return reply.text.split("\n")[0], reply.finish_reason


def judged(case: CopyCase, answer: str) -> dict[str, bool]:
    exact = answer.strip(STRIP) == case.target
    holds = case.target in answer
    rival = any(item in answer for item in case.rivals)
    return {
        "exact": exact,
        "contains": holds and not rival,
        "rival": rival and not holds,
    }


def scopes(case: CopyCase) -> list[str]:
    return [f"L{case.level}", f"L{case.level}.{case.category}", f"L{case.level}.{case.language}"]


def add_rates(output: SuiteOutput, name: str, outcomes: dict[str, list[Outcome]]) -> None:
    for scope, items in outcomes.items():
        output.add_rate(f"{name}.{scope}", [items])


def add_means(output: SuiteOutput, name: str, values: dict[str, list[float]]) -> None:
    for scope, items in values.items():
        if items:
            output.metrics[f"{name}.{scope}"] = statistics.fmean(items)


def evaluate(engine: InferenceEngine, options: dict[str, Any]) -> SuiteOutput:
    path = options["file"]
    cases = load_cases(path)
    output = SuiteOutput(inputs={path: file_sha256(Path(path))})
    framed = bool(options.get("framed", False))
    for form in options.get("formats", [CHAT, PLAIN]):
        rates: dict[str, dict[str, list[Outcome]]] = defaultdict(lambda: defaultdict(list))
        means: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for case in cases:
            probe = probe_of(engine, case, form)
            score = forced(engine, probe)
            answer, finish = generate(engine, case, probe, form)
            verdict = judged(case, answer) | {"tf_seq": score.exact}
            values = {
                "tf_first": score.first_probability,
                "tf_loss_first": score.first_loss,
                "tf_tok": score.token_accuracy,
                "id_first": float(score.id_first),
                "id_all": float(score.id_all),
            }
            if score.rest_loss is not None:
                values["tf_loss_rest"] = score.rest_loss
            if score.eos_probability is not None:
                values["tf_eos"] = score.eos_probability
            if framed and case.level > 1:
                shaped = forced(engine, probe_of(engine, case, form, True))
                verdict["fr_seq"] = shaped.exact
                values["fr_first"] = shaped.first_probability
                values["fr_loss_first"] = shaped.first_loss
                values["fr_tok"] = shaped.token_accuracy
                if shaped.rest_loss is not None:
                    values["fr_loss_rest"] = shaped.rest_loss
            for scope in scopes(case):
                for metric, flag in verdict.items():
                    rates[metric][scope].append(
                        Outcome(case.id, case.family, case.language, int(flag))
                    )
                for metric, number in values.items():
                    means[metric][scope].append(number)
            output.records.append(
                {
                    "id": case.id,
                    "format": form,
                    "kind": f"L{case.level}",
                    "language": case.language,
                    "category": case.category,
                    "prompt": case.prompt,
                    "target": case.target,
                    "answer": answer,
                    "finish_reason": finish,
                    **{key: bool(flag) for key, flag in verdict.items()},
                    "tf_first": round(score.first_probability, 6),
                }
            )
        for metric, by_scope in rates.items():
            add_rates(output, f"copy.{form}.{metric}", by_scope)
        for metric, averages in means.items():
            add_means(output, f"copy.{form}.{metric}", averages)
    return prefixed(output, str(options.get("prefix", "")))


def prefixed(output: SuiteOutput, prefix: str) -> SuiteOutput:
    if not prefix:
        return output
    return SuiteOutput(
        {f"{prefix}{name}": value for name, value in output.metrics.items()},
        {f"{prefix}{name}": runs for name, runs in output.items.items()},
        output.inputs,
        output.records,
    )
