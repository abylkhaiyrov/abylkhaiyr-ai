from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence

import torch
from torch import Tensor

from llm.data.grounded import Case
from llm.data.grounded_text import ANSWERS
from llm.inference.engine import InferenceEngine
from llm.tokenizer.bpe import Message

LENS_TASKS = ("extract", "distractor", "relation", "match", "negation", "update", "priority")
ANSWER_PREFIX_STYLE = 2


def prompt_ids(engine: InferenceEngine, case: Case, prefix: str = "") -> list[int]:
    ids = engine.tokenizer.encode_conversation(
        [Message("user", case.prompt)], add_generation_prompt=True
    )
    return ids + engine.tokenizer.encode(prefix) if prefix else ids


def answer_prefix(language: str) -> str:
    template = ANSWERS[language][ANSWER_PREFIX_STYLE]
    return template.split("{X}")[0]


def capture(engine: InferenceEngine, ids: Sequence[int]) -> list[Tensor]:
    model = engine.model
    states: list[Tensor] = []

    def keep(_module: torch.nn.Module, _inputs: object, output: Tensor) -> None:
        states.append(output[0, -1].detach().float().cpu())

    handles = [model.embedding.register_forward_hook(keep)]
    handles += [block.register_forward_hook(keep) for block in model.blocks]
    try:
        with torch.no_grad():
            model(torch.tensor([list(ids)], device=engine.device))
    finally:
        for handle in handles:
            handle.remove()
    return states


def readout(engine: InferenceEngine, state: Tensor) -> Tensor:
    model = engine.model
    with torch.no_grad():
        normed = model.final_norm(state.to(engine.device))
        return torch.log_softmax(model.lm_head(normed).float(), dim=-1).cpu()


def first_token(engine: InferenceEngine, text: str) -> int:
    return engine.tokenizer.encode(text)[0]


def lens_rows(
    engine: InferenceEngine, case: Case, rivals: Sequence[str]
) -> list[tuple[float, float]]:
    key = case.accept[0][0]
    key_id = first_token(engine, key)
    rival_ids = [first_token(engine, item) for item in rivals if item]
    states = capture(engine, prompt_ids(engine, case, answer_prefix(case.language)))
    rows = []
    for state in states:
        logprobs = readout(engine, state)
        best = max((logprobs[token].item() for token in rival_ids), default=float("-inf"))
        rows.append((logprobs[key_id].item(), best))
    return rows


def cosine_distance(first: Tensor, second: Tensor) -> float:
    return 1.0 - torch.nn.functional.cosine_similarity(first, second, dim=0).item()


def centroid_accuracy(train: list[tuple[Tensor, int]], test: list[tuple[Tensor, int]]) -> float:
    centroids = {
        label: torch.stack([vector for vector, tag in train if tag == label]).mean(0)
        for label in (0, 1)
    }
    hits = sum(
        int(min(centroids, key=lambda label: (vector - centroids[label]).norm().item()) == label)
        for vector, label in test
    )
    return hits / len(test)


def diagnose(engine: InferenceEngine, cases: Sequence[Case], limit: int) -> dict[str, object]:
    by_link: dict[str, dict[str, Case]] = defaultdict(dict)
    for case in cases:
        by_link[case.link][case.variant] = case
    links = [
        variants
        for variants in by_link.values()
        if {"original", "changed", "missing"} <= set(variants)
        and variants["original"].task in LENS_TASKS
    ][:limit]
    layers = len(engine.model.blocks) + 1
    lens: dict[str, list[list[tuple[float, float]]]] = {"original": [], "changed": []}
    shifts: dict[str, list[list[float]]] = defaultdict(list)
    answerable: list[tuple[list[Tensor], int]] = []
    previous: list[Tensor] | None = None
    for position, variants in enumerate(links):
        states = {
            name: capture(engine, prompt_ids(engine, variants[name]))
            for name in ("original", "changed", "missing", "restructured")
            if name in variants
        }
        for name in ("original", "changed"):
            lens[name].append(lens_rows(engine, variants[name], variants[name].rejected))
        for name in ("changed", "missing", "restructured"):
            if name in states:
                shifts[name].append(
                    [
                        cosine_distance(a, b)
                        for a, b in zip(states["original"], states[name], strict=True)
                    ]
                )
        if previous is not None:
            shifts["unrelated"].append(
                [cosine_distance(a, b) for a, b in zip(states["original"], previous, strict=True)]
            )
        previous = states["original"]
        answerable.append((states["original"], 1))
        answerable.append((states["missing"], 0))
    report: dict[str, object] = {"links": len(links), "layers": layers}
    for name, groups in lens.items():
        report[f"lens_{name}"] = [
            {
                "layer": layer,
                "key_logprob": statistics.fmean(g[layer][0] for g in groups),
                "rival_logprob": statistics.fmean(g[layer][1] for g in groups),
                "key_ahead": statistics.fmean(float(g[layer][0] > g[layer][1]) for g in groups),
            }
            for layer in range(layers)
        ]
    report["shift"] = {
        name: [statistics.fmean(row[layer] for row in rows) for layer in range(layers)]
        for name, rows in shifts.items()
    }
    half = len(answerable) // 2
    first_half, second_half = answerable[:half], answerable[half:]
    report["answerability_decodable"] = [
        statistics.fmean(
            [
                centroid_accuracy(
                    [(states[layer], tag) for states, tag in source],
                    [(states[layer], tag) for states, tag in target],
                )
                for source, target in ((first_half, second_half), (second_half, first_half))
            ]
        )
        for layer in range(layers)
    ]
    return report
