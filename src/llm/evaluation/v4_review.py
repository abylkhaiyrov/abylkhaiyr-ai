from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterator, Sequence

from pydantic import BaseModel

from llm.evaluation.metrics import normalize_words, prompt_echo, repeated_ngram_ratio
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.tokenizer.bpe import Message

RUBRIC = (
    "conversational_pass",
    "naturalness",
    "grammar",
    "coherence",
    "semantic_relevance",
    "instruction_following",
    "multi_turn_accuracy",
    "context_retention",
    "hallucination",
    "unsupported_refusal",
    "mathematical_accuracy",
    "code_pass_at_1",
)


class Review(BaseModel, extra="forbid"):
    id: str
    response_sha256: str
    ratings: dict[str, bool | None]


def proportion(successes: int, size: int) -> dict:
    if size == 0:
        return {"n": 0, "successes": 0, "rate": None, "wilson_95": None}
    z = 1.959963984540054
    rate = successes / size
    slope = 1 + z * z / size
    center = (rate + z * z / (2 * size)) / slope
    half_width = z * math.sqrt(rate * (1 - rate) / size + z * z / (4 * size**2)) / slope
    return {
        "n": size,
        "successes": successes,
        "rate": rate,
        "wilson_95": [max(0.0, center - half_width), min(1.0, center + half_width)],
    }


SCRIPT_EXEMPT = {"translation", "code", "math"}
KAZAKH_LETTERS = frozenset("әғқңөұүһі")


def accepted(response: str, groups: Sequence[Sequence[str]], rejected: Sequence[str] = ()) -> bool:
    folded = response.casefold()
    return all(
        any(option.casefold() in folded for option in group) for group in groups
    ) and not any(option.casefold() in folded for option in rejected)


def script_matches(language: str, response: str) -> bool:
    letters = [char for char in response if char.isalpha()]
    if not letters:
        return False
    cyrillic = sum("\u0400" <= char <= "\u04ff" for char in letters) / len(letters)
    if language == "en":
        return cyrillic < 0.2
    if language == "ru":
        return cyrillic >= 0.8 and not KAZAKH_LETTERS & set(response.casefold())
    return cyrillic >= 0.8


def predictions(
    engine: InferenceEngine,
    rows: Sequence[dict],
    params: SamplingParams,
    batch_size: int,
) -> Iterator[dict]:
    inputs = [(row, [Message(**message) for message in row["messages"][:-1]]) for row in rows]
    inputs.sort(key=lambda item: engine.chat_prompt_tokens(item[1], params.max_new_tokens))
    for offset in range(0, len(inputs), batch_size):
        batch = inputs[offset : offset + batch_size]
        generated = engine.chat_batch([messages for _, messages in batch], params)
        for (row, messages), result in zip(batch, generated, strict=True):
            finish = result.finish_reason
            yield {
                "id": row["id"],
                "source": row["source"],
                "language": row["language"],
                "category": row["category"],
                "messages": row["messages"][:-1],
                "reference": row["messages"][-1]["content"],
                "accept": row.get("accept"),
                "reject": row.get("reject") or [],
                "response": result.text,
                "response_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
                "stop": finish,
                "generated_tokens": len(result.token_ids),
                "prompt_echo": prompt_echo(messages[-1].content, result.text),
                "repetition": repeated_ngram_ratio(normalize_words(result.text)),
            }


def metrics(rows: Sequence[dict], reviews: Sequence[Review] = ()) -> dict:
    index = {row["id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate prediction IDs")
    tally: Counter[str] = Counter()
    eligible: Counter[str] = Counter()
    seen: set[str] = set()
    for review in reviews:
        if review.id in seen:
            raise ValueError(f"duplicate review: {review.id}")
        row = index.get(review.id)
        if row is None or row["response_sha256"] != review.response_sha256:
            raise ValueError(f"review does not match prediction: {review.id}")
        seen.add(review.id)
        for metric, value in review.ratings.items():
            if metric not in RUBRIC:
                raise ValueError(f"unknown rating: {metric}")
            if value is not None:
                eligible[metric] += 1
                tally[metric] += value
    single = [row for row in rows if row.get("accept")]
    scripted = [row for row in rows if row["category"] not in SCRIPT_EXEMPT]
    return {
        "single_answer_accuracy": proportion(
            sum(accepted(row["response"], row["accept"], row.get("reject", ())) for row in single),
            len(single),
        ),
        "script_match": proportion(
            sum(script_matches(row["language"], row["response"]) for row in scripted),
            len(scripted),
        ),
        "incomplete": proportion(sum(row["stop"] != "stop" for row in rows), len(rows)),
        "prompt_echo": proportion(sum(row["prompt_echo"] for row in rows), len(rows)),
        "nonempty": proportion(sum(bool(row["response"].strip()) for row in rows), len(rows)),
        "mean_repetition": sum(row["repetition"] for row in rows) / len(rows) if rows else None,
        "reviewed_cases": len(seen),
        "human_rubric": {key: proportion(tally[key], eligible[key]) for key in RUBRIC},
    }
