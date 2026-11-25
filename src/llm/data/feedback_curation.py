from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from llm.data.verified_corpus import language_ok, prompt_key, user_sequence_key
from llm.evaluation.acceptance import ERROR_CAUSES

Stage = Literal["captured", "classified", "corrected", "verified", "curated", "blocked", "rejected"]
VerificationKind = Literal["human_reviewer", "arithmetic", "knowledge_table"]
MAX_REDACTIONS = 3
MAX_TEXT = 4000
RETENTION_LIMIT = 1.25
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("secret", re.compile(r"\b(?:sk|pk|ghp|gho|xox[bp]|AKIA)[-_A-Za-z0-9]{12,}\b")),
    ("jwt", re.compile(r"\beyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}\b")),
    (
        "credential",
        re.compile(r"(?i)\b(?:password|passwd|pwd|пароль|құпиясөз|token|secret)\s*[:=]\s*\S+"),
    ),
    ("url", re.compile(r"https?://\S+")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("national_id", re.compile(r"\b\d{12}\b")),
    (
        "phone",
        re.compile(
            r"(?<!\w)\+\d[\d\s().-]{8,}\d|(?<!\w)8[\s(-]?\d{3}[)\s-]*\d{3}[\s-]?\d{2}[\s-]?\d{2}\b"
        ),
    ),
)
ARITHMETIC = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/×÷])\s*(-?\d+(?:\.\d+)?)")
OPERATORS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "×": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "÷": lambda a, b: a / b,
}


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: VerificationKind
    verifier: str = Field(min_length=1)
    evidence: str = Field(min_length=1, max_length=MAX_TEXT)


class FeedbackRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: str
    captured_at: str
    messages: list[dict[str, str]]
    response: str
    checkpoint_id: str
    profile: str
    rating: Literal[-1, 1]
    comment: str | None = None
    consent: bool
    stage: Stage = "captured"
    error_class: str | None = None
    correction: str | None = None
    verification: Verification | None = None
    redactions: int = 0
    block_reason: str | None = None


def scrub(text: str) -> tuple[str, int]:
    count = 0
    for label, pattern in PATTERNS:
        text, replaced = pattern.subn(f"<{label}>", text)
        count += replaced
    return text, count


def record_identifier(messages: list[dict[str, str]], response: str, checkpoint: str) -> str:
    payload = json.dumps([messages, response, checkpoint], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def capture(
    messages: list[dict[str, str]],
    response: str,
    checkpoint_id: str,
    profile: str,
    rating: Literal[-1, 1],
    consent: bool,
    comment: str | None = None,
) -> FeedbackRecord | None:
    if not consent:
        return None
    redactions = 0
    cleaned: list[dict[str, str]] = []
    for message in messages:
        text, count = scrub(message["content"][:MAX_TEXT])
        redactions += count
        cleaned.append({"role": message["role"], "content": text})
    answer, count = scrub(response[:MAX_TEXT])
    redactions += count
    note = None
    if comment:
        note, count = scrub(comment[:MAX_TEXT])
        redactions += count
    record = FeedbackRecord(
        record_id=record_identifier(cleaned, answer, checkpoint_id),
        captured_at=datetime.now(UTC).isoformat(),
        messages=cleaned,
        response=answer,
        checkpoint_id=checkpoint_id,
        profile=profile,
        rating=rating,
        comment=note,
        consent=True,
        redactions=redactions,
    )
    if redactions > MAX_REDACTIONS:
        return record.model_copy(update={"stage": "blocked", "block_reason": "too_much_pii"})
    return record


def auto_error_class(record: FeedbackRecord, language: str) -> str | None:
    question = record.messages[-1]["content"]
    if prompt_key(record.response) == prompt_key(question):
        return "echo_or_repeat"
    if not language_ok(record.response, language):
        return "language_mismatch"
    return None


def classify(record: FeedbackRecord, error_class: str) -> FeedbackRecord:
    if error_class not in ERROR_CAUSES:
        raise ValueError(f"unknown error class: {error_class}")
    if record.stage != "captured":
        raise ValueError("only captured records can be classified")
    return record.model_copy(update={"stage": "classified", "error_class": error_class})


def attach_correction(
    record: FeedbackRecord, correction: str, verification: Verification
) -> FeedbackRecord:
    if record.stage != "classified":
        raise ValueError("only classified records can receive a correction")
    cleaned, count = scrub(correction[:MAX_TEXT])
    if count:
        raise ValueError("correction contains personal data or secrets")
    if verification.verifier == record.checkpoint_id or verification.verifier.startswith("model:"):
        raise ValueError("a model cannot verify its own answer")
    return record.model_copy(
        update={"stage": "corrected", "correction": cleaned, "verification": verification}
    )


def arithmetic_holds(query: str, correction: str) -> bool:
    match = ARITHMETIC.search(query)
    if match is None:
        return False
    left, symbol, right = float(match[1]), match[2], float(match[3])
    if symbol in "/÷" and right == 0:
        return False
    expected = OPERATORS[symbol](left, right)
    numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", correction)]
    return any(abs(value - expected) < 1e-9 for value in numbers)


def verify(record: FeedbackRecord, table: dict[str, str] | None = None) -> FeedbackRecord:
    if record.stage != "corrected" or record.verification is None or record.correction is None:
        raise ValueError("only corrected records can be verified")
    kind = record.verification.kind
    query = record.messages[-1]["content"]
    if kind == "arithmetic" and not arithmetic_holds(query, record.correction):
        return record.model_copy(update={"stage": "rejected", "block_reason": "arithmetic_failed"})
    if kind == "knowledge_table":
        expected = (table or {}).get(prompt_key(query))
        if expected is None or prompt_key(expected) != prompt_key(record.correction):
            return record.model_copy(update={"stage": "rejected", "block_reason": "table_mismatch"})
    return record.model_copy(update={"stage": "verified"})


class StoreError(Exception):
    pass


class FeedbackStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: FeedbackRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def latest(self) -> dict[str, FeedbackRecord]:
        records: dict[str, FeedbackRecord] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                record = FeedbackRecord.model_validate_json(line)
                records[record.record_id] = record
        return records


def sft_row(record: FeedbackRecord, version: str) -> dict[str, Any]:
    assert record.correction is not None and record.verification is not None
    return {
        "source": f"feedback_{version}",
        "group": "feedback",
        "category": record.error_class,
        "language": "und",
        "entity": f"feedback:{record.record_id}",
        "split": "train",
        "provenance": {
            "record_id": record.record_id,
            "captured_at": record.captured_at,
            "checkpoint_id": record.checkpoint_id,
            "verification": record.verification.model_dump(),
            "consent": True,
            "redactions": record.redactions,
        },
        "license": "user-contributed with explicit consent",
        "messages": [
            *record.messages,
            {"role": "assistant", "content": record.correction},
        ],
    }


def curate(
    store: FeedbackStore,
    out_dir: Path,
    version: str,
    protected: set[str],
    languages: dict[str, str] | None = None,
    parent: str | None = None,
    training_budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = out_dir / f"corrections.{version}.jsonl"
    if target.exists():
        raise StoreError(f"dataset version already exists: {version}")
    rejected: Counter[str] = Counter()
    chosen: dict[str, FeedbackRecord] = {}
    corrections: dict[str, str] = {}
    conflicts: set[str] = set()
    for record in store.latest().values():
        if record.stage != "verified" or not record.consent:
            rejected["not_verified_or_no_consent"] += 1
            continue
        key = user_sequence_key(record.messages)
        if key in protected:
            rejected["benchmark_overlap"] += 1
            continue
        if scrub(record.correction or "")[1] or any(
            scrub(m["content"])[1] for m in record.messages
        ):
            rejected["residual_pii"] += 1
            continue
        if corrections.setdefault(key, record.correction or "") != record.correction:
            conflicts.add(key)
            continue
        chosen.setdefault(key, record)
    for key in conflicts:
        chosen.pop(key, None)
    rejected["conflicting_corrections"] += len(conflicts)
    rows = [sft_row(record, version) for record in chosen.values()]
    for row in rows:
        row["language"] = (languages or {}).get(row["provenance"]["record_id"], "und")
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = {
        "version": version,
        "parent": parent,
        "created_at": datetime.now(UTC).isoformat(),
        "rows": len(rows),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "rejected": dict(rejected),
        "error_classes": dict(Counter(row["category"] for row in rows)),
        "training_budget": training_budget or {},
        "source_policy": "explicit consent, PII scrubbed, verified by a non-model verifier",
    }
    (out_dir / f"corrections.{version}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return manifest


def approval_decision(
    baseline: dict[str, Any], candidate: dict[str, Any], retention_ratio: float
) -> dict[str, Any]:
    reasons: list[str] = []
    if baseline["benchmark_sha256"] != candidate["benchmark_sha256"]:
        reasons.append("benchmark_hash_mismatch")
    if candidate["manual_pass_rate"] is None or baseline["manual_pass_rate"] is None:
        reasons.append("missing_manual_evaluation")
    else:
        if candidate["manual_pass_rate"] <= baseline["manual_pass_rate"]:
            reasons.append("no_improvement")
        for language, rate in baseline["manual_pass_by_language"].items():
            if candidate["manual_pass_by_language"].get(language, 0.0) < rate - 1 / 33:
                reasons.append(f"language_regression_{language}")
    if retention_ratio > RETENTION_LIMIT:
        reasons.append("retention_regression")
    return {"approved": not reasons, "reasons": reasons}


def record_approval(ledger: Path, entry: dict[str, Any]) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
