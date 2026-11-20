from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llm.core.config import load_experiment_config
from llm.data.binary import tokenize_sft_split
from llm.data.cleaning import clean_document, detect_language
from llm.data.sft_quality import PromptBlocklist, QualityRules, SFTQualityFilter, normalize
from llm.evaluation.comparison import load_engine
from llm.evaluation.engine import EvaluationConfig, Evaluator
from llm.evaluation.prompts import evaluation_prompts
from llm.tokenizer.bpe import BPETokenizer
from llm.training.checkpoint import CheckpointManager
from llm.training.session import exclusive_session, run_training


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file: str
    license: Literal["MIT", "Apache-2.0", "CC0-1.0", "CC-BY-4.0"]


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=4000)


class CycleState(BaseModel):
    status: Literal["running", "paused", "rejected", "promoted", "failed"]
    reference: str
    evaluation_data: str = "data/processed"
    steps: int = Field(default=120, ge=1, le=5000)
    decision: dict[str, object] = Field(default_factory=dict)


def save_state(path: Path, data: BaseModel) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(data.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def collect(inbox: Path, tokenizer: BPETokenizer) -> list[Example]:
    manifest_path = inbox / "sources.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or len(manifest) > 32:
        raise ValueError("source manifest must contain up to 32 entries")
    blocklist = PromptBlocklist(evaluation_prompts("data/benchmarks"))
    quality = SFTQualityFilter(QualityRules(), tokenizer)
    admitted: dict[str, Example] = {}
    conflicts: set[str] = set()
    for entry in manifest:
        source = Source.model_validate(entry)
        path = (inbox / source.file).resolve(strict=True)
        if not path.is_relative_to(inbox.resolve()) or path.stat().st_size > 32 * 2**20:
            raise ValueError("source path or size is outside the inbox budget")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = Example.model_validate_json(line)
                except ValueError:
                    continue
                row.prompt, row.answer = map(clean_document, (row.prompt, row.answer))
                if detect_language(row.prompt) not in {"ru", "kk", "en"}:
                    continue
                record = {"source": source.file, "group": "skills", "messages": messages(row)}
                if not blocklist.blocks(record) and quality.admit(record):
                    key = normalize(row.prompt)
                    existing = admitted.get(key)
                    if existing is not None and normalize(existing.answer) != normalize(row.answer):
                        conflicts.add(key)
                    admitted.setdefault(key, row)
    return [admitted[key] for key in sorted(admitted) if key not in conflicts]


def messages(row: Example) -> list[dict[str, str]]:
    return [{"role": "user", "content": row.prompt}, {"role": "assistant", "content": row.answer}]


def split_rows(rows: list[Example]) -> tuple[list[Example], list[Example]]:
    train: list[Example] = []
    validation: list[Example] = []
    for row in rows:
        key = hashlib.sha256(normalize(row.prompt).encode()).digest()
        (validation if key[0] < 32 else train).append(row)
    return train, validation


def evaluate(
    checkpoint: str, tokenizer: str, device: str, data_dir: str = "data/processed"
) -> dict[str, float]:
    engine = load_engine(checkpoint, tokenizer, device)
    evaluator = Evaluator(engine, EvaluationConfig(data_dir, max_windows=48))
    return {**evaluator.language_modeling(), **evaluator.conversation(), **evaluator.arithmetic()}


def decide(before: dict[str, float], after: dict[str, float]) -> list[str]:
    required = ("conv_pass_rate", "conv_repetition_ratio", "test_loss", "arithmetic_exact_match")
    languages = {
        key for scores in (before, after) for key in scores if key.startswith("conv_lang_")
    }
    if any(
        not math.isfinite(scores.get(key, float("nan")))
        for scores in (before, after)
        for key in (*required, *languages)
    ):
        return ["missing or non-finite quality metrics"]
    failures = []
    if after["conv_pass_rate"] <= before["conv_pass_rate"]:
        failures.append("conversation pass rate did not improve")
    if after["conv_pass_rate"] < 0.5:
        failures.append("conversation pass rate below 50%")
    if any(after[key] < before[key] for key in before if key.startswith("conv_lang_")):
        failures.append("language-specific conversation regression")
    if after["conv_repetition_ratio"] > before["conv_repetition_ratio"] + 0.02:
        failures.append("repetition regression")
    if after["test_loss"] > before["test_loss"] + math.log(1.10):
        failures.append("held-out corpus perplexity increased by over 10%")
    if after["arithmetic_exact_match"] < before["arithmetic_exact_match"]:
        failures.append("arithmetic regression")
    return failures


class ActiveCheckpoint(BaseModel):
    checkpoint: str
    previous: str


def active_checkpoint(work: Path) -> ActiveCheckpoint | None:
    pointer = work / "active.json"
    return ActiveCheckpoint.model_validate_json(pointer.read_bytes()) if pointer.is_file() else None


def rollback(work: Path) -> str:
    with exclusive_session(work / ".cycle.lock"):
        current = active_checkpoint(work)
        if current is None:
            raise ValueError("no promoted checkpoint to roll back")
        CheckpointManager.verify(Path(current.previous))
        save_state(
            work / "active.json",
            ActiveCheckpoint(
                checkpoint=current.previous,
                previous=current.previous,
            ),
        )
        return current.previous


def run_cycle(
    inbox: Path,
    work: Path,
    steps: int,
    device: str = "cuda",
    *,
    reference: str = "checkpoints/tiny_gpu_sft/best",
    evaluation_data: str = "data/processed",
) -> str:
    if not 1 <= steps <= 5000:
        raise ValueError("steps must be in 1..5000")
    with exclusive_session(work / ".cycle.lock"):
        config = load_experiment_config("configs/tiny_gpu_sft.yaml")
        tokenizer = BPETokenizer.load(config.tokenizer.path)
        rows = collect(inbox, tokenizer)
        train, validation = split_rows(rows)
        if len(train) < 32 or len(validation) < 4:
            return "insufficient_data"
        ledger = work / "consumed.txt"
        consumed = (
            set(ledger.read_text(encoding="utf-8").splitlines()) if ledger.exists() else set()
        )
        keys = {hashlib.sha256(normalize(row.prompt).encode()).hexdigest() for row in train}
        fingerprint = hashlib.sha256(
            json.dumps([row.model_dump() for row in rows], sort_keys=True).encode()
        ).hexdigest()[:20]
        folder = work / fingerprint
        folder.mkdir(parents=True, exist_ok=True)
        state_file = folder / "state.json"
        pointer = work / "active.json"
        state = CycleState(
            status="running",
            steps=steps,
            evaluation_data=evaluation_data,
            reference=(
                ActiveCheckpoint.model_validate_json(pointer.read_bytes()).checkpoint
                if pointer.exists()
                else reference
            ),
        )
        if state_file.exists():
            state = CycleState.model_validate_json(state_file.read_bytes())
            if state.status in {"promoted", "rejected"}:
                return state.status
        elif len(keys - consumed) < 32:
            return "no_new_data"
        steps = state.steps
        state.status = "running"
        save_state(state_file, state)
        try:
            data = folder / "data/sft"
            data.mkdir(parents=True, exist_ok=True)
            for split, examples in (("train", train), ("validation", validation)):
                prefix = data / split
                path = prefix.with_suffix(".jsonl")
                if not path.exists():
                    path.write_text(
                        "".join(json.dumps({"messages": messages(row)}) + "\n" for row in examples),
                        encoding="utf-8",
                    )
                tokenize_sft_split(path, prefix, tokenizer)
            config.train = config.train.model_copy(
                update={
                    "data_dir": str(data.parent),
                    "init_from": state.reference,
                    "checkpoint_dir": str(folder / "weights"),
                    "output_dir": str(folder / "training"),
                    "max_steps": steps,
                    "eval_interval": min(steps, 50),
                    "checkpoint_interval": min(steps, 25),
                    "device": device,
                    "precision": "auto",
                    "mixture": {},
                    "sft_replay_fraction": 0.0,
                }
            )
            config.train.optimizer.learning_rate = 0.00005
            config.train.optimizer.warmup_steps = min(10, steps // 4)
            manager = CheckpointManager(config.train.checkpoint_dir, keep=3)
            latest = manager.latest_valid()
            if latest is None or manager.verify(latest)["step"] < steps:
                result = run_training(config, resume=True, stop_file=work / "STOP")
                if result.interrupted:
                    state.status = "paused"
                    save_state(state_file, state)
                    return state.status
            candidate = str(Path(config.train.checkpoint_dir) / manager.best_label())
            with exclusive_session():
                scores = {
                    "before": evaluate(
                        state.reference, config.tokenizer.path, device, state.evaluation_data
                    ),
                    "after": evaluate(
                        candidate, config.tokenizer.path, device, state.evaluation_data
                    ),
                }
            failures = decide(scores["before"], scores["after"])
            state.status = "rejected" if failures else "promoted"
            state.decision = {
                "candidate": candidate,
                "metrics": scores,
                "failures": failures,
                "train_examples": len(train),
                "validation_examples": len(validation),
            }
            pending = ledger.with_suffix(".tmp")
            pending.write_text("\n".join(sorted(consumed | keys)) + "\n", encoding="utf-8")
            pending.replace(ledger)
            if not failures:
                published = Path("checkpoints") / f"continuous_{fingerprint}_sft" / "best"
                if not published.exists():
                    staging = folder / "publish"
                    if staging.exists():
                        shutil.rmtree(staging)
                    shutil.copytree(candidate, staging)
                    CheckpointManager.verify(staging)
                    published.parent.mkdir(parents=True, exist_ok=True)
                    staging.replace(published)
                CheckpointManager.verify(published)
                save_state(
                    pointer,
                    ActiveCheckpoint(
                        checkpoint=str(published),
                        previous=state.reference,
                    ),
                )
            save_state(state_file, state)
            return state.status
        except Exception as error:
            state.status = "failed"
            state.decision = {"error": str(error)}
            save_state(state_file, state)
            raise
