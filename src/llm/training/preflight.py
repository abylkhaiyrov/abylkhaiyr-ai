from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from llm.core.checksums import file_sha256
from llm.core.config import StrictModel
from llm.core.errors import ResourceError
from llm.training.checkpoint import CheckpointManager


class CorpusCheck(StrictModel):
    manifest: str
    root: str
    arms: list[str] = Field(min_length=1)
    overlap_audit: str | None = None


def frozen_hashes(inputs: dict[str, str]) -> dict[str, str]:
    return {label: file_sha256(Path(path)) for label, path in inputs.items()}


def verify_lock(hashes: dict[str, str], lock_file: str, labels: list[str]) -> None:
    locked: dict[str, str] = json.loads(Path(lock_file).read_text(encoding="utf-8"))
    changed = [label for label in labels if locked.get(label) != hashes[label]]
    if changed:
        raise ResourceError(f"frozen input changed: {', '.join(changed)}")


def verify_overlap_audit(path: str) -> None:
    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    if audit["a"] or audit["b"] or any(audit["short_original_anchors"].values()):
        raise ResourceError("evaluation overlap detected")


def verify_corpus(check: CorpusCheck, initial_checkpoint: str | None, tokens_required: int) -> None:
    root = Path(check.root)
    corpus = json.loads(Path(check.manifest).read_text(encoding="utf-8"))
    if initial_checkpoint is not None:
        initial = CheckpointManager.verify(Path(initial_checkpoint))
        if corpus["tokenizer"] != initial["tokenizer_version"]:
            raise ResourceError("initial checkpoint and corpus tokenizers differ")
    if check.overlap_audit is not None:
        verify_overlap_audit(check.overlap_audit)
    for arm in check.arms:
        details = corpus[arm]
        if details["tokens"]["train"]["stored_tokens"]["all"] <= tokens_required:
            raise ResourceError(f"arm {arm} would require repeated exposures")
        for name, digest in details["sha256"].items():
            if file_sha256(root / arm / name) != digest:
                raise ResourceError(f"arm {arm} contains changed data: {name}")
    first, *others = check.arms
    for name, digest in corpus[first]["sha256"].items():
        if name.startswith(("train", "tokens/train")):
            continue
        if any(corpus[other]["sha256"].get(name) != digest for other in others):
            raise ResourceError(f"arm holdouts differ: {name}")
