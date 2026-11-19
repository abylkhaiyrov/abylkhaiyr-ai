from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import load_dataset
from torch import Tensor

from llm.core.errors import DataError
from llm.data.binary import ConversationLoader, read_jsonl, token_dtype, tokenize_sft_split
from llm.tokenizer.bpe import BPETokenizer

PREFERENCE_DIR = "preference"
TURN_PATTERN = re.compile(r"\n\n(Human|Assistant): ")
ROLE_MAP = {"Human": "user", "Assistant": "assistant"}
HH_DATASET = "Anthropic/hh-rlhf"
HH_LICENSE = "MIT"


def parse_hh_dialogue(text: str) -> list[dict[str, str]]:
    parts = TURN_PATTERN.split("\n\n" + text.strip("\n"))
    messages = []
    for role, content in zip(parts[1::2], parts[2::2], strict=False):
        if content.strip():
            messages.append({"role": ROLE_MAP[role], "content": content.strip()})
    return messages


def build_preference_files(
    output_dir: str | Path, max_pairs: int, validation_pairs: int
) -> dict[str, int]:
    root = Path(output_dir) / PREFERENCE_DIR
    root.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(HH_DATASET, split="train", streaming=True)
    rows: list[dict] = []
    for row in dataset:
        chosen, rejected = parse_hh_dialogue(row["chosen"]), parse_hh_dialogue(row["rejected"])
        if (
            chosen
            and rejected
            and chosen[-1]["role"] == "assistant"
            and rejected[-1]["role"] == "assistant"
            and chosen[:-1] == rejected[:-1]
            and chosen[-1] != rejected[-1]
        ):
            rows.append({"chosen": chosen, "rejected": rejected})
        if len(rows) >= max_pairs + validation_pairs:
            break
    splits = {"train": rows[validation_pairs:], "validation": rows[:validation_pairs]}
    for split, items in splits.items():
        with (root / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (root / "license.json").write_text(
        json.dumps({"dataset": HH_DATASET, "license": HH_LICENSE}), encoding="utf-8"
    )
    return {split: len(items) for split, items in splits.items()}


def tokenize_preference_split(root: Path, split: str, tokenizer: BPETokenizer) -> int:
    source = root / PREFERENCE_DIR / f"{split}.jsonl"
    if not source.is_file():
        raise DataError(f"preference data not found: {source}")
    for side in ("chosen", "rejected"):
        flat = root / PREFERENCE_DIR / f"{split}.{side}.jsonl"
        with flat.open("w", encoding="utf-8") as handle:
            for row in read_jsonl(source):
                handle.write(json.dumps({"messages": row[side]}, ensure_ascii=False) + "\n")
        count = tokenize_sft_split(flat, root / PREFERENCE_DIR / f"{split}.{side}", tokenizer)
    return count


class PreferenceLoader:
    def __init__(
        self,
        root: str | Path,
        split: str,
        tokenizer: BPETokenizer,
        max_length: int,
        batch_size: int,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        base = Path(root) / PREFERENCE_DIR
        dtype = token_dtype(tokenizer.vocab_size)
        self.chosen = ConversationLoader(
            base / f"{split}.chosen", dtype, max_length, batch_size, tokenizer.pad_id, seed, shuffle
        )
        self.rejected = ConversationLoader(
            base / f"{split}.rejected",
            dtype,
            max_length,
            batch_size,
            tokenizer.pad_id,
            seed,
            shuffle,
        )

    def set_batch_size(self, batch_size: int) -> None:
        self.chosen.set_batch_size(batch_size)
        self.rejected.set_batch_size(batch_size)

    def next_batch(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (*self.chosen.next_batch(), *self.rejected.next_batch())

    def state_dict(self) -> dict[str, int]:
        return self.chosen.state_dict()

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.chosen.load_state_dict(state)
        self.rejected.load_state_dict(state)
