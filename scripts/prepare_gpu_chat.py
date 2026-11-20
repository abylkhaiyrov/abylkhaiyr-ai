"""Build instruction data for the local GPU model without altering pretraining data."""

from __future__ import annotations

import json
from pathlib import Path

from llm.core.config import load_experiment_config
from llm.data.binary import read_jsonl
from llm.data.instruction_corpus import InstructionBuildConfig, build_instruction_corpus
from llm.tokenizer.bpe import BPETokenizer


def main() -> None:
    source = Path("data/gpu_chat_source")
    groups = source / "sft_groups"
    groups.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        with (groups / f"{split}.chat.jsonl").open("w", encoding="utf-8") as handle:
            for row in read_jsonl(Path("data/processed/sft") / f"{split}.jsonl"):
                handle.write(json.dumps({**row, "group": "chat"}, ensure_ascii=False) + "\n")
    build = InstructionBuildConfig(
        source_dir=str(source),
        output_dir="data/gpu_chat",
        corpus_file="data/processed/corpus/train.jsonl",
        synthetic_count=18000,
        identity_count=600,
        code_count=2000,
        reading_per_language=1500,
        seed=1234,
    )
    model = load_experiment_config("configs/tiny_gpu.yaml").model
    manifest = build_instruction_corpus(build, BPETokenizer.load("data/tokenizer"), model)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
