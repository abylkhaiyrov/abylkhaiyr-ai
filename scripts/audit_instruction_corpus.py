from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.data.audit import audit_corpus, write_audit
from llm.tokenizer.bpe import BPETokenizer


def main() -> int:
    parser = argparse.ArgumentParser(prog="audit_instruction_corpus")
    parser.add_argument("--data-dir", default="data/v3_1")
    parser.add_argument("--tokenizer", default="data/tokenizer")
    args = parser.parse_args()
    report = audit_corpus(args.data_dir, BPETokenizer.load(args.tokenizer).eos_id)
    write_audit(report, Path(args.data_dir) / "audit.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
