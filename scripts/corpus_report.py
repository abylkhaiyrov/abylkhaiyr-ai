import json
import sys
import time
from pathlib import Path

from llm.data.report import REPORT_FILE, corpus_report
from llm.tokenizer.bpe import BPETokenizer

root = Path(sys.argv[1])
started = time.monotonic()
report = corpus_report(root, BPETokenizer.load(sys.argv[2]))
(root / REPORT_FILE).write_text(json.dumps(report, indent=1), encoding="utf-8")
print(report["documents"], report["total_tokens"], f"{time.monotonic() - started:.0f}s")
