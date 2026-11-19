import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from llm.core.config import PERMISSIVE_LICENSES
from llm.data.binary import read_jsonl, token_dtype
from llm.data.dedup import stable_hash

SPLITS = ("train", "validation", "test")
root = Path(sys.argv[1])
vocab_size = int(sys.argv[2])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
dtype = token_dtype(vocab_size)
report: dict[str, object] = {}

report["token_files"] = {
    split: {
        path.name: int(path.stat().st_size // dtype.itemsize)
        for path in sorted((root / "tokens").glob(f"{split}*.bin"))
    }
    for split in SPLITS
}
report["manifest_tokens"] = {s: manifest["tokenized"][s]["tokens"] for s in SPLITS}
report["licenses"] = dict(Counter(info["license"] for info in manifest["sources"].values()))
report["benchmark_overlap_rejected"] = {
    n: i["rejected"]["benchmark_overlap"]
    for n, i in manifest["sources"].items()
    if "benchmark_overlap" in i["rejected"]
}

corpus_hashes: dict[str, set[int]] = {}
corpus_counts: dict[str, Counter[str]] = {}
for split in SPLITS:
    rows = list(read_jsonl(root / "corpus" / f"{split}.jsonl"))
    corpus_hashes[split] = {stable_hash(r["text"]) for r in rows}
    corpus_counts[split] = Counter(r["group"] for r in rows)
report["corpus_documents_by_group"] = {s: dict(c) for s, c in corpus_counts.items()}
report["corpus_overlap"] = {
    f"{a}&{b}": len(corpus_hashes[a] & corpus_hashes[b])
    for i, a in enumerate(SPLITS)
    for b in SPLITS[i + 1 :]
}

sft_hashes: dict[str, set[int]] = {}
sft_counts: dict[str, Counter[str]] = {}
sft_prompts: dict[str, set[int]] = {}
sft_tokens: dict[str, int] = {}
for split in SPLITS:
    rows = list(read_jsonl(root / "sft" / f"{split}.jsonl"))
    sft_hashes[split] = {stable_hash(json.dumps(r["messages"], sort_keys=True)) for r in rows}
    sft_prompts[split] = {stable_hash(r["messages"][0]["content"]) for r in rows}
    sft_counts[split] = Counter(r["source"] for r in rows)
    ids = np.fromfile(root / "sft" / f"{split}.ids.bin", dtype=dtype)
    sft_tokens[split] = int(ids.size)
report["sft_tokens"] = sft_tokens
report["sft_examples_by_source"] = {s: dict(c) for s, c in sft_counts.items()}
report["sft_overlap_full"] = {
    f"{a}&{b}": len(sft_hashes[a] & sft_hashes[b])
    for i, a in enumerate(SPLITS)
    for b in SPLITS[i + 1 :]
}
report["sft_overlap_prompt"] = {
    f"{a}&{b}": len(sft_prompts[a] & sft_prompts[b])
    for i, a in enumerate(SPLITS)
    for b in SPLITS[i + 1 :]
}
report["permissive_allowlist"] = list(PERMISSIVE_LICENSES)
out = root / "audit.json"
out.write_text(json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps(report, indent=1))
