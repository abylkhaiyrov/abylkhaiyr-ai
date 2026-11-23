from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from llm.core.checksums import file_sha256
from llm.core.config import ModelConfig
from llm.core.profiles import count_parameters
from llm.data.binary import read_jsonl
from llm.data.dedup import stable_hash
from llm.data.v4_sources import load_sources
from llm.evaluation.metrics import normalize_words

ANCHOR_LENGTH = 24
EMPTY_PARENTHESES = re.compile(r"\(\s*(?:[,;]\s*)*\)")
INTERWIKI_LINE = re.compile(r"(?m)^\s*(?:en|fr|uk|de|es|ru):[^/\s]+\s*$")
CUES = {
    "en": (
        r"\b(?:because|therefore|consequently)\b",
        r"\b(?:in contrast|compared with|whereas|unlike)\b",
    ),
    "ru": (
        r"\b(?:потому что|поэтому|вследствие)\b",
        r"\b(?:в отличие от|по сравнению с|тогда как)\b",
    ),
    "kk": (r"\b(?:өйткені|сондықтан|себебі)\b", r"\b(?:салыстырғанда|қарағанда|өзгеше)\b"),
}
CAUSAL_CUES = {language: re.compile(cues[0], re.IGNORECASE) for language, cues in CUES.items()}
COMPARISON_CUES = {language: re.compile(cues[1], re.IGNORECASE) for language, cues in CUES.items()}


@dataclass(frozen=True)
class Anchor:
    split: str
    id: str
    words: tuple[str, ...]


def document_id(row: dict) -> str:
    return f"{row['source']}/{row['source_id']}"


def holdout_anchors(root: Path, bank: Path) -> Iterable[Anchor]:
    for split in ("validation", "test"):
        for row in read_jsonl(root / f"{split}.jsonl"):
            words = normalize_words(row["text"])
            if len(words) < ANCHOR_LENGTH:
                continue
            end = len(words) - ANCHOR_LENGTH
            for index in sorted({0, end // 2, end}):
                yield Anchor(split, document_id(row), tuple(words[index : index + ANCHOR_LENGTH]))
    for row in read_jsonl(bank):
        words = normalize_words(row["prefix"])
        if len(words) >= 8:
            yield Anchor("diagnostic", row["id"], tuple(words))


def corpus_scan(root: Path, anchors: Iterable[Anchor]) -> dict:
    index: dict[tuple[str, ...], list[Anchor]] = defaultdict(list)
    beginnings = set()
    totals: Counter[str] = Counter()
    for anchor in anchors:
        index[anchor.words[:4]].append(anchor)
        beginnings.add(anchor.words[0])
        totals[anchor.split] += 1
    matched: dict[str, set[str]] = defaultdict(set)
    samples: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    n_language_document: Counter[str] = Counter()
    formatting: dict[str, Counter[str]] = defaultdict(Counter)
    for row in read_jsonl(root / "train.jsonl"):
        text = row["text"]
        language = row["language"]
        words = normalize_words(text)
        n_language_document[language] += 1
        counts = formatting[language]
        counts["empty_parentheses"] += bool(EMPTY_PARENTHESES.search(text))
        counts["interwiki_line"] += bool(INTERWIKI_LINE.search(text))
        counts["causal_connector"] += bool(CAUSAL_CUES[language].search(text))
        counts["comparison_connector"] += bool(COMPARISON_CUES[language].search(text))
        for offset, word in enumerate(words):
            if word not in beginnings:
                continue
            for anchor in index.get(tuple(words[offset : offset + 4]), ()):
                if tuple(words[offset : offset + len(anchor.words)]) == anchor.words:
                    matched[anchor.split].add(anchor.id)
        identifier = document_id(row)
        priority = stable_hash(f"stage17-audit/{identifier}")
        selected = samples[row["language"]]
        if len(selected) >= 8 and priority >= selected[-1][0]:
            continue
        selected.append((priority, row))
        selected.sort(key=lambda pair: pair[0])
        if len(selected) > 8:
            selected.pop()
    return {
        "holdout_anchors": dict(totals),
        "holdout_documents_with_matching_anchor": {
            split: sorted(matched[split]) for split in totals
        },
        "scanned_documents": dict(n_language_document),
        "formatting_and_lexical_cues": formatting,
        "sample": [row for entries in samples.values() for _, row in entries],
    }


@dataclass(frozen=True)
class Checkpoint:
    path: str
    parameters: int
    processed_tokens: int
    tokenizer: str | None


def inventory(root: Path) -> list[Checkpoint]:
    result = []
    for path in sorted(root.rglob("meta.json")):
        if "stage17" in path.parts or not path.with_name("model.safetensors").is_file():
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        if "model_config" not in meta or "tokens_seen" not in meta:
            continue
        result.append(
            Checkpoint(
                str(path.parent),
                count_parameters(ModelConfig.model_validate(meta["model_config"])),
                int(meta["tokens_seen"]),
                meta.get("tokenizer_version"),
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/stage17"))
    parser.add_argument("--corpus", type=Path, default=Path("data/v4_expanded"))
    parser.add_argument("--inputs", type=Path, default=Path("data/v4_inputs"))
    args = parser.parse_args()
    source_catalog = load_sources(Path("configs/data_v4_sources.json"))
    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    verified = {}
    for name in manifest["sources"]:
        source = source_catalog[name]
        digest = file_sha256(source.local(args.inputs))
        if digest != source.sha256 or digest != manifest["sources"][name]["sha256"]:
            parser.error(f"source verification failed for {name}")
        verified[name] = {"revision": source.revision, "license": source.license, "sha256": digest}
    result = corpus_scan(
        args.corpus, holdout_anchors(args.corpus, args.root / "diagnostic_v2.jsonl")
    )
    result["sources"] = verified
    result["historical_checkpoints"] = [asdict(value) for value in inventory(Path("checkpoints"))]
    (args.root / "corpus_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
