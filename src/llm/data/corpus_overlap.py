from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from urllib.parse import unquote, urlsplit

from llm.data.dedup import MinHashDeduplicator, stable_hash
from llm.data.v4_corpus import Document
from llm.data.v4_splits import normalized


def normalize_words(text: str) -> list[str]:
    return normalized(text).split()


WINDOW = 16
MIN_PARAGRAPH = 10
NEAR_PARAGRAPH = 20
MAX_ANCHOR_FREQUENCY = 12
WIKIDATA = re.compile(r"wikidata\.org/wiki/(Q\d+)", re.IGNORECASE)
INTERLANGUAGE = re.compile(r"(?m)^\s*(?:en|ru|kk|simple):([^\n]+)$")


def paragraphs(text: str) -> Iterator[str]:
    for line in text.splitlines():
        words = normalize_words(line)
        if len(words) >= MIN_PARAGRAPH:
            yield " ".join(words)


def family(document: Document) -> str:
    if document.source.startswith(("wiki_", "belebele_")):
        page = unquote(urlsplit(document.url).path.removeprefix("/wiki/")).replace("_", " ")
        return f"wikipedia/{' '.join(normalize_words(page))}"
    if document.source.startswith("oasst_"):
        return f"oasst1/{document.source_id}"
    if document.source.startswith("python_"):
        return "cpython"
    if document.source == "kazsandra":
        return f"kazsandra/{document.source_id}"
    return document.source


def family_keys(document: Document) -> set[str]:
    keys = {family(document)}
    if document.source.startswith(("wiki_", "belebele_")):
        keys.update(f"qid/{key.upper()}" for key in WIKIDATA.findall(document.text))
        keys.update(
            f"wikipedia/{normalized(name)}" for name in INTERLANGUAGE.findall(document.text)
        )
    return keys


def shingles(words: list[str]) -> frozenset[tuple[str, ...]]:
    return frozenset(tuple(words[i : i + 3]) for i in range(len(words) - 2))


class ParagraphIndex:
    def __init__(self, blocks: set[str]) -> None:
        self.blocks = [
            shingles(words)
            for block in sorted(blocks)
            if len(words := block.split()) >= NEAR_PARAGRAPH
        ]
        counts = Counter(key for block in self.blocks for key in block)
        self.anchors: dict[tuple[str, ...], set[int]] = defaultdict(set)
        for index, block in enumerate(self.blocks):
            rare = sorted(block, key=lambda key: (counts[key], key))[:12]
            for key in rare:
                if counts[key] <= MAX_ANCHOR_FREQUENCY:
                    self.anchors[key].add(index)

    def matches(self, text: str) -> bool:
        words = text.split()
        if len(words) < NEAR_PARAGRAPH:
            return False
        probes = shingles(words)
        candidates = {index for key in probes for index in self.anchors.get(key, ())}
        for index in candidates:
            block = self.blocks[index]
            common = len(block & probes)
            if common >= 0.70 * (len(block) + len(probes) - common):
                return True
        return False


class HoldoutIndex:
    def __init__(self, documents: Iterable[Document]) -> None:
        self.documents: set[int] = set()
        self.paragraphs: set[int] = set()
        self.families: set[str] = set()
        self.windows: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
        self.near = MinHashDeduplicator(48, 0.67, 3, 1818)
        blocks: set[str] = set()
        for row in documents:
            words = normalize_words(row.text)
            self.documents.add(stable_hash(" ".join(words)))
            self.families.update(family_keys(row))
            blocks.update(paragraphs(row.text))
            self.near.is_duplicate(" ".join(words))
            for offset in range(len(words) - WINDOW + 1):
                self.windows[tuple(words[offset : offset + 4])].add(
                    tuple(words[offset + 4 : offset + WINDOW])
                )
        self.paragraphs.update(map(stable_hash, blocks))
        self.near_paragraphs = ParagraphIndex(blocks)

    def matches(self, row: Document, *, aliases: Iterable[str] = ()) -> set[str]:
        words = normalize_words(row.text)
        normalized = " ".join(words)
        reasons = set()
        if stable_hash(normalized) in self.documents:
            reasons.add("document")
        if (family_keys(row) | set(aliases)) & self.families:
            reasons.add("family")
        blocks = list(paragraphs(row.text))
        if any(stable_hash(line) in self.paragraphs for line in blocks):
            reasons.add("paragraph")
        for offset in range(len(words) - WINDOW + 1):
            tails = self.windows.get(tuple(words[offset : offset + 4]))
            if tails and tuple(words[offset + 4 : offset + WINDOW]) in tails:
                reasons.add("window")
                break
        if self.near.matches(normalized):
            reasons.add("near_document")
        if any(self.near_paragraphs.matches(line) for line in blocks):
            reasons.add("near_paragraph")
        return reasons
