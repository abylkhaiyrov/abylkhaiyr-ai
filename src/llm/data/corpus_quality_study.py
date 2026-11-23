from __future__ import annotations

import heapq
import re
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from llm.data.binary import read_jsonl
from llm.data.corpus_overlap import HoldoutIndex, family_keys, paragraphs
from llm.data.dedup import MinHashDeduplicator, stable_hash
from llm.data.v4_corpus import Document
from llm.data.v4_dialogues import oasst_dialogues
from llm.data.v4_quality import LANGUAGES, CyrillicLanguageFilter, clean_v4_text
from llm.data.v4_sources import Source
from llm.tokenizer.bpe import BPETokenizer

BUDGET_PER_LANGUAGE = 1_750_000
EMPTY_MARKUP = re.compile(r"\s*\([\s,;:]*\)")
BROKEN_WIKI_LINE = re.compile(
    r"(?m)^[ \t]*(?:\{\||\|[}-]|\{\{|(?:area|population|площадь|население|ауданы)\s*[—–:]\s*(?:км²|km²|чел\.|адам))[ \t]*\n",
    re.IGNORECASE,
)
INTERWIKI = re.compile(r"(?m)^\s*[a-z]{2,3}:\S+(?:\s+\S+){0,5}\s*$")
BOILERPLATE = re.compile(
    r"(?im)^\s*(external links|references|see also|notes|примечания|литература|"
    r"ссылки|см\. также|дереккөздер|сілтемелер|тағы қараңыз)\s*\n"
)
TOPICS = {
    "programming": r"programming|software|algorithm|database|программировани|алгоритм|"
    r"бағдарламала|деректер қор",
    "mathematical_explanations": r"mathematic|equation|geometry|арифметик|уравнен|"
    r"теорем|геометри|теңдеу|математик",
    "logic": r"formal logic|logical inference|deduction|дедукци|формальная логика|логическ[а-я]+ вывод|"
    r"логикалық|ой қорыту",
    "technical_documentation": r"protocol specification|technical standard|"
    r"технические характеристик|стандарт связи|техникалық сипаттама|хаттама",
    "practical": r"first aid|home repair|food safety|gardening|первая помощь|"
    r"пищев[а-я]+ безопасност|садовод|алғашқы көмек|көгалдандыру|бақша",
    "popular_science": r"physics|biolog|astronom|ecolog|chemical|физик|биолог|"
    r"астроном|экологи|химия|физикалық|биологиялық|экологиялық",
    "causal": r"because|therefore|consequently|потому что|вследствие|поэтому|"
    r"өйткені|сондықтан|себебі",
    "comparison": r"whereas|compared with|in contrast|в отличие от|по сравнению с|"
    r"тогда как|салыстырғанда|айырмашылығы|қарағанда",
    "education": r"lesson|classroom|teaching method|педагогик|учебн[а-я]+ пособие|"
    r"обучени|оқыту|білім беру|оқулық",
}
PATTERNS = {
    category: re.compile(rf"\b(?:{pattern})", re.IGNORECASE) for category, pattern in TOPICS.items()
}


@dataclass(frozen=True)
class Encoded:
    row: Document
    tokens: int


def read_documents(path: Path) -> Iterator[Document]:
    for row in read_jsonl(path):
        yield Document(
            row["text"],
            row["source"],
            row["source_id"],
            row["url"],
            row["language"],
            row["category"],
        )


def passage_holdouts(sources: dict[str, Source], root: Path) -> dict[str, list[Document]]:
    splits: dict[str, list[Document]] = {"validation": [], "test": []}
    for language in LANGUAGES:
        source = sources[f"belebele_{language}"]
        seen: set[tuple[str, str]] = set()
        for row in read_jsonl(source.local(root)):
            passage, url = row["flores_passage"], row["link"]
            key = (url, passage)
            if key in seen:
                continue
            seen.add(key)
            split = "test" if stable_hash(url) % 4 == 0 else "validation"
            splits[split].append(
                Document(
                    passage, source.name, f"{stable_hash(passage):016x}", url, language, "prose"
                )
            )
    return splits


def enhanced(document: Document) -> Document | None:
    if not document.source.startswith("wiki_"):
        return document
    text = BOILERPLATE.split(document.text, maxsplit=1)[0]
    cleaned = clean_v4_text(text, 160)
    if cleaned is None:
        return None
    category = "encyclopedia"
    head = cleaned[:650]
    for label, pattern in PATTERNS.items():
        if len(pattern.findall(head)) >= (2 if label in {"causal", "comparison"} else 1):
            category = label
            break
    return replace(document, text=cleaned, category=category)


def human_dialogues(
    source: Source, root: Path, quality: CyrillicLanguageFilter, tokenizer: BPETokenizer
) -> Iterator[Document]:
    categories = {
        "causal": "causal",
        "comparison": "comparison",
        "code": "programming",
        "math": "mathematical_explanations",
    }
    for dialogue in oasst_dialogues(source.name, source.local(root), quality, tokenizer):
        text = "\n\n".join(message.content for message in dialogue.messages)
        yield Document(
            text,
            source.name,
            dialogue.family.removeprefix("oasst1/"),
            f"{source.license_url}#{dialogue.source_id}",
            dialogue.language,
            categories.get(dialogue.category, "conversation"),
        )


def clean_candidates(
    rows: Iterable[Document],
    holdouts: HoldoutIndex,
    rejected: Counter[str],
    *,
    curated: bool,
) -> Iterator[Document]:
    near = MinHashDeduplicator(48, 0.78, 4, 1818)
    unique: set[int] = set()
    seen_paragraphs: set[int] = set()
    for row in rows:
        original = row
        if row.category != "code":
            text = row.text
            if row.source.startswith("wiki_"):
                text = EMPTY_MARKUP.sub("", INTERWIKI.sub("", BROKEN_WIKI_LINE.sub("", text)))
                if text != row.text:
                    rejected["technical_repair"] += 1
            cleaned = clean_v4_text(text, 48)
            if cleaned is None:
                rejected["damaged"] += 1
                continue
            row = replace(row, text=cleaned)
        if curated:
            result = enhanced(row)
            if result is None:
                rejected["boilerplate_only"] += 1
                continue
            row = result
        matches = holdouts.matches(row, aliases=family_keys(original))
        if matches:
            rejected["held_out"] += 1
            rejected.update(f"holdout/{reason}" for reason in matches)
            continue
        normalized = " ".join(row.text.casefold().split())
        identifier = stable_hash(normalized)
        blocks = {stable_hash(line) for line in paragraphs(row.text)}
        if identifier in unique or near.is_duplicate(normalized):
            rejected["duplicate_document"] += 1
            continue
        if blocks and len(blocks & seen_paragraphs) * 5 >= len(blocks) * 4:
            rejected["duplicate_paragraphs"] += 1
            continue
        unique.add(identifier)
        seen_paragraphs.update(blocks)
        yield row


def encoded_counts(rows: list[Document], tokenizer: BPETokenizer) -> list[Encoded]:
    return [
        Encoded(row, len(tokens) + 2)
        for row, tokens in zip(rows, tokenizer.encode_stream(row.text for row in rows), strict=True)
    ]


def balanced(rows: Iterable[Encoded], target: int = BUDGET_PER_LANGUAGE) -> list[Encoded]:
    buckets: dict[tuple[str, str], deque[Encoded]] = defaultdict(deque)
    for row in sorted(
        rows, key=lambda item: stable_hash(f"{item.row.source}/{item.row.source_id}")
    ):
        buckets[row.row.language, row.row.category].append(row)
    result = []
    for language in LANGUAGES:
        queue = [(0, category) for lang, category in buckets if lang == language]
        heapq.heapify(queue)
        selected_tokens = 0
        while queue and selected_tokens < target:
            used, category = heapq.heappop(queue)
            bucket = buckets[language, category]
            item = bucket.popleft()
            result.append(item)
            selected_tokens += item.tokens
            if bucket:
                heapq.heappush(queue, (used + item.tokens, category))
        if selected_tokens < target:
            raise ValueError(f"{language} has {selected_tokens} curated tokens, need {target}")
    return result
