from __future__ import annotations

from dataclasses import replace

import pytest

from llm.data.corpus_overlap import HoldoutIndex
from llm.data.corpus_quality_study import Encoded, balanced, enhanced
from llm.data.dedup import MinHashDeduplicator
from llm.data.v4_corpus import Document


def document(text: str, key: str, *, source: str = "wiki_en", language: str = "en") -> Document:
    return Document(
        text, source, key, f"https://en.wikipedia.org/wiki/{key}", language, "encyclopedia"
    )


def test_holdout_finds_embedded_windows_and_partial_paraphrases() -> None:
    passage = (
        "A nearly empty streetside bakery opened its doors before the first tram "
        "arrived carrying cleaners and nurses from the far western district."
    )
    sealed = HoldoutIndex([document(passage, "bakery")])
    embedded = document(f"A separate introduction. {passage} Another ending.", "other")
    assert "window" in sealed.matches(embedded)
    edited = passage.replace("before", "shortly before").replace("district", "neighborhood")
    assert "near_paragraph" in sealed.matches(document(f"Local history\n{edited}", "rewritten"))
    assert not sealed.matches(document("The school issued a timetable for next week.", "school"))


def test_translated_family_is_blocked_without_matching_words() -> None:
    source = document("A small English article about cranberry cultivation.", "cranberries")
    translated = document(
        "Жидекті бақта да өсіруге болады.\nen:cranberries",
        "Мүкжидек",
        source="wiki_kk",
        language="kk",
    )
    assert "family" in HoldoutIndex([source]).matches(translated)


def test_lookup_does_not_insert_new_duplicates() -> None:
    index = MinHashDeduplicator(48, 0.8, 4, 1234)
    text = "A newspaper article describes a new bridge across the nearby river"
    assert not index.matches(text)
    assert not index.is_duplicate(text)
    assert index.matches(text)
    assert index.is_duplicate(text)


def test_balancing_uses_measured_tokens_without_repeating_documents() -> None:
    source = document("A natural sample of prose that passes the minimum.", "first")
    candidates = [
        Encoded(
            replace(
                source, source_id=f"{lang}/{category}/{index}", language=lang, category=category
            ),
            tokens,
        )
        for lang in ("en", "ru", "kk")
        for category, tokens in (("news", 9), ("manual", 4))
        for index in range(3)
    ]
    result = balanced(candidates, target=11)
    assert len({item.row.source_id for item in result}) == len(result)
    for language in ("en", "ru", "kk"):
        chosen = [item for item in result if item.row.language == language]
        assert {item.row.category for item in chosen} == {"manual", "news"}
        assert sum(item.tokens for item in chosen) == 13
    with pytest.raises(ValueError, match="curated tokens"):
        balanced(candidates, target=80)


def test_biology_lexeme_does_not_trigger_logic_category() -> None:
    source = document("Биологические исследования растений. " * 8, "biology", language="ru")
    result = enhanced(source)
    assert result is not None
    assert result.category == "popular_science"
