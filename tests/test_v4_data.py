from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from llm.data.v4_corpus import Document, summarize_token_streams, write_splits
from llm.data.v4_curated import book_body, cpython_code
from llm.data.v4_dialogues import Dialogue
from llm.data.v4_quality import CyrillicLanguageFilter, clean_v4_text
from llm.data.v4_sources import fetch_snapshot, load_sources
from llm.data.v4_splits import DialogueDeduplicator, select_training
from llm.tokenizer.bpe import Message


@pytest.fixture
def languages() -> CyrillicLanguageFilter:
    return CyrillicLanguageFilter(
        {
            "ru": ["Да, конечно. Сколько времени занимает дорога до города?"],
            "kk": ["Иә. Бұл кітап кім туралы жазылған? Бүгін күн жылы болды."],
            "ky": ["Ооба. Бул китеп ким жөнүндө жазылган? Бул кандай шаар?"],
        }
    )


def test_short_responses_and_kyrgyz_rejection(languages: CyrillicLanguageFilter) -> None:
    assert languages.accepts("Да.", "ru", allow_short=True)
    assert languages.accepts("Иә.", "kk", allow_short=True)
    for sample in ("Альбулалар деген эмне ?", "Ооба.", "Бул китеп ким жөнүндө жазылган?"):
        assert not languages.accepts(sample, "ru", allow_short=True)
        assert not languages.accepts(sample, "kk", allow_short=True)


def test_corrupted_text_and_source_markers_are_removed() -> None:
    assert clean_v4_text("broken \ufffd text", 4) is None
    assert clean_v4_text("plain text", 4) == "plain text"
    text = "Header\n*** START OF THE PROJECT GUTENBERG EBOOK ***\nBook.\n"
    text += "*** END OF THE PROJECT GUTENBERG EBOOK ***\nFooter"
    assert book_body(text) == "Book.\n"
    assert cpython_code("def func():\n    return 5\n") == "def func():\n    return 5"


def test_license_and_cached_bytes_are_checked(tmp_path: Path) -> None:
    catalog = tmp_path / "sources.json"
    catalog.write_text(
        json.dumps(
            [
                {
                    "name": "test",
                    "repository": "owner/source",
                    "revision": "1" * 40,
                    "path": "data/test.jsonl",
                    "sha256": "1" * 64,
                    "license": "unknown",
                    "category": "qa",
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="unapproved license"):
        load_sources(catalog)
    with pytest.raises(ValueError, match="unexpected source checksum"):
        fetch_snapshot("https://example.invalid/ignored", catalog, "1" * 64)


def dialog(key: str, prompt: str, family: str, other: str = "") -> Dialogue:
    turns = (Message("user", prompt),)
    if other:
        turns += (Message("assistant", "How can I help?"), Message("user", other))
    return Dialogue(
        "source", key, family, "en", "instruction", (*turns, Message("assistant", "Yes."))
    )


def test_dialogue_isolation_preserves_equal_short_responses() -> None:
    holdout = DialogueDeduplicator()
    assert holdout.add(dialog("first", "May I keep this leaflet?", "one"))
    rows = [
        dialog("same_family", "Do you have time tonight?", "one"),
        dialog("same_root", "May I keep this leaflet?", "two", "Please check."),
        dialog("new", "Is this seat available?", "three"),
        dialog("newer", "Could I call you tomorrow?", "four"),
    ]
    assert {row.source_id for row in select_training(rows, holdout)} == {"new", "newer"}


def test_corpus_dedup_prioritizes_evaluation(tmp_path: Path) -> None:
    document = "This document appears under more than one edition of the corpus."
    train = Document(document, "wiki", "source1", "url", "en", "education", "train")
    validation = Document(document, "wiki", "source2", "other", "en", "education", "validation")
    counts, duplicate = write_splits(iter([train, validation]), tmp_path)
    assert counts == {"validation.en.education": 1}
    assert duplicate == {"train": 1}


def test_validation_stream_interleaves_language_blocks(tmp_path: Path) -> None:
    dtype = np.dtype(np.uint16)
    for value, language in enumerate(("en", "ru", "kk")):
        np.full(1024, value, dtype=dtype).tofile(tmp_path / f"validation.{language}.bin")
    result = summarize_token_streams(
        tmp_path, "validation", {"en": 1024, "ru": 1024, "kk": 1024}, dtype
    )
    actual = np.fromfile(tmp_path / "validation.bin", dtype=dtype)
    assert actual[::512].tolist() == [0, 1, 2, 0, 1, 2]
    assert result["distinct_token_ids"] == 3
