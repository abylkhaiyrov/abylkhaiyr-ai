from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from llm.data.dedup import stable_hash
from llm.data.v4_corpus import Document, partition_of, summarize_token_streams, write_splits
from llm.data.v4_curated import book_body, cpython_code
from llm.data.v4_dialogues import Dialogue, article_lead
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


def test_article_lead_cleans_template_residue_and_rejects_gaps() -> None:
    lead = article_lead(
        "Lake Foo ( ; ) is a freshwater lake , located in the north of the country and "
        "fed by three small rivers that flow down from the surrounding hills.\nSecond paragraph."
    )
    assert lead is not None
    assert "( " not in lead and " ," not in lead and "Second" not in lead
    assert (
        article_lead("Area is  km² and population is  people in total across the whole region.")
        is None
    )
    assert article_lead("Short.") is None


def test_partition_is_deterministic_and_shared_with_documents() -> None:
    document = Document("text", "wiki_kk", "42", "url", "kk", "encyclopedia")
    assert document.split == partition_of("wiki_kk", "42")
    shares = Counter(partition_of("wiki_kk", str(index)) for index in range(2000))
    assert 0.9 < shares["train"] / 2000 < 0.97


def test_blocked_prompts_never_reach_training() -> None:
    from llm.data.v4_splits import DialogueDeduplicator, normalized, select_training
    from llm.tokenizer.bpe import Message

    def row(identifier: str, question: str) -> Dialogue:
        return Dialogue(
            "s",
            identifier,
            f"f/{identifier}",
            "en",
            "definition",
            (Message("user", question), Message("assistant", "A long enough answer.")),
        )

    index = DialogueDeduplicator()
    index.block({stable_hash(normalized("Which one is better?"))})
    kept = list(
        select_training([row("1", "Which one is better?"), row("2", "What is a lake?")], index)
    )
    assert [item.source_id for item in kept] == ["2"]


def test_curated_dialogues_keep_provenance_and_form_their_own_sft_group(
    tmp_path, tokenizer
) -> None:
    from llm.data.v4_dialogues import curated_dialogues
    from llm.data.v4_splits import sft_group

    path = tmp_path / "curated.jsonl"
    row = {
        "id": "curated/ru/smalltalk/0",
        "language": "ru",
        "category": "smalltalk",
        "messages": [
            {"role": "user", "content": "Привет!"},
            {"role": "assistant", "content": "Здравствуйте!"},
        ],
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    [dialogue] = list(curated_dialogues(path, tokenizer))
    assert dialogue.key == "curated/curated/ru/smalltalk/0"
    assert dialogue.family == "curated/curated/ru/smalltalk/0"
    assert sft_group(dialogue) == "curated_ru"
