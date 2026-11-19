import json

import numpy as np
import pytest

from llm.core.config import DataConfig, DataSourceConfig
from llm.core.errors import DataError
from llm.data.binary import (
    ConversationLoader,
    PackedTokenLoader,
    tokenize_corpus_split,
    tokenize_sft_split,
)
from llm.data.cleaning import (
    clean_document,
    detect_language,
    remove_personal_data,
    repeated_line_ratio,
    strip_html,
)
from llm.data.dedup import ExactDeduplicator, MinHashDeduplicator
from llm.data.pipeline import CorpusBuilder, QualityFilter, assign_split


def make_config(**overrides) -> DataConfig:
    values = {
        "sources": [DataSourceConfig(name="s", dataset="d", text_field="text", license="MIT")],
        "min_chars": 20,
    }
    return DataConfig(**{**values, **overrides})


def test_html_stripping_and_unescape():
    assert strip_html("<p>Hi &amp; <b>bye</b></p><script>x()</script>").split() == [
        "Hi",
        "&",
        "bye",
    ]


def test_personal_data_removed():
    text = "Mail me a@b.com or call +7 701 123 45 67, host 192.168.0.1, card 4111 1111 1111 1111"
    cleaned = remove_personal_data(text)
    for leaked in ("a@b.com", "701 123", "192.168", "4111"):
        assert leaked not in cleaned


def test_clean_document_normalizes_whitespace():
    assert clean_document("a  \r\n\r\n\r\n\r\n b\x00") == "a\n\nb"


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("The history of the city is long and it was founded by the people of the region", "en"),
        ("Это длинное предложение на русском языке, и оно про историю города", "ru"),
        ("Қазақстан Республикасы — Еуразия континентінде орналасқан ұлы мемлекет", "kk"),
        ("Це речення українською мовою їх", "other"),
        ("12345 67890", "other"),
    ],
)
def test_language_detection(text, language):
    assert detect_language(text) == language


def test_repetition_ratio():
    assert repeated_line_ratio("a\na\na\nb") == 0.75
    assert repeated_line_ratio("single") == 0.0


def test_exact_dedup():
    dedup = ExactDeduplicator()
    assert not dedup.is_duplicate("abc")
    assert dedup.is_duplicate("abc")


def test_near_duplicate_detection_separates_similar_from_different():
    dedup = MinHashDeduplicator(permutations=64, threshold=0.8, shingle_size=3, seed=1)
    base = " ".join(f"word{i}" for i in range(200))
    near = base + " extra tail"
    different = " ".join(f"other{i}" for i in range(200))
    assert not dedup.is_duplicate(base)
    assert dedup.is_duplicate(near)
    assert not dedup.is_duplicate(different)


def test_splits_are_deterministic_and_roughly_proportional():
    config = make_config(validation_fraction=0.1, test_fraction=0.1)
    labels = [assign_split(f"doc {i}", config) for i in range(5000)]
    assert labels == [assign_split(f"doc {i}", config) for i in range(5000)]
    assert 350 < labels.count("test") < 650 and 350 < labels.count("validation") < 650


def test_quality_filter_reasons():
    quality = QualityFilter(make_config())
    assert quality.rejection_reason("short", "text") == "too_short"
    assert quality.rejection_reason("@#$%^&*()" * 10, "text") == "symbols"
    assert quality.rejection_reason("same line here\n" * 10, "text") == "repetition"
    assert (
        quality.rejection_reason("Это достаточно длинный русский текст для проверки", "text")
        is None
    )
    assert quality.rejection_reason("lorem ipsum dolor sit amet consectetur", "text") == "language"


def test_no_near_duplicates_leak_across_splits(tmp_path, monkeypatch):
    from llm.data import pipeline
    from llm.data.sources import Record

    docs = []
    for topic in range(60):
        body = " ".join(f"тема{topic}слово{i}" for i in range(80))
        docs.append(body)
        docs.append(body + " дополнение")
    records = [Record("s", "text", "ru", text) for text in docs]
    monkeypatch.setattr(pipeline, "stream_source", lambda source, raw: iter(records))
    config = make_config(
        output_dir=str(tmp_path),
        validation_fraction=0.3,
        test_fraction=0.3,
        detect_language=False,
        min_chars=10,
        near_duplicate_threshold=0.8,
    )
    manifest = CorpusBuilder(config).build()
    assert manifest["sources"]["s"]["rejected"]["near_duplicate"] == 60
    seen = {}
    for split in ("train", "validation", "test"):
        for line in (tmp_path / "corpus" / f"{split}.jsonl").read_text().splitlines():
            topic = json.loads(line)["text"].split()[0]
            assert seen.setdefault(topic, split) == split


def test_packed_loader_shapes_targets_and_resume(tmp_path):
    tokens = np.arange(1000, dtype=np.uint16)
    path = tmp_path / "t.bin"
    tokens.tofile(path)
    loader = PackedTokenLoader(path, np.dtype(np.uint16), 16, 4, seed=0)
    x, y = loader.next_batch()
    assert x.shape == y.shape == (4, 16)
    assert (y[:, :-1] == x[:, 1:]).all() and (y[:, -1] == x[:, -1] + 1).all()
    state = loader.state_dict()
    expected = [loader.next_batch()[0] for _ in range(3)]
    loader.load_state_dict(state)
    assert all((loader.next_batch()[0] == e).all() for e in expected)


def test_packed_loader_covers_epoch_without_repeats(tmp_path):
    np.arange(1000, dtype=np.uint16).tofile(tmp_path / "t.bin")
    loader = PackedTokenLoader(tmp_path / "t.bin", np.dtype(np.uint16), 10, 1, seed=3)
    starts = [int(loader.next_batch()[0][0, 0]) for _ in range(loader.num_chunks)]
    assert sorted(starts) == [i * 10 for i in range(loader.num_chunks)]


def test_packed_loader_rejects_tiny_files(tmp_path):
    np.arange(5, dtype=np.uint16).tofile(tmp_path / "t.bin")
    with pytest.raises(DataError):
        PackedTokenLoader(tmp_path / "t.bin", np.dtype(np.uint16), 16, 1, seed=0)


def test_tokenize_corpus_and_sft_end_to_end(tmp_path, tokenizer):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("\n".join(json.dumps({"text": t}) for t in ["hello world", "привет мир"]))
    count = tokenize_corpus_split(corpus, tmp_path / "c.bin", tokenizer)
    stored = np.fromfile(tmp_path / "c.bin", dtype=np.uint16)
    assert count == len(stored)
    assert stored[0] == tokenizer.bos_id and tokenizer.eos_id in stored
    sft = tmp_path / "s.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "answer"}]}
    ]
    sft.write_text("\n".join(json.dumps(r) for r in rows * 3))
    assert tokenize_sft_split(sft, tmp_path / "s", tokenizer) == 3
    loader = ConversationLoader(tmp_path / "s", np.dtype(np.uint16), 64, 2, tokenizer.pad_id, 0)
    inputs, targets = loader.next_batch()
    assert inputs.shape == targets.shape
    trained = targets[0][targets[0] != -100].tolist()
    assert trained == [*tokenizer.encode("answer"), tokenizer.eos_id]


def test_dotted_prefixes_do_not_collide(tmp_path, tokenizer):
    for side, answer in (("chosen", "good"), ("rejected", "bad")):
        flat = tmp_path / f"train.{side}.jsonl"
        row = {
            "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": answer}]
        }
        flat.write_text(json.dumps(row))
        tokenize_sft_split(flat, tmp_path / f"train.{side}", tokenizer)
    dtype = np.dtype(np.uint16)
    chosen = ConversationLoader(tmp_path / "train.chosen", dtype, 64, 1, tokenizer.pad_id, 0, False)
    rejected = ConversationLoader(
        tmp_path / "train.rejected", dtype, 64, 1, tokenizer.pad_id, 0, False
    )
    assert chosen.next_batch()[1].tolist() != rejected.next_batch()[1].tolist()
