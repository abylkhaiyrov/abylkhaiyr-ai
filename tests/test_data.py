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


def test_source_reads_explicit_data_files(monkeypatch):
    from llm.data import sources

    calls = {}

    def fake_load(dataset, config, **kwargs):
        calls.update(dataset=dataset, config=config, **kwargs)
        return [{"text": "sample training document"}]

    monkeypatch.setattr(sources, "load_dataset", fake_load)
    source = DataSourceConfig(
        name="direct",
        dataset="parquet",
        data_files="sample.parquet",
        text_field="text",
        license="MIT",
    )
    records = list(sources.stream_source(source, "cache"))
    assert records[0].text == "sample training document"
    assert calls["data_files"] == "sample.parquet"
    assert calls["streaming"] is True


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
    monkeypatch.setattr(pipeline, "stream_source", lambda source, raw, licenses=None: iter(records))
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


def test_math_generator_is_verified_balanced_and_deterministic():
    from llm.data.mathgen import GENERATORS, build_dataset, verify_problem

    first = build_dataset(300, seed=5)
    assert [p.key for p in first] == [p.key for p in build_dataset(300, seed=5)]
    assert all(verify_problem(p) for p in first)
    assert len({p.category for p in first}) >= len(GENERATORS) - 1
    assert len({p.key for p in first}) == len(first)


def test_math_verifier_rejects_wrong_solution():
    from dataclasses import replace

    from llm.data.mathgen import build_dataset, verify_problem

    problem = build_dataset(1, seed=3)[0]
    wrong = replace(problem, solution=problem.solution.rsplit("####", 1)[0] + "#### 99999")
    assert verify_problem(problem) and not verify_problem(wrong)


def test_benchmark_and_training_math_never_overlap():
    from llm.data.mathgen import build_dataset

    benchmark = build_dataset(100, seed=777)
    train = build_dataset(3000, seed=1, exclude={p.key for p in benchmark})
    assert not {p.key for p in benchmark} & {p.key for p in train}


def test_curriculum_orders_easy_before_hard():
    import numpy as np

    from llm.data.mathgen import build_dataset, curriculum_order

    ordered = curriculum_order(build_dataset(1500, seed=2), seed=2)
    third = len(ordered) // 3
    early = np.mean([p.level for p in ordered[:third]])
    late = np.mean([p.level for p in ordered[-third:]])
    assert late > early + 1.0


def test_decontamination_blocks_benchmark_ngrams():
    from llm.data.decontamination import Decontaminator

    guard = Decontaminator(
        ["Natalia sold clips to 48 of her friends in April and then half as many in May"], 8
    )
    assert guard.is_contaminated(
        "text before Natalia sold clips to 48 of her friends in April and after"
    )
    assert not guard.is_contaminated(
        "A completely different sentence about the weather in Almaty today"
    )


def test_mixture_loader_weights_resume_and_groups(tmp_path):
    from llm.data.binary import MixtureLoader

    loaders = {}
    for name, offset in (("a", 0), ("b", 1000)):
        (np.arange(900, dtype=np.uint16) + offset).tofile(tmp_path / f"{name}.bin")
        loaders[name] = PackedTokenLoader(
            tmp_path / f"{name}.bin", np.dtype(np.uint16), 16, 2, seed=0
        )
    mixture = MixtureLoader(loaders, {"a": 3.0, "b": 1.0}, seed=0)
    picks = [int(mixture.next_batch()[0][0, 0]) >= 1000 for _ in range(400)]
    assert 0.15 < sum(picks) / 400 < 0.35
    state = mixture.state_dict()
    expected = [mixture.next_batch()[0].tolist() for _ in range(5)]
    mixture.load_state_dict(state)
    assert [mixture.next_batch()[0].tolist() for _ in range(5)] == expected


def test_conversation_batches_pad_to_fixed_multiples_without_changing_targets(tmp_path, tokenizer):
    import numpy as np

    from llm.data.binary import IGNORE_INDEX, ConversationLoader, tokenize_sft_split
    from llm.tokenizer.bpe import Message

    rows = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]},
        {
            "messages": [
                {"role": "user", "content": "hello " * 6},
                {"role": "assistant", "content": "ok"},
            ]
        },
    ]
    source = tmp_path / "rows.jsonl"
    source.write_text("\n".join(__import__("json").dumps(r) for r in rows), encoding="utf-8")
    prefix = tmp_path / "split"
    tokenize_sft_split(source, prefix, tokenizer)
    loader = ConversationLoader(prefix, np.dtype(np.uint16), 128, 2, tokenizer.pad_id, 0, False)
    inputs, targets = loader.next_batch()
    assert (inputs.shape[1] + 1) % 64 == 0 or inputs.shape[1] + 1 == 128
    ids, labels = tokenizer.encode_conversation_with_labels(
        [Message("user", "hi"), Message("assistant", "yo")], IGNORE_INDEX
    )
    supervised = [int(t) for t in targets[0] if t != IGNORE_INDEX]
    assert supervised == [t for t in labels[1:] if t != IGNORE_INDEX]
    assert (inputs[0, len(ids) :] == tokenizer.pad_id).all()


def _write_conversations(tmp_path, tokenizer, pairs):
    import json

    from llm.data.binary import tokenize_sft_split

    source = tmp_path / "packed.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
        for q, a in pairs
    ]
    source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    prefix = tmp_path / "packed"
    tokenize_sft_split(source, prefix, tokenizer)
    return prefix


def test_packed_loader_keeps_every_conversation_whole_and_resumes_exactly(tmp_path, tokenizer):
    import numpy as np
    import torch

    from llm.data.binary import PackedConversationLoader

    pairs = [(f"question {i}", f"answer number {i} " * (1 + i % 3)) for i in range(20)]
    prefix = _write_conversations(tmp_path, tokenizer, pairs)
    dtype = np.dtype(np.uint16)

    def make():
        return PackedConversationLoader(prefix, dtype, 64, 2, tokenizer.pad_id, 0, True)

    loader = make()
    seen = 0
    for _ in range(3):
        inputs, _, segments = loader.next_batch()
        for row in range(inputs.shape[0]):
            for number in segments[row].unique().tolist():
                if number == 0:
                    continue
                positions = (segments[row] == number).nonzero().flatten()
                assert positions.tolist() == list(range(int(positions[0]), int(positions[-1]) + 1))
                assert inputs[row, positions[0]] == tokenizer.bos_id
                seen += 1
    assert seen >= 6
    state = loader.state_dict()
    expected = [loader.next_batch() for _ in range(3)]
    resumed = make()
    resumed.load_state_dict(state)
    for left, right in zip(expected, [resumed.next_batch() for _ in range(3)], strict=True):
        for a, b in zip(left, right, strict=True):
            assert torch.equal(a, b)


def test_length_bucketing_groups_similar_lengths_and_keeps_every_conversation(tmp_path, tokenizer):
    import numpy as np

    from llm.data.binary import ConversationLoader

    pairs = [(f"q{i}", "word " * (1 + (i * 7) % 30)) for i in range(64)]
    prefix = _write_conversations(tmp_path, tokenizer, pairs)
    dtype = np.dtype(np.uint16)
    plain = ConversationLoader(prefix, dtype, 256, 4, tokenizer.pad_id, 0, True)
    bucketed = ConversationLoader(
        prefix, dtype, 256, 4, tokenizer.pad_id, 0, True, pad_multiple=1, bucket_window=8
    )
    assert sorted(bucketed.order.tolist()) == list(range(64))

    def spread(loader):
        order = loader.order.reshape(-1, 4)
        return float(
            np.mean([loader.lengths[row].max() - loader.lengths[row].min() for row in order])
        )

    assert spread(bucketed) < spread(plain) * 0.5
    assert plain.pad_multiple == 64


def test_mixture_curriculum_ramps_probabilities_and_resumes(tmp_path, tokenizer):
    import numpy as np

    from llm.data.binary import ConversationLoader, MixtureLoader

    prefix = _write_conversations(tmp_path, tokenizer, [(f"q{i}", f"a{i}") for i in range(10)])
    dtype = np.dtype(np.uint16)

    def make():
        loaders = {
            name: ConversationLoader(prefix, dtype, 64, 2, tokenizer.pad_id, 0, True)
            for name in ("easy", "hard")
        }
        return MixtureLoader(
            loaders, {"easy": 0.5, "hard": 0.5}, 0, {"easy": 0.9, "hard": 0.1}, ramp_batches=100
        )

    loader = make()
    assert np.allclose(loader.probabilities, [0.9, 0.1])
    for _ in range(50):
        loader.next_batch()
    assert np.allclose(loader.probabilities, [0.7, 0.3])
    for _ in range(80):
        loader.next_batch()
    assert np.allclose(loader.probabilities, [0.5, 0.5])
    state = loader.state_dict()
    follow = [loader.next_batch()[0].tolist() for _ in range(3)]
    resumed = make()
    resumed.load_state_dict(state)
    assert [resumed.next_batch()[0].tolist() for _ in range(3)] == follow
