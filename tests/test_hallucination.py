import json

from llm.evaluation.hallucination import Probe, admits, build_probes, score_probes

DOC = (
    "The old library stands near the river in the center of the town. It was built in 1895 "
    "by local craftsmen and still keeps thousands of rare books. Visitors often come to read "
    "quietly in the large hall. The garden behind it is open every day."
)


def test_probe_set_is_deterministic_balanced_and_covers_three_languages(tmp_path):
    corpus = tmp_path / "test.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps({"language": "en", "text": DOC.replace("library", f"library {i}", 1)})
            for i in range(80)
        ),
        encoding="utf-8",
    )
    first = build_probes(corpus, 3, per_kind=2)
    assert first == build_probes(corpus, 3, per_kind=2)
    assert {p.language for p in first} == {"en", "ru", "kk"}
    kinds = {p.kind for p in first}
    assert {"fake_entity", "personal_or_realtime"} <= kinds
    assert any(p.kind == "context_answerable" and p.answer == "1895" for p in first)
    assert len(build_probes(None, 3, per_kind=2)) < len(first)


def test_scoring_separates_hallucination_over_refusal_and_correct_answers():
    probes = [
        Probe("fake_entity", "en", "Who is the president of Zorvania?", True),
        Probe("personal_or_realtime", "ru", "Который сейчас час?", True),
        Probe("context_answerable", "en", "text. in which year?", False, "1895"),
    ]
    liar = score_probes(probes, lambda p: "It is 1895 and the answer is Peter.")
    assert liar["hallucination_rate"] == 1.0 and liar["answerable_accuracy"] == 1.0
    refuser = score_probes(probes, lambda p: "I do not know.")
    assert refuser["hallucination_rate"] == 0.0
    assert refuser["over_refusal_rate"] == 1.0 and refuser["answerable_accuracy"] == 0.0
    assert admits("Не знаю.", "ru") and admits("Which one do you mean?", "en")
    assert not admits("Paris.", "en")
