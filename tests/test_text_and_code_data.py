import json
import random
import re

from llm.data import codegen
from llm.data.codespecs import NEW_SPECS, typed
from llm.data.textgen import (
    BUILDERS,
    NUMBER,
    Passage,
    build_reading_dataset,
    keyword_candidates,
    passages_from_document,
    reading_row,
    usable_sentence,
    words_of,
)
from llm.evaluation import suites
from llm.evaluation.conversation import benchmark_prompts

DOCS = {
    "en": (
        "The old library stands near the river in the center of the town. It was built in 1895 "
        "by local craftsmen and still keeps thousands of rare books. Visitors often come to read "
        "quietly in the large hall. The garden behind it is open every day."
    ),
    "ru": (
        "Старая библиотека стоит у реки в центре нашего города. Она была построена в 1895 году "
        "местными мастерами и до сих пор хранит тысячи редких книг. Посетители часто приходят "
        "почитать в тишине большого зала. Сад за зданием открыт каждый день."
    ),
    "kk": (
        "Ескі кітапхана қаламыздың орталығында өзен жағасында орналасқан. Ол 1895 жылы жергілікті "
        "шеберлермен салынған және әлі күнге дейін мыңдаған сирек кітапты сақтайды. Келушілер "
        "үлкен залда тыныш отырып оқуды жиі ұнатады. Ғимарат артындағы бақ күн сайын ашық."
    ),
}


def passages(language: str) -> list[Passage]:
    return list(passages_from_document(DOCS[language], language, random.Random(0)))


def test_usable_sentence_rejects_markup_artifacts_and_short_text():
    assert usable_sentence("This is a perfectly normal sentence about a town.")
    assert not usable_sentence("Too short.")
    assert not usable_sentence("A sentence with  double spaces inside of it for sure.")
    assert not usable_sentence("A sentence with an empty () bracket pair inside it today.")
    assert not usable_sentence("See http://example.org for the long sentence about it.")


def test_reading_tasks_are_verifiable_against_the_passage():
    rng = random.Random(1)
    checked = set()
    for language in DOCS:
        for passage in passages(language):
            pool = keyword_candidates(passage) + ["неведомое", "bananas", "алма"]
            for build in BUILDERS:
                for _ in range(8):
                    built = build(rng, passage, rng.choice(["en", "ru", "kk"]), pool)
                    if built is None:
                        continue
                    question, answer, category = built
                    checked.add(category)
                    if category == "first_sentence":
                        assert answer == passage.sentences[0]
                    if category == "last_sentence":
                        assert answer == passage.sentences[-1]
                    if category == "count_sentences":
                        assert str(len(passage.sentences)) in answer
                    if category == "list_numbers":
                        assert answer == ", ".join(NUMBER.findall(passage.text))
                    if category == "which_sentence":
                        word = re.findall(r"[“«](.+?)[”»]", question)[0]
                        assert word.lower() in {w.lower() for w in words_of(answer)}
                    if category == "contains_word":
                        word = re.findall(r"[“«](.+?)[”»]", question)[0]
                        present = word.lower() in {w.lower() for w in words_of(passage.text)}
                        assert answer in (
                            ("Yes.", "Да.", "Иә.") if present else ("No.", "Нет.", "Жоқ.")
                        )
    assert {"first_sentence", "last_sentence", "count_sentences", "language_id"} <= checked


def test_reading_dataset_uses_equal_language_groups_and_no_duplicates(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    with corpus.open("w", encoding="utf-8") as handle:
        for index in range(40):
            for language, text in DOCS.items():
                handle.write(
                    json.dumps({"language": language, "text": text + f" Номер {index}."}) + "\n"
                )
    rows = build_reading_dataset(corpus, 6, 0)
    assert rows and len({str(r["messages"]) for r in rows}) == len(rows)
    assert {r["group"] for r in rows} <= {"reading_en", "reading_ru", "reading_kk"}
    assert all(r["source"] == "synthetic_reading" for r in rows)
    row = reading_row(random.Random(0), passages("en")[0], ["bananas"])
    assert row is None or row["messages"][1]["content"]


def test_new_code_specs_are_independent_of_the_benchmarks():
    names = {spec.name for spec in NEW_SPECS}
    held_out = {case.function_name for case in suites.CODE_CASES}
    assert not names & held_out
    assert not names & {spec.name for spec in codegen.SPECS}
    prompts = " ".join(benchmark_prompts()).lower()
    assert not any(re.search(rf"\b{name}\b", prompts) for name in names)
    assert len(names) == len(NEW_SPECS)


def test_new_python_solutions_pass_and_mutants_fail():
    for spec in NEW_SPECS:
        assert codegen.passes(spec, "python", spec.python), spec.name
        if spec.slow:
            assert codegen.passes(spec, "python", spec.slow), spec.name
        if spec.mutant:
            assert not codegen.passes(spec, "python", codegen.mutate(spec)), spec.name


def test_typescript_is_derived_with_matching_parameter_types():
    javascript = "function f(a, b) {\n  return a + b;\n}"
    assert typed(javascript, ("number", "number"), "number").startswith(
        "function f(a: number, b: number): number"
    )


def test_code_dataset_covers_many_task_kinds_without_sql_dominance():
    rows = codegen.build_code_dataset(1500, 5)
    sources = {r["source"] for r in rows}
    assert {"synthetic_code_write", "synthetic_code_bugfix", "synthetic_code_optimize"} <= sources
    sql = sum(r["language"] == "sql" for r in rows)
    assert sql / len(rows) < 0.1
    assert {r["language"] for r in rows} >= {"python", "javascript", "typescript", "java"}
