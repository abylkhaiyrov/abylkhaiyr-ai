import random
import re

from llm.core.config import ModelConfig
from llm.data import lexicon as lx
from llm.data.dialoguegen import FRUITS, add_typos
from llm.data.identity import load_identity_facts
from llm.data.instructgen import build_instruction_dataset
from llm.data.sft_quality import equations_hold
from llm.tokenizer.bpe import Message


def facts():
    return load_identity_facts(
        ModelConfig(
            vocab_size=8192,
            hidden_size=384,
            num_hidden_layers=8,
            num_attention_heads=6,
            num_key_value_heads=2,
            intermediate_size=1024,
            max_position_embeddings=512,
        )
    )


def rows(category: str, count: int = 600) -> list[dict]:
    return [r for r in build_instruction_dataset(facts(), count, 11) if r["category"] == category]


def test_new_conversational_categories_exist_and_wrong_assistant_turns_are_not_supervised():
    for category in (
        "recall_question",
        "continue_sequence",
        "typo_fact",
        "negation_fruit",
        "get_acquainted",
    ):
        assert rows(category), category
    corrections = rows("fix_mistake")
    assert corrections
    for row in corrections:
        first, last = row["messages"][1], row["messages"][3]
        assert first["supervised"] is False and "supervised" not in last
        match = re.match(r"(\d+) (.) (\d+) = (-?\d+)\.", first["content"])
        assert match is not None
        a, symbol, b, wrong = int(match[1]), match[2], int(match[3]), int(match[4])
        truth = {"+": a + b, "-": a - b, "×": a * b}[symbol]
        assert wrong != truth
        assert f"{a} {symbol} {b} = {truth}" in last["content"]


def test_recall_and_sequence_answers_are_verifiable():
    for row in rows("recall_question"):
        asked = row["messages"][0]["content"]
        assert asked in row["messages"][3]["content"]
    for row in rows("continue_sequence"):
        shown = [int(x) for x in re.findall(r"\d+", row["messages"][0]["content"])]
        step = shown[1] - shown[0]
        first = [int(x) for x in re.findall(r"\d+", row["messages"][1]["content"])]
        second = [int(x) for x in re.findall(r"\d+", row["messages"][3]["content"])]
        assert first == [shown[-1] + step * (i + 1) for i in range(3)]
        assert second == [first[-1] + step * (i + 1) for i in range(3)]


def test_negation_answers_never_name_a_fruit_of_the_forbidden_colour():
    colours = {term.en: term for term in lx.COLORS}
    for row in rows("negation_fruit"):
        question, answer = row["messages"][0]["content"], row["messages"][1]["content"]
        language = row["language"]
        forbidden = next(c for c, t in colours.items() if t.in_language(language) in question)
        named = [
            colour
            for fruit, colour in FRUITS
            if answer.startswith(fruit.in_language(language).capitalize())
        ]
        assert named and forbidden not in named


def test_typos_change_the_question_but_keep_the_reference_answer():
    for row in rows("typo_fact"):
        fact_answers = {f.answer.in_language(row["language"]) for f in lx.FACTS}
        questions = {f.question.in_language(row["language"]) for f in lx.FACTS}
        assert row["messages"][1]["content"] in fact_answers
        assert row["messages"][0]["content"] not in questions
    assert add_typos("a", "en", random.Random(0)) == "a"
    assert equations_hold("2 + 2 = 4")


def test_supervised_flag_survives_tokenization_labels(tokenizer):
    from llm.data.binary import IGNORE_INDEX

    row = rows("fix_mistake")[0]
    messages = [
        Message(m["role"], m["content"], m.get("supervised", True)) for m in row["messages"]
    ]
    _, labels = tokenizer.encode_conversation_with_labels(messages, IGNORE_INDEX)
    supervised = [x for x in labels if x != IGNORE_INDEX]
    assert len(supervised) <= len(tokenizer.encode(row["messages"][3]["content"])) + 2
