import pytest

from llm.reasoning.intent import classify_intent, classify_task

CASES = [
    ("Hello!", "dialogue"),
    ("Привет, как дела?", "dialogue"),
    ("Сәлем!", "dialogue"),
    ("Спасибо большое", "dialogue"),
    ("Who are you?", "dialogue"),
    ("Что ты умеешь делать?", "capability"),
    ("что умеешь делать?", "capability"),
    ("что ты умеешь?", "capability"),
    ("что можешь?", "capability"),
    ("what can you do?", "capability"),
    ("не істей аласың?", "capability"),
    ("Чем можешь помочь?", "capability"),
    ("Как дела?", "dialogue"),
    ("Кто ты?", "dialogue"),
    ("Что такое PostgreSQL?", "information"),
    ("Что умеет делать PostgreSQL?", "information"),
    ("Сен кімсің?", "dialogue"),
    ("What is the capital of France?", "information"),
    ("Что такое фотосинтез?", "information"),
    ("Explain how a magnet works", "information"),
    ("Бактерия деген не?", "information"),
    ("What is 23 + 19?", "math"),
    ("Сколько будет 15 умножить на 4? Вычисли 15 * 4", "math"),
    ("Реши уравнение: 3x + 6 = 21", "math"),
    ("A box has 8 apples. You add 7 more. How many apples are there?", "math"),
    ("Anna is taller than Ben. Ben is taller than Carl. Who is the shortest?", "logic"),
    ("Аня старше Бори, а Боря старше Васи. Кто самый младший?", "logic"),
    ("All cats are animals. Tom is a cat. Is Tom an animal? Answer yes or no.", "logic"),
    ("Write a Python function add(a, b) that returns the sum.", "code"),
    ("Напиши функцию на Python, которая возвращает квадрат числа", "code"),
    ("SELECT name FROM users -- explain this SQL query", "code"),
    ("Translate into Russian: book", "translation"),
    ("Переведи на английский слово: море", "translation"),
    ("«Рахмет» сөзін орыс тіліне аудар.", "translation"),
    ("Summarize the following text: The cat sleeps all day.", "analysis"),
    ("Сколько слов в этом предложении: Кошка спит на диване.", "analysis"),
    ("Текст: Завод выпускает 450 деталей. Выдели ключевые факты.", "analysis"),
]


@pytest.mark.parametrize(("text", "intent"), CASES)
def test_intent_classification(text, intent):
    assert classify_intent(text) == intent


def test_only_verifiable_intents_use_tools():
    assert classify_task("What is 23 + 19?") == "math"
    assert classify_task("Anna is taller than Ben. Who is the tallest?") == "logic"
    assert classify_task("Write a Python function add(a, b)") == "code"
    assert classify_task("Translate into Russian: book") == "general"
    assert classify_task("Hello!") == "general"
    assert classify_task("What is the capital of France?") == "general"


def test_capability_examples_do_not_contain_the_evaluated_phrasings():
    from llm.reasoning.intent_examples import EXAMPLES

    evaluated = {
        "что умеешь делать?",
        "что ты умеешь?",
        "что можешь?",
        "what can you do?",
        "не істей аласың?",
        "чем можешь помочь?",
    }
    assert not evaluated & {text.casefold() for text, _ in EXAMPLES}
