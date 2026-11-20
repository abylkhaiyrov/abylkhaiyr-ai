from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path

from llm.data.binary import tokenize_sft_split
from llm.tokenizer.bpe import BPETokenizer

PATTERNS = {
    "ru": (
        (
            "В коробке {a} карандашей.",
            "Положил ещё {b}.",
            "Сколько карандашей всего? Только число.",
        ),
        (
            "Для поездки выбрал поезд {a}.",
            "Поменял билет на поезд {b}.",
            "Назови номер нового поезда.",
        ),
        (
            "Цена была {a} тенге.",
            "Цена стала {b} тенге.",
            "Сколько стоит сейчас? Напиши только число.",
        ),
        ("Выведи {b}.", "Не добавляй число {a}.", "Что нужно вывести? Только число."),
    ),
    "en": (
        (
            "The box has {a} pencils.",
            "I added {b} more.",
            "How many pencils now? Give only the number.",
        ),
        ("I booked train {a}.", "I changed to train {b}.", "What is the new train number?"),
        (
            "The price used to be {a} tenge.",
            "It now costs {b} tenge.",
            "Give the current price as a number.",
        ),
        ("Output {b}.", "Leave out the number {a}.", "What should you output? Just the number."),
    ),
    "kk": (
        (
            "Қорапта {a} қарындаш бар.",
            "Тағы {b} қарындаш салдым.",
            "Енді барлығы нешеу? Тек санды жаз.",
        ),
        (
            "{a}-пойызға билет алдым.",
            "Билетті {b}-пойызға ауыстырдым.",
            "Жаңа пойыздың нөмірі қандай?",
        ),
        (
            "Бағасы {a} теңге еді.",
            "Қазір {b} теңге тұрады.",
            "Қазіргі бағасы қанша? Тек санды жаз.",
        ),
        ("{b} санын шығар.", "{a} санын қоспа.", "Қай санды шығару керек? Тек санды жаз."),
    ),
}
ACK = {"ru": "Понятно.", "en": "Understood.", "kk": "Түсіндім."}


def examples(split: str) -> Iterator[dict[str, object]]:
    domain = range(3, 25) if split == "train" else range(25, 37)
    pairs = [(a, b) for a in domain for b in range(2, 20) if a != b]
    random.Random(14701).shuffle(pairs)
    selected = pairs[:150] if split == "train" else pairs[:30]
    for language, patterns in PATTERNS.items():
        for index, (a, b) in enumerate(selected):
            kind = index % len(patterns)
            initial, update, question = (text.format(a=a, b=b) for text in patterns[kind])
            texts = (
                initial,
                ACK[language],
                update,
                ACK[language],
                question,
                str(a + b if kind == 0 else b),
            )
            yield {
                "source": "grounded_arithmetic_and_context",
                "language": language,
                "messages": [
                    {"role": "assistant" if index % 2 else "user", "content": content}
                    for index, content in enumerate(texts)
                ],
            }


def main() -> None:
    groups = Path("data/dialogue_v4/sft_groups")
    tokenizer = BPETokenizer.load("data/tokenizer")
    for split in ("train", "validation"):
        prefix = groups / f"{split}.grounded_disjoint"
        source = prefix.with_name(prefix.name + ".jsonl")
        source.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples(split)),
            encoding="utf-8",
        )
        print(split, tokenize_sft_split(source, prefix, tokenizer))


if __name__ == "__main__":
    main()
