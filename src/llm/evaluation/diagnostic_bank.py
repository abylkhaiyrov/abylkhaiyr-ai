from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

LANGUAGES = ("en", "ru", "kk")
TASKS = (
    "grammar",
    "reading",
    "knowledge",
    "commonsense",
    "semantic",
    "negation",
    "causal",
    "instruction",
    "context",
    "logic",
    "math",
    "code",
)
ITEMS_PER_CELL = 32
BANK_SEED = 717
ELEMENTS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
)
NAMES = ("Neral", "Dovin", "Melar", "Zarek", "Paven", "Vesan", "Tidor", "Sulen")
OBJECTS = (
    ("map", "bell", "brush", "key", "book", "rope", "cup", "lamp"),
    ("карту", "звонок", "щётку", "ключ", "книгу", "верёвку", "чашку", "лампу"),
    ("картаны", "қоңырауды", "щетканы", "кілтті", "кітапты", "арқанды", "кесені", "шамды"),
)


@dataclass(frozen=True)
class Item:
    id: str
    language: str
    task: str
    group: str
    family: str
    source: str
    prefix: str
    options: tuple[str, ...]
    answer: int
    separator: str = " "


@dataclass(frozen=True)
class Locale:
    grammar: str
    reading: str
    knowledge: str
    commonsense: str
    semantic: str
    negation: str
    causal: str
    instruction: str
    context: str
    logic: str
    math: str
    code: str


LOCALES = (
    Locale(
        "Choose the grammatical wording. Yesterday at {hour}:15, {name}",
        "In a supply log, {name} packed the {obj} in box {a}. Box {b} was empty. Which box held the item? Answer:",
        "What is the atomic number of the element with the symbol {symbol}? Answer:",
        "{name} needs to leave in {a} minutes. Walking takes {b}, the bus takes {c}, and cycling takes {d}. Which travel time meets the deadline? Answer:",
        "The sign says 'Open every day except day {a}.' Which day is excluded? Answer:",
        "{name} checked every box but did not open box {a}. Which one remained unopened? Answer:",
        "A machine stops when the timer reaches zero. It starts with {a} minutes remaining. How many minutes pass before it stops? Answer:",
        "Return the {order} number only. List: {a}, {b}, {c}, {d}. Answer:",
        "The meeting was first assigned room {a}. Later {name} wrote that it was moved to room {b}. Which room is current? Answer:",
        "{name} found four scores: {a}, {b}, {c}, {d}. Only scores in the interval {low} through {high} advance. Which score advances? Answer:",
        "Calculate {expression}. Answer:",
        "What value does the Python expression {expression} produce? Answer:",
    ),
    Locale(
        "Выбери грамматически правильную форму. Вчера в {hour}:15 {name}",
        "В журнале указано, что {name} положил {obj} в ящик {a}. Ящик {b} был пустым. Где лежал предмет? Ответ:",
        "Каков атомный номер элемента с обозначением {symbol}? Ответ:",
        "{name} должен быть на месте через {a} минут. Пешком идти {b}, на автобусе ехать {c}, на велосипеде — {d}. Какое время в пути позволяет успеть? Ответ:",
        "На табличке написано: «Открыто ежедневно, кроме дня {a}». Какой день исключён? Ответ:",
        "{name} проверил все ящики, но не открыл ящик {a}. Какой ящик остался закрытым? Ответ:",
        "Машина выключится, когда таймер дойдёт до нуля. В начале на нём {a} минут. Через сколько минут машина выключится? Ответ:",
        "Назови только {order} число из списка: {a}, {b}, {c}, {d}. Ответ:",
        "Встречу назначили в комнате {a}. Позже {name} написал, что её перенесли в комнату {b}. Куда теперь идти? Ответ:",
        "{name} выписал четыре оценки: {a}, {b}, {c}, {d}. Проходят только оценки от {low} до {high} включительно. Какая оценка проходит? Ответ:",
        "Вычисли {expression}. Ответ:",
        "Какое значение вернёт выражение Python {expression}? Ответ:",
    ),
    Locale(
        "Грамматикалық тұрғыдан дұрыс нұсқаны таңда. Кеше сағат {hour}:15-те {name}",
        "{name} тізімде {obj} {a} нөмірлі жәшікке салғанын жазды. {b} нөмірлі жәшік бос еді. Зат қай жәшікте болды? Жауап:",
        "{symbol} таңбасымен белгіленетін элементтің атомдық нөмірі қандай? Жауап:",
        "{name} орнына {a} минуттан кешікпей жетуі керек. Жаяу жол {b}, автобуспен {c}, велосипедпен {d} минутқа созылады. Қай жол уақыты үлгеруге мүмкіндік береді? Жауап:",
        "Тақтайшада «{a}-күннен басқа күндері ашық» деп жазылған. Қай күн кірмейді? Жауап:",
        "{name} барлық жәшікті тексерді, бірақ {a} нөмірлі жәшікті ашпады. Қай жәшік жабық қалды? Жауап:",
        "Таймер нөлге жеткенде құрылғы тоқтайды. Басында {a} минут қалды. Құрылғы неше минуттан кейін тоқтайды? Жауап:",
        "Тізімнен тек {order} санды жаз: {a}, {b}, {c}, {d}. Жауап:",
        "Кездесу әуелі {a} нөмірлі бөлмеге қойылды. Кейін {name} кездесуді {b} нөмірлі бөлмеге ауыстырғанын жазды. Енді қай бөлмеге бару керек? Жауап:",
        "{name} төрт ұпайды жазды: {a}, {b}, {c}, {d}. Тек {low} мен {high} аралығындағы ұпайлар өтеді (шеттері кіреді). Қай ұпай өтеді? Жауап:",
        "{expression} мәнін есепте. Жауап:",
        "Python тіліндегі {expression} өрнегінің мәні қандай? Жауап:",
    ),
)

GRAMMAR = (
    (
        "walked home",
        "bought bread",
        "found the key",
        "closed the window",
        "washed the cup",
        "made tea",
        "opened the book",
        "rang the bell",
    ),
    (
        "пошёл домой",
        "купил хлеб",
        "нашёл ключ",
        "закрыл окно",
        "вымыл чашку",
        "заварил чай",
        "открыл книгу",
        "позвонил в звонок",
    ),
    (
        "үйге барды",
        "нан алды",
        "кілтті тапты",
        "терезені жапты",
        "кесені жуды",
        "шай демдеді",
        "кітапты ашты",
        "қоңырауды соқты",
    ),
)
GRAMMAR_ERRORS = (
    (
        "walk home",
        "buy bread",
        "find the key",
        "close the window",
        "wash the cup",
        "make tea",
        "open the book",
        "ring the bell",
    ),
    (
        "пойти домой",
        "купить хлеб",
        "найти ключ",
        "закрыть окно",
        "вымыть чашку",
        "заварить чай",
        "открыть книгу",
        "позвонить в звонок",
    ),
    (
        "үйге бар",
        "нан ал",
        "кілтті тап",
        "терезені жап",
        "кесені жу",
        "шай демде",
        "кітапты аш",
        "қоңырауды соқ",
    ),
)
ORDERS = (
    ("first", "second", "third", "fourth"),
    ("первое", "второе", "третье", "четвёртое"),
    ("бірінші", "екінші", "үшінші", "төртінші"),
)


def shuffled_item(
    language: str,
    task: str,
    index: int,
    prefix: str,
    correct: str,
    distractors: tuple[str, str, str],
    positions: list[int],
) -> Item:
    options = list(distractors)
    random.Random(f"{BANK_SEED}/{language}/{task}/{index}").shuffle(options)
    position = positions[index]
    options.insert(position, correct)
    return Item(
        f"diag17/{language}/{task}/{index:02d}",
        language,
        task,
        task,
        f"{task}/{index // 4:02d}",
        "nist_facts" if task == "knowledge" else "stage17_original",
        prefix,
        tuple(options),
        position,
    )


def reasoning_case(
    task: str,
    locale: Locale,
    locale_index: int,
    index: int,
) -> tuple[str, int, tuple[int, int, int, int]]:
    start = 21 + index * 16
    samples = random.Random(f"{BANK_SEED}/{task}/{index}").sample(range(start, start + 15, 3), 4)
    a, b, c, d = samples
    interval = samples[index % 4]
    fields = {
        "name": NAMES[index % len(NAMES)],
        "obj": OBJECTS[locale_index][index // 4],
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "low": interval - 1,
        "high": interval + 1,
        "symbol": ELEMENTS[index],
        "order": ORDERS[locale_index][index % 4],
    }
    if task == "commonsense":
        fields.update(a=15 + index, b=13 + index, c=17 + index, d=19 + index)
        answer = 13 + index
    elif task == "instruction":
        answer = (a, b, c, d)[index % 4]
    elif task == "context":
        answer = b
    elif task == "logic":
        answer = interval
    elif task == "knowledge":
        answer = index + 1
    else:
        answer = a
    if task == "math":
        first, second = index + 11, index % 8 + 2
        operation = index % 4
        fields["expression"] = (
            f"{first} + {second}",
            f"{first} - {second}",
            f"{first} × {second}",
            f"{first * second} ÷ {second}",
        )[operation]
        answer = (first + second, first - second, first * second, first)[operation]
    if task == "code":
        first, second = index + 11, index % 8 + 2
        operation = index % 4
        fields["expression"] = (
            f"[{first}, {second}][1] + 10",
            f"{first * second + 2} // {second + 2} + 10",
            f"len([{first}, {second}, {first}, {second}]) + {first}",
            f"{first} % {second} + 10",
        )[operation]
        answer = (
            second + 10,
            (first * second + 2) // (second + 2) + 10,
            4 + first,
            first % second + 10,
        )[operation]
    return getattr(locale, task).format_map(fields), answer, (a, b, c, d)


def items() -> Iterator[Item]:
    for locale_index, (language, locale) in enumerate(zip(LANGUAGES, LOCALES, strict=True)):
        for task in TASKS:
            positions = list(range(4)) * (ITEMS_PER_CELL // 4)
            random.Random(f"{BANK_SEED}/{language}/{task}").shuffle(positions)
            for index in range(ITEMS_PER_CELL):
                if task == "grammar":
                    word = index // 4
                    prefix = locale.grammar.format(
                        name=NAMES[index % len(NAMES)], hour=6 + index // 2
                    )
                    correct = GRAMMAR[locale_index][word]
                    wrong = GRAMMAR_ERRORS[locale_index][word]
                    distractors = (wrong, f"{wrong} {wrong}", f"{wrong} {NAMES[(index + 1) % 8]}")
                else:
                    prefix, answer, values = reasoning_case(task, locale, locale_index, index)
                    if task in {"reading", "instruction", "context", "logic"}:
                        others = [value for value in values if value != answer]
                    else:
                        rank = min(index % 4, answer - 1)
                        rng = random.Random(f"{BANK_SEED}/{task}/{index}")
                        lower = rng.sample(range(max(1, answer - 9), answer), rank)
                        upper = rng.sample(range(answer + 1, answer + 10), 3 - rank)
                        others = lower + upper
                    correct = str(answer)
                    distractors = (str(others[0]), str(others[1]), str(others[2]))
                yield shuffled_item(language, task, index, prefix, correct, distractors, positions)
