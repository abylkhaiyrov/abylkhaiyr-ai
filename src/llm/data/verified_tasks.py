from __future__ import annotations

from llm.data.verified_common import (
    LANGUAGES,
    Row,
    capitalize,
    exchange,
    load_config,
    make_row,
    rng_for,
)

SOURCE = "verified_programmatic"
HELD_OUT_ARITHMETIC = {("+", 17, 25), ("*", 8, 9), ("+", 5, 3)}
HELD_OUT_PROFESSIONS = {"врач", "pilot", "дәрігер"}
REPEAT_COUNTS = {
    "ru": {2: "два раза", 3: "три раза", 4: "четыре раза", 5: "пять раз"},
    "en": {2: "two times", 3: "three times", 4: "four times", 5: "five times"},
    "kk": {2: "екі рет", 3: "үш рет", 4: "төрт рет", 5: "бес рет"},
}
REPEAT_PROMPTS = {
    "ru": ["Напиши слово «{w}» {n} через запятую.", "Повтори слово «{w}» {n}, разделяя запятыми."],
    "en": [
        'Write the word "{w}" {n} separated by commas.',
        'Repeat the word "{w}" {n}, separated by commas.',
    ],
    "kk": ["«{w}» сөзін {n} үтір арқылы жаз.", "«{w}» сөзін {n} үтірмен бөліп қайтала."],
}
UPPER_PROMPTS = {
    "ru": ["Напиши слово «{w}» заглавными буквами.", "Запиши «{w}» большими буквами."],
    "en": ['Write the word "{w}" in capital letters.', 'Write "{w}" in uppercase.'],
    "kk": ["«{w}» сөзін бас әріптермен жаз.", "«{w}» сөзін үлкен әріптермен жазып бер."],
}
WORD_INDEX = {"ru": 0, "en": 1, "kk": 2}
OPERATIONS = {
    "+": {
        "ru": ["Сколько будет {a} + {b}?", "Посчитай: {a} + {b}."],
        "en": ["What is {a} + {b}?", "Calculate {a} + {b}."],
        "kk": ["{a} + {b} неше болады?", "Есепте: {a} + {b}."],
    },
    "-": {
        "ru": ["Сколько будет {a} - {b}?", "Вычисли {a} - {b}."],
        "en": ["What is {a} - {b}?", "Calculate {a} - {b}."],
        "kk": ["{a} - {b} неше болады?", "Есепте: {a} - {b}."],
    },
    "*": {
        "ru": ["Чему равно {a} умножить на {b}?", "Сколько будет {a} × {b}?"],
        "en": ["What is {a} times {b}?", "Calculate {a} × {b}."],
        "kk": ["{a} санын {b} санына көбейтсе неше болады?", "Есепте: {a} × {b}."],
    },
}
SYMBOLS = {"+": "+", "-": "-", "*": "×"}
BASKET = {
    "ru": (
        "В корзине было {a} {noun}. Из неё убрали {b}. Сколько {noun} осталось?",
        "{a} - {b} = {c}. Осталось {c} {noun}.",
    ),
    "en": (
        "A basket had {a} {noun}. {b} of them were taken out. How many {noun} are left?",
        "{a} - {b} = {c}. {c} {noun} are left.",
    ),
    "kk": (
        "Себетте {a} {noun} болды. Одан {b} алынды. Қанша {noun} қалды?",
        "{a} - {b} = {c}. {c} {noun} қалды.",
    ),
}
NOUNS = {
    "ru": ["яблок", "конфет", "книг", "карандашей", "шаров"],
    "en": ["apples", "candies", "books", "pencils", "balloons"],
    "kk": ["алма", "кәмпит", "кітап", "қарындаш", "шар"],
}
FOLLOW_UP = {
    "ru": {
        "+": ("А если прибавить к этому ещё {c}?", "{s} + {c} = {t}."),
        "-": ("А если из этого вычесть {c}?", "{s} - {c} = {t}."),
    },
    "en": {
        "+": ("And if we add {c} more to that?", "{s} + {c} = {t}."),
        "-": ("And if we subtract {c} from that?", "{s} - {c} = {t}."),
    },
    "kk": {
        "+": ("Енді оған тағы {c} қоссақ ше?", "{s} + {c} = {t}."),
        "-": ("Ал осыдан {c} алып тастасақ ше?", "{s} - {c} = {t}."),
    },
}
FIRST_QUESTION = {
    "ru": "Сколько будет {a} + {b}?",
    "en": "What is {a} + {b}?",
    "kk": "{a} + {b} неше болады?",
}
RU_INSTRUMENTAL = {
    "учитель": "учителем",
    "повар": "поваром",
    "водитель": "водителем",
    "строитель": "строителем",
    "художник": "художником",
    "инженер": "инженером",
    "музыкант": "музыкантом",
    "продавец": "продавцом",
}
EN_ARTICLE = {"engineer": "an engineer"}
PLACES = {
    "ru": [
        ("Библиотека", "библиотека"),
        ("Магазин", "магазин"),
        ("Музей", "музей"),
        ("Аптека", "аптека"),
        ("Кафе", "кафе"),
    ],
    "en": [
        ("library", "library"),
        ("shop", "shop"),
        ("museum", "museum"),
        ("pharmacy", "pharmacy"),
        ("cafe", "cafe"),
    ],
    "kk": [
        ("Кітапхана", "кітапхана"),
        ("Дүкен", "дүкен"),
        ("Мұражай", "мұражай"),
        ("Дәріхана", "дәріхана"),
        ("Кафе", "кафе"),
    ],
}
KK_TIME_SUFFIX = {
    1: "де",
    2: "де",
    3: "те",
    4: "те",
    5: "те",
    6: "да",
    7: "де",
    8: "де",
    9: "да",
    10: "да",
    11: "де",
}
CITIES = {
    "ru": ["Алматы", "Караганде", "Семее", "Актобе", "Павлодаре", "Таразе", "Костанае", "Атырау"],
    "en": ["Almaty", "Karaganda", "Semey", "Aktobe", "Pavlodar", "Taraz", "Kostanay", "Atyrau"],
    "kk": [
        "Алматыда",
        "Қарағандыда",
        "Семейде",
        "Ақтөбеде",
        "Павлодарда",
        "Таразда",
        "Қостанайда",
        "Атырауда",
    ],
}
CITY_NOMINATIVE = {"ru": CITIES["ru"], "en": CITIES["en"], "kk": CITIES["kk"]}
MISSING_ATTRIBUTE = {
    "ru": [
        ("Сколько лет {p}?", "В тексте не сказано, сколько лет {p}."),
        ("Какой цвет любит {p}?", "В тексте не сказано, какой цвет любит {p}."),
        ("Есть ли у {p} семья?", "В тексте не сказано, есть ли у {p} семья."),
    ],
    "en": [
        ("How old is {p}?", "The text does not say how old {p} is."),
        ("What color does {p} like?", "The text does not say what color {p} likes."),
        ("Does {p} have a family?", "The text does not say whether {p} has a family."),
    ],
    "kk": [
        ("{p} неше жаста?", "Мәтінде {p} неше жаста екені айтылмаған."),
        ("{p} қандай түсті жақсы көреді?", "Мәтінде {p} қандай түсті жақсы көретіні айтылмаған."),
        ("{p} отбасы бар ма?", "Мәтінде {p} отбасының бар-жоғы айтылмаған."),
    ],
}
PROFESSION_TEXT = {
    "ru": (
        "Текст: {p1} живёт {city_loc}. {p1} работает {j1}. {p2} работает {j2}. Вопрос: Кем работает {q}?",
        "{q} работает {ja}.",
    ),
    "en": (
        "Text: {p1} lives in {city_loc}. {p1} works as {j1}. {p2} works as {j2}. Question: What is {q}'s job?",
        "{q} works as {ja}.",
    ),
    "kk": (
        "Мәтін: {p1} {city_loc} тұрады. {p1} {j1} болып жұмыс істейді. {p2} {j2} болып жұмыс істейді. Сұрақ: {q} кім болып жұмыс істейді?",
        "{q} {ja} болып жұмыс істейді.",
    ),
}
PLACE_TEXT = {
    "ru": (
        "Текст: {place} открывается в {a} утра и закрывается в {b} вечера. Вопрос: Во сколько {noun} {verb}?",
        "{place} {verb} в {x} {part}.",
    ),
    "en": (
        "Text: The {place} opens at {a} in the morning and closes at {b} in the evening. Question: When does the {place} {verb}?",
        "The {place} {verb}s at {x} in the {part}.",
    ),
    "kk": (
        "Мәтін: {place} таңғы {a}-{sa} ашылып, кешкі {b}-{sb} жабылады. Сұрақ: {place} қашан {verb}?",
        "{place} {part} {x}-{sx} {verbed}.",
    ),
}
PEOPLE_KEY = {"ru": "ru", "en": "en", "kk": "kk"}


def pools(language: str) -> list[str]:
    return load_config("knowledge.json")["people"][PEOPLE_KEY[language]]


def repeat_rows(copies: int = 3) -> list[Row]:
    rows: list[Row] = []
    words = load_config("knowledge.json")["words"]
    for language in LANGUAGES:
        for number, triple in enumerate(words):
            word = triple[WORD_INDEX[language]]
            rng = rng_for("repeat", language, number)
            for copy in range(copies):
                count = rng.choice(sorted(REPEAT_COUNTS[language]))
                prompt = REPEAT_PROMPTS[language][copy % 2].format(
                    w=word, n=REPEAT_COUNTS[language][count]
                )
                rows.append(
                    exchange(
                        "instruction",
                        "repeat_word",
                        language,
                        f"repeat:{language}:{number}",
                        prompt,
                        ", ".join([word] * count),
                        SOURCE,
                    )
                )
            prompt = UPPER_PROMPTS[language][number % 2].format(w=word)
            rows.append(
                exchange(
                    "instruction",
                    "uppercase",
                    language,
                    f"upper:{number}",
                    prompt,
                    word.upper(),
                    SOURCE,
                )
            )
    return rows


def compute(operation: str, a: int, b: int) -> int:
    return {"+": a + b, "-": a - b, "*": a * b}[operation]


def arithmetic_rows(per_language: int = 900) -> list[Row]:
    rows: list[Row] = []
    for language in LANGUAGES:
        rng = rng_for("arith", language)
        seen: set[tuple[str, int, int]] = set()
        while len(seen) < per_language:
            operation = rng.choice("+-*")
            top = 12 if operation == "*" else 99
            a, b = rng.randint(2, top), rng.randint(2, top)
            if operation == "-" and b > a:
                a, b = b, a
            if (operation, a, b) in seen or (operation, a, b) in HELD_OUT_ARITHMETIC:
                continue
            seen.add((operation, a, b))
            template = OPERATIONS[operation][language][len(seen) % 2]
            answer = f"{a} {SYMBOLS[operation]} {b} = {compute(operation, a, b)}."
            rows.append(
                exchange(
                    "math",
                    f"arithmetic_{operation}",
                    language,
                    f"arith:{operation}:{a}:{b}",
                    template.format(a=a, b=b),
                    answer,
                    SOURCE,
                )
            )
        prompt, answer_template = BASKET[language]
        for number in range(per_language // 4):
            a = rng.randint(10, 20)
            b = rng.randint(5, a - 5)
            noun = rng.choice(NOUNS[language])
            rows.append(
                exchange(
                    "math",
                    "basket",
                    language,
                    f"basket:{a}:{b}:{noun}",
                    prompt.format(a=a, b=b, noun=noun),
                    answer_template.format(a=a, b=b, c=a - b, noun=noun),
                    SOURCE,
                )
            )
    return rows


def memory_rows(per_language: int = 600) -> list[Row]:
    spec = load_config("knowledge.json")
    rows: list[Row] = []
    for language in LANGUAGES:
        rng = rng_for("memory", language)
        names = pools(language)
        for number in range(per_language):
            kind = number % 3
            if kind == 0:
                rows.append(name_memory(language, rng.choice(names), number))
            elif kind == 1:
                rows.append(pet_memory(language, rng, spec, number))
            else:
                rows.append(carry_memory(language, rng, number))
    return rows


NAME_STATEMENT = {
    "ru": (
        ["Меня зовут {n}. Запомни это.", "Моё имя {n}.", "Знай, что меня зовут {n}."],
        "Приятно познакомиться, {n}! Я запомню, как тебя зовут.",
    ),
    "en": (
        ["My name is {n}. Remember it.", "I am {n}.", "Please remember that my name is {n}."],
        "Nice to meet you, {n}! I will remember your name.",
    ),
    "kk": (
        [
            "Менің атым {n}. Осыны есте сақта.",
            "Мен {n} деп аталамын.",
            "Менің есімім {n} екенін біле жүр.",
        ],
        "Танысқаныма қуаныштымын, {n}! Атыңды есте сақтаймын.",
    ),
}
NAME_QUESTION = {
    "ru": (["Как меня зовут?", "Ты помнишь, как меня зовут?"], "Тебя зовут {n}."),
    "en": (["What is my name?", "Do you remember my name?"], "Your name is {n}."),
    "kk": (["Менің атым кім?", "Менің атымды білесің бе?"], "Сенің атың — {n}."),
}


def name_memory(language: str, name: str, number: int) -> Row:
    statements, ack = NAME_STATEMENT[language]
    questions, answer = NAME_QUESTION[language]
    return make_row(
        "multi_turn",
        "remember_name",
        language,
        f"name:{language}:{name}",
        [
            ("user", statements[number % 3].format(n=name)),
            ("assistant", ack.format(n=name)),
            ("user", questions[number % 2]),
            ("assistant", answer.format(n=name)),
        ],
        SOURCE,
    )


PET_STATEMENT = {
    "ru": (
        "У меня есть {a} по имени {an} и {b} по имени {bn}.",
        "Здорово! Запомню: {apos} зовут {an}, а {bpos} зовут {bn}.",
        "Как зовут {qpos}?",
        "{ypos} зовут {name}.",
    ),
    "en": (
        "I have a {a} named {an} and a {b} named {bn}.",
        "Great! I will remember: the {a} is {an} and the {b} is {bn}.",
        "What is my {q}'s name?",
        "Your {q}'s name is {name}.",
    ),
    "kk": (
        "Менің {apos} аты {an}, ал {bpos} аты {bn}.",
        "Керемет! Есімде сақтаймын: {apos} аты {an}, ал {bpos} аты {bn}.",
        "Менің {qpos} аты кім?",
        "Сенің {ypos} аты — {name}.",
    ),
}


def pet_memory(language: str, rng, spec: dict, number: int) -> Row:
    pets = spec["pets"][language]
    first, second = rng.sample(pets, 2)
    first_name, second_name = rng.sample(spec["pet_names"][language], 2)
    ask_first = number % 2 == 0
    asked, asked_name = (first, first_name) if ask_first else (second, second_name)
    statement, ack, question, answer = PET_STATEMENT[language]
    fields = {
        "a": first[0],
        "b": second[0],
        "an": first_name,
        "bn": second_name,
        "apos": first[1],
        "bpos": second[1],
        "qpos": asked[1],
        "ypos": asked[2],
        "q": asked[0],
        "name": asked_name,
    }
    ru_answer = capitalize(answer.format(**fields)) if language == "ru" else answer.format(**fields)
    return make_row(
        "multi_turn",
        "remember_pet",
        language,
        f"pet:{language}:{first_name}:{second_name}:{first[0]}:{second[0]}:{ask_first}",
        [
            ("user", statement.format(**fields)),
            ("assistant", ack.format(**fields)),
            ("user", question.format(**fields)),
            ("assistant", ru_answer),
        ],
        SOURCE,
    )


def carry_memory(language: str, rng, number: int) -> Row:
    while True:
        a, b, c = rng.randint(2, 40), rng.randint(2, 40), rng.randint(2, 30)
        operation = rng.choice("+-")
        total = a + b
        result = total + c if operation == "+" else total - c
        if (a, b) != (5, 3) and result >= 0:
            break
    question, answer = FOLLOW_UP[language][operation]
    fields = {"s": total, "c": c, "t": result}
    return make_row(
        "multi_turn",
        "carry_over",
        language,
        f"carry:{a}:{b}:{c}:{operation}",
        [
            ("user", FIRST_QUESTION[language].format(a=a, b=b)),
            ("assistant", f"{a} + {b} = {total}."),
            ("user", question.format(**fields)),
            ("assistant", answer.format(**fields)),
        ],
        SOURCE,
    )


PROFESSION_RENDERERS = {
    "ru": lambda job: RU_INSTRUMENTAL[job[0]],
    "en": lambda job: EN_ARTICLE.get(job[1], f"a {job[1]}"),
    "kk": lambda job: job[2],
}


def profession_rows(per_language: int = 500) -> list[Row]:
    rows: list[Row] = []
    items = next(
        item for item in load_config("knowledge.json")["categories"] if item["id"] == "professions"
    )["items"]
    for language in LANGUAGES:
        rng = rng_for("profession", language)
        names = pools(language)
        template, answer_template = PROFESSION_TEXT[language]
        render = PROFESSION_RENDERERS[language]
        for number in range(per_language):
            p1, p2 = rng.sample(names, 2)
            job1, job2 = rng.sample(items, 2)
            city = rng.randrange(len(CITIES[language]))
            asked_second = number % 2 == 0
            asked, asked_job = (p2, job2) if asked_second else (p1, job1)
            place = CITIES[language][city]
            fields = {
                "p1": p1,
                "p2": p2,
                "j1": render(job1),
                "j2": render(job2),
                "city_loc": f"в {place}" if language == "ru" else place,
                "q": asked,
                "ja": render(asked_job),
            }
            key = f"profession:{language}:{p1}:{p2}:{job1[0]}:{job2[0]}:{city}:{asked_second}"
            rows.append(
                exchange(
                    "context",
                    "profession",
                    language,
                    key,
                    template.format(**fields),
                    answer_template.format(**fields),
                    SOURCE,
                )
            )
    return rows


PLACE_VERBS = {
    "ru": ("закрывается", "открывается"),
    "en": ("close", "open"),
    "kk": ("жабылады", "ашылады"),
}
PLACE_PARTS = {"ru": ("вечера", "утра"), "en": ("evening", "morning"), "kk": ("кешкі", "таңғы")}


def place_rows(per_language: int = 300) -> list[Row]:
    rows: list[Row] = []
    for language in LANGUAGES:
        rng = rng_for("place", language)
        template, answer_template = PLACE_TEXT[language]
        for number in range(per_language):
            place, noun = rng.choice(PLACES[language])
            opens, closes = rng.randint(7, 11), rng.randint(4, 10)
            ask_close = number % 2 == 0
            target = closes if ask_close else opens
            side = 0 if ask_close else 1
            verb = PLACE_VERBS[language][side]
            fields = {
                "place": place,
                "noun": noun,
                "a": opens,
                "b": closes,
                "verb": verb,
                "verbed": verb,
                "part": PLACE_PARTS[language][side],
                "x": target,
                "sa": KK_TIME_SUFFIX[opens],
                "sb": KK_TIME_SUFFIX[closes],
                "sx": KK_TIME_SUFFIX[target],
            }
            key = f"place:{language}:{place}:{opens}:{closes}:{ask_close}"
            rows.append(
                exchange(
                    "context",
                    "opening_hours",
                    language,
                    key,
                    template.format(**fields),
                    answer_template.format(**fields),
                    SOURCE,
                )
            )
    return rows


def missing_rows(per_language: int = 400) -> list[Row]:
    rows: list[Row] = []
    for language in LANGUAGES:
        rng = rng_for("missing", language)
        names = pools(language)
        for number in range(per_language):
            person = rng.choice(names)
            city = rng.choice(CITIES[language])
            question, answer = MISSING_ATTRIBUTE[language][number % 3]
            text = {
                "ru": f"Текст: {person} живёт в {city}. Вопрос: ",
                "en": f"Text: {person} lives in {city}. Question: ",
                "kk": f"Мәтін: {person} {city} тұрады. Сұрақ: ",
            }[language]
            rows.append(
                exchange(
                    "context",
                    "answerability_missing",
                    language,
                    f"missing:{language}:{person}:{city}:{number % 3}",
                    text + question.format(p=person),
                    answer.format(p=person),
                    SOURCE,
                )
            )
    return rows


def task_rows() -> list[Row]:
    return [
        *repeat_rows(),
        *arithmetic_rows(),
        *memory_rows(),
        *profession_rows(),
        *place_rows(),
        *missing_rows(),
    ]
