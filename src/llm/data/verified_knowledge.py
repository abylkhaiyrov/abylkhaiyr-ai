from __future__ import annotations

from llm.data.verified_common import (
    LANGUAGES,
    Row,
    capitalize,
    exchange,
    load_config,
    rng_for,
)

CAPITAL_FORWARD = {
    "ru": (["Какая столица {c}?", "Назови столицу {c}."], "Столица {c} — {cap}."),
    "en": (
        ["What is the capital of {c}?", "Name the capital of {c}."],
        "The capital of {c} is {cap}.",
    ),
    "kk": (["{c} астанасы қай қала?", "{c} астанасын ата."], "{c} астанасы — {cap}."),
}
CAPITAL_REVERSE = {
    "ru": (
        ["{cap} — столица какой страны?", "Столицей какой страны является {cap}?"],
        "{cap} — столица {c}.",
    ),
    "en": (
        ["{cap} is the capital of which country?", "Which country has {cap} as its capital?"],
        "{cap} is the capital of {c}.",
    ),
    "kk": (
        ["{cap} қай елдің астанасы?", "{cap} қай мемлекеттің астанасы болып табылады?"],
        "{cap} — {c} астанасы.",
    ),
}
COUNTRY_FORM = {"ru": 1, "en": 2, "kk": 4}
CAPITAL_FORM = {"ru": 5, "en": 6, "kk": 7}
TRANSLATION_PROMPTS = {
    ("ru", "en"): ["Переведи на английский: {x}", "Как по-английски «{x}»?"],
    ("ru", "kk"): ["Переведи на казахский: {x}", "Как по-казахски «{x}»?"],
    ("en", "ru"): ["Translate into Russian: {x}", 'How do you say "{x}" in Russian?'],
    ("en", "kk"): ["Translate into Kazakh: {x}", 'How do you say "{x}" in Kazakh?'],
    ("kk", "ru"): ["Орыс тіліне аудар: {x}", "«{x}» орысша қалай болады?"],
    ("kk", "en"): ["Ағылшын тіліне аудар: {x}", "«{x}» ағылшынша қалай болады?"],
}
TRANSLATION_WRAP = {
    "ru": {"en": "«{x}» по-английски — «{y}».", "kk": "«{x}» по-казахски — «{y}»."},
    "en": {"ru": '"{x}" in Russian is "{y}".', "kk": '"{x}" in Kazakh is "{y}".'},
    "kk": {"ru": "«{x}» орысша — «{y}».", "en": "«{x}» ағылшынша — «{y}»."},
}
LIST_PROMPTS = {
    "ru": ["Назови {n} {name}.", "Перечисли {n} {name}."],
    "en": ["Name {n} {name}.", "List {n} {name}."],
    "kk": ["{n} {name} ата.", "{n} {name} атап бер."],
}
LIST_COUNTS = (2, 3, 4)
LISTS_PER_COUNT = 5
TRANSLATION_INDEX = {"ru": 0, "en": 1, "kk": 2}
SOURCE = "verified_knowledge"


def capital_rows() -> list[Row]:
    rows: list[Row] = []
    for entry in load_config("knowledge.json")["capitals"]:
        key = f"capital:{entry[2]}"
        for language in LANGUAGES:
            country, cap = entry[COUNTRY_FORM[language]], entry[CAPITAL_FORM[language]]
            for table, tag in ((CAPITAL_FORWARD, "forward"), (CAPITAL_REVERSE, "reverse")):
                questions, answer = table[language]
                for question in questions:
                    rows.append(
                        exchange(
                            "facts",
                            f"capital_{tag}",
                            language,
                            key,
                            question.format(c=country, cap=cap),
                            answer.format(c=country, cap=cap),
                            SOURCE,
                        )
                    )
    return rows


def entry_rows(section: str, group: str, category: str) -> list[Row]:
    rows: list[Row] = []
    for entry in load_config("knowledge.json")[section]:
        for language in LANGUAGES:
            for question in entry[language]:
                rows.append(
                    exchange(
                        group,
                        category,
                        language,
                        f"{section}:{entry['id']}",
                        question,
                        entry[f"{language}_a"],
                        SOURCE,
                    )
                )
    return rows


def list_answer(items: list[str], numbered: bool) -> str:
    if numbered:
        return "\n".join(f"{index}. {item}" for index, item in enumerate(items, 1))
    return capitalize(", ".join(items)) + "."


def list_rows() -> list[Row]:
    rows: list[Row] = []
    for category in load_config("knowledge.json")["categories"]:
        for language in LANGUAGES:
            index = TRANSLATION_INDEX[language]
            for count in LIST_COUNTS:
                rng = rng_for("list", category["id"], language, count)
                for variant in range(LISTS_PER_COUNT):
                    chosen = [item[index] for item in rng.sample(category["items"], count)]
                    prompts = LIST_PROMPTS[language]
                    user = prompts[variant % 2].format(n=count, name=category[f"{language}_name"])
                    key = f"list:{category['id']}:{count}"
                    rows.append(
                        exchange(
                            "instruction",
                            "list_items",
                            language,
                            key,
                            user,
                            list_answer(chosen, numbered=variant % 3 == 0),
                            SOURCE,
                        )
                    )
    return rows


def translation_rows(section: str, entity_prefix: str) -> list[Row]:
    rows: list[Row] = []
    for number, triple in enumerate(load_config("knowledge.json")[section]):
        for (source, target), prompts in TRANSLATION_PROMPTS.items():
            x, y = triple[TRANSLATION_INDEX[source]], triple[TRANSLATION_INDEX[target]]
            wrap = TRANSLATION_WRAP[source][target]
            for variant, prompt in enumerate(prompts):
                if section != "words" and variant == 1:
                    continue
                answer = y if variant == 0 else wrap.format(x=x, y=y)
                rows.append(
                    exchange(
                        "translation",
                        f"{section}_{source}_{target}",
                        source,
                        f"{entity_prefix}:{number}",
                        prompt.format(x=x),
                        answer,
                        SOURCE,
                        answer_language=target if variant == 0 else source,
                    )
                )
    return rows


def social_rows() -> list[Row]:
    rows: list[Row] = []
    for intent in load_config("social.json")["intents"]:
        for language in LANGUAGES:
            spec = intent[language]
            for index, user in enumerate(spec["user"]):
                answer = spec["answer"][index % len(spec["answer"])]
                rows.append(
                    exchange(
                        "social",
                        intent["id"],
                        language,
                        f"social:{intent['id']}:{index}",
                        user,
                        answer,
                        SOURCE,
                    )
                )
    return rows


def knowledge_rows() -> list[Row]:
    return [
        *capital_rows(),
        *entry_rows("facts", "facts", "fact"),
        *entry_rows("concepts", "explain", "concept"),
        *list_rows(),
        *translation_rows("words", "word"),
        *translation_rows("sentences", "sentence"),
        *translation_rows("phrases", "phrase"),
        *social_rows(),
    ]
