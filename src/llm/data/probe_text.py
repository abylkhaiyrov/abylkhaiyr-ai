from __future__ import annotations

LANGUAGES = ("en", "ru", "kk")
RELATIONS = ("river", "founder", "mountain")
ENTITY_SYLLABLES = ("bre", "qui", "sto", "lyn", "vor", "kae", "thu", "mer", "ost", "ily", "gri")
PERSON_SYLLABLES = ("wil", "nex", "dra", "pho", "ari", "ulm", "eth")
SUFFIXES = ("a", "en", "is", "ar", "on", "ith", "ova")
MACHINE_NAMES = ("Orbix", "Tavelo", "Krunda", "Mestro", "Yvarn", "Dolpek")
TRAIN_QUESTIONS = 3
QUESTIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "en": {
        "river": (
            "What is the river of {E} called?",
            "Which river flows through {E}?",
            "{E} stands on the banks of which river?",
            "Name the river that {E} lies on.",
            "By what name do people call the river close to {E}?",
        ),
        "founder": (
            "Who founded {E}?",
            "Who is the founder of {E}?",
            "{E} was established by whom?",
            "Who built the settlement of {E}?",
            "Tell me who set up {E}.",
        ),
        "mountain": (
            "Which mountain stands near {E}?",
            "What is the mountain beside {E} called?",
            "Near {E} rises which mountain?",
            "Name the peak above {E}.",
            "Tell me the mountain that overlooks {E}.",
        ),
    },
    "ru": {
        "river": (
            "Как называется река у {E}?",
            "Какая река протекает через {E}?",
            "{E} стоит на берегу какой реки?",
            "Назови реку, на которой расположен {E}.",
            "Скажи, как называется река возле {E}.",
        ),
        "founder": (
            "Кто основал {E}?",
            "Кто основатель {E}?",
            "Кем был основан {E}?",
            "Кто построил поселение {E}?",
            "Скажи, кто основал {E}.",
        ),
        "mountain": (
            "Какая гора находится возле {E}?",
            "Как называется гора рядом с {E}?",
            "Возле {E} возвышается какая гора?",
            "Назови вершину над {E}.",
            "Скажи, какая гора возвышается над {E}.",
        ),
    },
    "kk": {
        "river": (
            "{E} маңындағы өзен қалай аталады?",
            "{E} арқылы қай өзен ағады?",
            "{E} қай өзеннің жағасында тұр?",
            "{E} орналасқан өзеннің атын ата.",
            "{E} жанындағы өзеннің атын айтшы.",
        ),
        "founder": (
            "{E} қонысын кім құрды?",
            "{E} қонысының негізін салушы кім?",
            "{E} кім тарапынан құрылған?",
            "{E} елді мекенін кім салды?",
            "{E} қонысын кім негіздегенін айтшы.",
        ),
        "mountain": (
            "{E} жанындағы тау қалай аталады?",
            "{E} қасында қай тау тұр?",
            "{E} үстінде қай тау көтеріледі?",
            "{E} үстінен қарап тұрған шыңның атын ата.",
            "{E} маңындағы таудың атын айтшы.",
        ),
    },
}
STATEMENTS: dict[str, dict[str, tuple[str, ...]]] = {
    "en": {
        "river": (
            "The river of {E} is called {X}.",
            "The river that flows through {E} is {X}.",
            "{E} stands on the banks of the river {X}.",
        ),
        "founder": (
            "{E} was founded by {X}.",
            "The founder of {E} is {X}.",
            "{E} was established by {X}.",
        ),
        "mountain": (
            "The mountain near {E} is {X}.",
            "The mountain beside {E} is called {X}.",
            "{X} rises near {E}.",
        ),
    },
    "ru": {
        "river": (
            "Река у {E} называется {X}.",
            "Через {E} протекает река {X}.",
            "{E} стоит на берегу реки {X}.",
        ),
        "founder": (
            "{E} основал {X}.",
            "Основатель {E} — {X}.",
            "{E} был основан человеком по имени {X}.",
        ),
        "mountain": (
            "Возле {E} находится гора {X}.",
            "Гора рядом с {E} называется {X}.",
            "Возле {E} возвышается {X}.",
        ),
    },
    "kk": {
        "river": (
            "{E} маңындағы өзен {X} деп аталады.",
            "{E} арқылы {X} өзені ағады.",
            "{E} {X} өзенінің жағасында тұр.",
        ),
        "founder": (
            "{E} қонысын {X} құрды.",
            "{E} қонысының негізін салушы — {X}.",
            "{E} қонысын {X} есімді адам құрған.",
        ),
        "mountain": (
            "{E} жанындағы тау {X} деп аталады.",
            "{E} қасында {X} тауы тұр.",
            "{E} маңында {X} тауы көтеріледі.",
        ),
    },
}
REVERSE_QUESTIONS: dict[str, dict[str, str]] = {
    "en": {
        "river": "Which settlement stands on the river {X}?",
        "founder": "Which settlement was founded by {X}?",
        "mountain": "Near which settlement is the mountain {X}?",
    },
    "ru": {
        "river": "Какой посёлок стоит на реке {X}?",
        "founder": "Какой посёлок основал {X}?",
        "mountain": "Возле какого посёлка находится гора {X}?",
    },
    "kk": {
        "river": "{X} өзенінің жағасында қай қоныс тұр?",
        "founder": "Қай қонысты {X} құрды?",
        "mountain": "{X} тауы қай қонысқа жақын?",
    },
}
NEGATED_STATEMENT = {
    "en": "The river of {E} is not {W}. It is {X}.",
    "ru": "Река у {E} называется не {W}, а {X}.",
    "kk": "{E} маңындағы өзен {W} емес, {X} деп аталады.",
}
CAUSE_STATEMENT = {
    "en": "In {E} the market closed. The reason: {C}.",
    "ru": "В {E} рынок закрылся. Причина: {C}.",
    "kk": "{E} қонысында базар жабылды. Себебі: {C}.",
}
CAUSE_QUESTION = {
    "en": "Why did the market close in {E}?",
    "ru": "Почему в {E} закрылся рынок?",
    "kk": "{E} қонысында базар неліктен жабылды?",
}
CAUSES: dict[str, tuple[tuple[str, str], ...]] = {
    "en": (("heavy snow", "snow"), ("a flood", "flood"), ("a power failure", "power")),
    "ru": (
        ("сильный снегопад", "снегопад"),
        ("наводнение", "наводнен"),
        ("отключение электричества", "электричеств"),
    ),
    "kk": (
        ("қатты қар жауды", "қар"),
        ("су тасқыны болды", "тасқын"),
        ("электр жарығы өшті", "жарық"),
    ),
}
RULE_TRAIN = {
    "en": "Machine {M} adds {K} to every input. What is the output for input {n}?",
    "ru": "Машина {M} прибавляет {K} к каждому входу. Что получится на выходе для входа {n}?",
    "kk": "{M} машинасы кез келген кірісіне {K} қосады. {n} кірісі үшін шығыс қандай?",
}
RULE_ALT = {
    "en": "Feed {n} into machine {M}. What number comes out?",
    "ru": "Подай {n} в машину {M}. Какое число получится?",
    "kk": "{n} санын {M} машинасына бер. Қандай сан шығады?",
}
RULE_ANSWER = {
    "en": "The output for {n} is {out}.",
    "ru": "Для {n} на выходе получится {out}.",
    "kk": "{n} үшін шығыс {out} болады.",
}
ANSWER_SUFFIX = {"en": "Answer:", "ru": "Ответ:", "kk": "Жауап:"}
