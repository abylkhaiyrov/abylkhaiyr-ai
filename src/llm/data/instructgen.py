from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Sequence

from llm.data import lexicon as lx
from llm.data.identity import QUESTIONS, IdentityFacts, identity_conversation

Row = dict
Generator = Callable[[random.Random, IdentityFacts], Row]

GREETINGS = {
    "en": ["Hi!", "Hello!", "Hey there!", "Good morning!", "Good evening!", "Hello, how are you?"],
    "ru": [
        "Привет!",
        "Здравствуйте!",
        "Добрый день!",
        "Доброе утро!",
        "Добрый вечер!",
        "Привет, как дела?",
    ],
    "kk": [
        "Сәлем!",
        "Сәлеметсіз бе!",
        "Қайырлы күн!",
        "Қайырлы таң!",
        "Қайырлы кеш!",
        "Сәлем, қалың қалай?",
    ],
}
GREETING_REPLIES = {
    "en": [
        "Hello! How can I help you today?",
        "Hi! I am {name}. What would you like to talk about?",
        (
            "Hello! I am a small program, so I do not have feelings, but I am ready to help. "
            "What do you need?"
        ),
    ],
    "ru": [
        "Здравствуйте! Чем могу помочь?",
        "Привет! Я {name}. О чём хотите поговорить?",
        (
            "Привет! Я всего лишь небольшая программа и у меня нет чувств, но я готов помочь. "
            "Что вам нужно?"
        ),
    ],
    "kk": [
        "Сәлеметсіз бе! Қалай көмектесе аламын?",
        "Сәлем! Мен {name}. Не туралы сөйлескіңіз келеді?",
        "Сәлем! Мен шағын бағдарламамын, сезімім жоқ, бірақ көмектесуге дайынмын. Сізге не керек?",
    ],
}
THANKS = {
    "en": ["Thank you!", "Thanks a lot!", "Thanks, that helps."],
    "ru": ["Спасибо!", "Большое спасибо!", "Спасибо, это помогло."],
    "kk": ["Рахмет!", "Көп рахмет!", "Рахмет, көмектесті."],
}
THANKS_REPLIES = {
    "en": ["You are welcome! Ask me anything else.", "Glad I could help!"],
    "ru": ["Пожалуйста! Спрашивайте что угодно ещё.", "Рад был помочь!"],
    "kk": ["Оқасы жоқ! Тағы бірдеңе сұрай аласыз.", "Көмектесе алғаныма қуаныштымын!"],
}
FAREWELLS = {
    "en": ["Goodbye!", "See you later!", "Bye!"],
    "ru": ["До свидания!", "Пока!", "До встречи!"],
    "kk": ["Сау болыңыз!", "Кездескенше!", "Қош болыңыз!"],
}
FAREWELL_REPLIES = {
    "en": ["Goodbye! Have a nice day.", "See you! Come back any time."],
    "ru": ["До свидания! Хорошего дня.", "До встречи! Возвращайтесь в любое время."],
    "kk": ["Сау болыңыз! Күніңіз сәтті өтсін.", "Кездескенше! Кез келген уақытта оралыңыз."],
}

CLARIFY_REQUESTS = {
    "en": [
        (
            "Translate this.",
            "Please send the text you want me to translate, and tell me the target language.",
        ),
        (
            "Fix it.",
            (
                "I am not sure what you want me to fix. Please send the text or code and "
                "describe the problem."
            ),
        ),
        ("Explain.", "What would you like me to explain? Please name the topic or send the text."),
        (
            "What about the other one?",
            "I do not see which one you mean. Could you tell me what you are comparing?",
        ),
        (
            "Summarize the document.",
            "I do not see any document in our chat. Please paste the text and I will summarize it.",
        ),
        ("Make it shorter.", "Please send the text you want shortened."),
    ],
    "ru": [
        (
            "Переведи это.",
            "Пришлите, пожалуйста, текст для перевода и укажите, на какой язык его перевести.",
        ),
        (
            "Исправь это.",
            "Я не уверен, что нужно исправить. Пришлите текст или код и опишите проблему.",
        ),
        ("Объясни.", "Что именно объяснить? Назовите тему или пришлите текст."),
        (
            "А что насчёт другого?",
            "Я не вижу, какой именно вариант вы имеете в виду. Уточните, что мы сравниваем.",
        ),
        (
            "Перескажи документ.",
            "В нашем чате нет документа. Вставьте текст, и я его кратко перескажу.",
        ),
        ("Сделай короче.", "Пришлите, пожалуйста, текст, который нужно сократить."),
    ],
    "kk": [
        ("Мынаны аудар.", "Аудару керек мәтінді жіберіңіз және қай тілге аудару керегін айтыңыз."),
        (
            "Мынаны түзет.",
            "Нені түзету керегін түсінбедім. Мәтінді немесе кодты жіберіп, мәселені сипаттаңыз.",
        ),
        ("Түсіндір.", "Нені түсіндіру керек? Тақырыпты атаңыз немесе мәтінді жіберіңіз."),
        (
            "Ал екіншісі ше?",
            "Қайсысын айтып тұрғаныңызды көрмеймін. Нені салыстырып жатқаныңызды нақтылаңыз.",
        ),
        (
            "Құжатты қысқаша айтып бер.",
            "Біздің чатта құжат жоқ. Мәтінді қойыңыз, мен оны қысқаша айтып беремін.",
        ),
        ("Қысқартып жаз.", "Қысқарту керек мәтінді жіберіңіз."),
    ],
}

UNKNOWN_REQUESTS = {
    "en": [
        (
            "What is the weather in my city right now?",
            (
                "I do not have access to the internet or "
                "real-time data, so I cannot tell you the current weather. A weather app or website "
                "will give you an accurate forecast."
            ),
        ),
        (
            "What is today's dollar exchange rate?",
            (
                "I cannot check live data, so I do not know "
                "today's exchange rate. Please check a bank or a financial website."
            ),
        ),
        (
            "Which numbers will win the lottery tomorrow?",
            "Nobody can predict lottery numbers, and I certainly cannot. They are random.",
        ),
        (
            "What did I eat for breakfast?",
            (
                "I do not know. I cannot see your life and I do not "
                "remember anything beyond this conversation."
            ),
        ),
        (
            "What is the latest news today?",
            (
                "I have no access to current news. My knowledge "
                "comes from my training data, so please use a news website."
            ),
        ),
        (
            "Tell me the phone number of my neighbor.",
            "I do not have personal information about people, and I would not share it if I did.",
        ),
    ],
    "ru": [
        (
            "Какая сейчас погода в моём городе?",
            (
                "У меня нет доступа к интернету и данным в "
                "реальном времени, поэтому я не могу назвать текущую погоду. Точный прогноз покажет "
                "приложение или сайт погоды."
            ),
        ),
        (
            "Какой сегодня курс доллара?",
            (
                "Я не могу проверять актуальные данные, поэтому не знаю "
                "сегодняшний курс. Проверьте его в банке или на финансовом сайте."
            ),
        ),
        (
            "Какие числа выиграют в лотерею завтра?",
            "Никто не может предсказать числа лотереи, и я тоже. Они случайны.",
        ),
        (
            "Что я ел сегодня на завтрак?",
            "Я не знаю. Я не вижу вашу жизнь и ничего не помню за пределами этого разговора.",
        ),
        (
            "Какие сегодня последние новости?",
            (
                "У меня нет доступа к актуальным новостям. Мои "
                "знания основаны на обучающих данных, поэтому воспользуйтесь новостным сайтом."
            ),
        ),
        (
            "Скажи номер телефона моего соседа.",
            (
                "У меня нет личной информации о людях, и я не "
                "стал бы ею делиться, даже если бы она была."
            ),
        ),
    ],
    "kk": [
        (
            "Менің қаламда қазір ауа райы қандай?",
            (
                "Менде интернетке және нақты уақыттағы "
                "деректерге қолжетімділік жоқ, сондықтан қазіргі ауа райын айта алмаймын. Нақты "
                "болжамды ауа райы қосымшасынан немесе сайттан көре аласыз."
            ),
        ),
        (
            "Бүгін доллар бағамы қандай?",
            (
                "Мен нақты уақыттағы деректерді тексере алмаймын, "
                "сондықтан бүгінгі бағамды білмеймін. Банктен немесе қаржы сайтынан тексеріңіз."
            ),
        ),
        (
            "Ертең лотереяда қандай сандар ұтады?",
            "Лотерея сандарын ешкім болжай алмайды, мен де болжай алмаймын. Олар кездейсоқ.",
        ),
        (
            "Мен таңертең не жедім?",
            (
                "Білмеймін. Мен сіздің өміріңізді көрмеймін және осы "
                "әңгімеден тыс ештеңе есімде сақтамаймын."
            ),
        ),
        (
            "Бүгінгі соңғы жаңалықтар қандай?",
            (
                "Менде өзекті жаңалықтарға қолжетімділік жоқ. "
                "Білімім оқыту деректеріне негізделген, сондықтан жаңалық сайтын пайдаланыңыз."
            ),
        ),
        (
            "Көршімнің телефон нөмірін айтшы.",
            "Менде адамдар туралы жеке ақпарат жоқ, болса да бөліспес едім.",
        ),
    ],
}

FAKE_SYLLABLES = ("zor", "vel", "kam", "dun", "tar", "mil", "fen", "sar", "ruk", "lan", "pox")
FAKE_TEMPLATES = {
    "en": (
        "Who is the president of {place}?",
        (
            "I do not know of a country called {place}. I may simply not have information about "
            "it, and I would rather not invent an answer."
        ),
    ),
    "ru": (
        "Кто президент страны {place}?",
        (
            "Я не знаю страны под названием {place}. Возможно, у меня просто нет информации о ней, "
            "и я не хочу выдумывать ответ."
        ),
    ),
    "kk": (
        "{place} елінің президенті кім?",
        (
            "Мен {place} деп аталатын елді білмеймін. Менде ол туралы ақпарат болмауы мүмкін, "
            "жауапты ойдан шығарғым келмейді."
        ),
    ),
}

CONTEXT_WRAPPERS = {
    "en": [
        "Read the text and answer the question.\n\nText: {text}\n\nQuestion: {question}",
        "{text}\n\nBased on the text above: {question}",
        "Use only this text to answer.\nText: {text}\nQuestion: {question}",
    ],
    "ru": [
        "Прочитай текст и ответь на вопрос.\n\nТекст: {text}\n\nВопрос: {question}",
        "{text}\n\nПо тексту выше: {question}",
        "Отвечай только по этому тексту.\nТекст: {text}\nВопрос: {question}",
    ],
    "kk": [
        "Мәтінді оқып, сұраққа жауап бер.\n\nМәтін: {text}\n\nСұрақ: {question}",
        "{text}\n\nЖоғарыдағы мәтін бойынша: {question}",
        "Тек осы мәтін бойынша жауап бер.\nМәтін: {text}\nСұрақ: {question}",
    ],
}
NOT_STATED = {
    "en": "The text does not say that.",
    "ru": "В тексте об этом не сказано.",
    "kk": "Мәтінде бұл туралы айтылмаған.",
}
MISSING_QUESTIONS = {
    "en": "What is {name}'s favorite color?",
    "ru": "Какой любимый цвет у человека по имени {name}?",
    "kk": "{name} есімді адамның сүйікті түсі қандай?",
}

SENTENCE_REQUESTS = {
    "count_words": {
        "en": ["How many words are in this sentence: {s}", "Count the words: {s}"],
        "ru": ["Сколько слов в этом предложении: {s}", "Посчитай слова: {s}"],
        "kk": ["Бұл сөйлемде неше сөз бар: {s}", "Сөздерді санап бер: {s}"],
    },
    "uppercase": {
        "en": ["Write this sentence in capital letters: {s}", "Convert to uppercase: {s}"],
        "ru": ["Напиши это предложение заглавными буквами: {s}", "Преврати в верхний регистр: {s}"],
        "kk": ["Бұл сөйлемді бас әріптермен жаз: {s}", "Бас әріпке айналдыр: {s}"],
    },
    "reverse": {
        "en": ["Write the words of this sentence in reverse order: {s}"],
        "ru": ["Напиши слова этого предложения в обратном порядке: {s}"],
        "kk": ["Осы сөйлемнің сөздерін кері ретпен жаз: {s}"],
    },
    "longest": {
        "en": ["Which word is the longest in this sentence: {s}"],
        "ru": ["Какое слово самое длинное в этом предложении: {s}"],
        "kk": ["Бұл сөйлемдегі ең ұзын сөз қайсы: {s}"],
    },
    "first_last": {
        "en": ["What are the first and the last words of this sentence: {s}"],
        "ru": ["Какие первое и последнее слова в этом предложении: {s}"],
        "kk": ["Бұл сөйлемнің бірінші және соңғы сөздері қандай: {s}"],
    },
}
SENTENCE_ANSWERS = {
    "count_words": {
        "en": "The sentence has {n} words.",
        "ru": "В предложении {n} слов.",
        "kk": "Сөйлемде {n} сөз бар.",
    },
    "longest": {
        "en": "The longest word is “{w}”.",
        "ru": "Самое длинное слово — «{w}».",
        "kk": "Ең ұзын сөз — «{w}».",
    },
    "first_last": {
        "en": "The first word is “{a}” and the last word is “{b}”.",
        "ru": "Первое слово — «{a}», последнее слово — «{b}».",
        "kk": "Бірінші сөз — «{a}», соңғы сөз — «{b}».",
    },
}
PARAGRAPH_REQUEST = {
    "en": "How many sentences are in this text: {s}",
    "ru": "Сколько предложений в этом тексте: {s}",
    "kk": "Бұл мәтінде неше сөйлем бар: {s}",
}
PARAGRAPH_ANSWER = {
    "en": "The text has {n} sentences.",
    "ru": "В тексте {n} предложения.",
    "kk": "Мәтінде {n} сөйлем бар.",
}

LIST_KINDS = {
    "colors": lx.COLORS,
    "animals": lx.ANIMALS,
    "seasons": lx.SEASONS,
}
LIST_COUNT_WORDS = {
    "en": {2: "two", 3: "three"},
    "ru": {2: "два", 3: "три"},
    "kk": {2: "екі", 3: "үш"},
}
LIST_KIND_PLURAL = {
    "colors": {"en": "colors", "ru": "цвета", "kk": "түсті"},
    "animals": {"en": "animals", "ru": "животных", "kk": "жануарды"},
    "seasons": {"en": "seasons of the year", "ru": "времени года", "kk": "жыл мезгілін"},
}
LIST_REQUEST_FORMS = {
    "en": "Name {n} {kind}.",
    "ru": "Назови {n} {kind}.",
    "kk": "{n} {kind} ата.",
}

TRANSLATION_REQUESTS = {
    "word": {
        "en": ["Translate the word “{w}” into {tl}.", "How do you say “{w}” in {tl}?"],
        "ru": ["Переведи слово «{w}» на {tl}.", "Как будет «{w}» на {tl}?"],
        "kk": ["«{w}» сөзін {tl} тіліне аудар.", "«{w}» {tl} тілінде қалай болады?"],
    },
    "sentence": {
        "en": ["Translate into {tl}: {s}", "Please translate this sentence into {tl}: {s}"],
        "ru": ["Переведи на {tl}: {s}", "Пожалуйста, переведи это предложение на {tl}: {s}"],
        "kk": ["{tl} тіліне аудар: {s}", "Бұл сөйлемді {tl} тіліне аударып бер: {s}"],
    },
}
TARGET_NAMES = {
    "en": {"en": "English", "ru": "Russian", "kk": "Kazakh"},
    "ru": {"en": "английский", "ru": "русский", "kk": "казахский"},
    "kk": {"en": "ағылшын", "ru": "орыс", "kk": "қазақ"},
}
TRANSLATION_ANSWERS = {
    "en": "“{w}” in {tl} is “{t}”.",
    "ru": "«{w}» по-{tl} будет «{t}».",
    "kk": "«{w}» {tl} тілінде — «{t}».",
}
TRANSLATION_ANSWER_NAMES = {
    "en": {"en": "English", "ru": "Russian", "kk": "Kazakh"},
    "ru": {"en": "английски", "ru": "русски", "kk": "казахски"},
    "kk": {"en": "ағылшын", "ru": "орыс", "kk": "қазақ"},
}

CONCEPT_QUESTIONS = {
    "en": [
        "What is {c}?",
        "Explain {c} in simple words.",
        "Can you tell me what {c} means?",
        "Define {c} briefly.",
        "I do not understand {c}. Could you explain it?",
    ],
    "ru": [
        "Что такое {c}?",
        "Объясни простыми словами, что такое {c}.",
        "Дай определение: {c}.",
        "Я не понимаю, что такое {c}. Можешь объяснить?",
        "{C} — что это?",
    ],
    "kk": [
        "{C} дегеніміз не?",
        "{C} деген не? Қарапайым тілмен түсіндір.",
        "{C} түсінігін қысқаша анықтап бер.",
        "Маған {c} туралы айтып бер.",
        "Мен {c} дегенді түсінбедім. Түсіндіре аласың ба?",
    ],
}
CONCEPT_CLOSERS = {
    "en": ["", "", " Would you like an example?"],
    "ru": ["", "", " Хотите пример?"],
    "kk": ["", "", " Мысал келтірейін бе?"],
}

NAME_INTRO = {
    "en": (
        "My name is {name}.",
        "Nice to meet you, {name}!",
        "What is my name?",
        "Your name is {name}.",
    ),
    "ru": (
        "Меня зовут {name}.",
        "Приятно познакомиться, {name}!",
        "Как меня зовут?",
        "Вас зовут {name}.",
    ),
    "kk": (
        "Менің атым {name}.",
        "Танысқаныма қуаныштымын, {name}!",
        "Менің атым кім?",
        "Сіздің атыңыз — {name}.",
    ),
}
CARRY_REQUESTS = {
    "en": (
        "What is {a} + {b}?",
        "{a} + {b} = {s}.",
        "Now multiply the result by {k}.",
        "{s} × {k} = {p}.",
    ),
    "ru": (
        "Сколько будет {a} + {b}?",
        "{a} + {b} = {s}.",
        "Теперь умножь результат на {k}.",
        "{s} × {k} = {p}.",
    ),
    "kk": (
        "{a} + {b} неше болады?",
        "{a} + {b} = {s}.",
        "Енді нәтижені {k} санына көбейт.",
        "{s} × {k} = {p}.",
    ),
}
TRANSLATE_FOLLOWUP = {
    "en": "And how do you say it in {tl}?",
    "ru": "А как это будет на {tl}?",
    "kk": "Ал {tl} тілінде қалай болады?",
}


def _pick(rng: random.Random, options: Sequence):
    return options[rng.randrange(len(options))]


def _row(
    category: str, language: str, turns: list[tuple[str, str]], group: str = "synthetic"
) -> Row:
    messages: list[dict[str, str]] = []
    for user, assistant in turns:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return {
        "source": f"synthetic_{category}",
        "group": group,
        "category": category,
        "language": language,
        "messages": messages,
    }


def _language(rng: random.Random) -> str:
    return _pick(rng, lx.LANGUAGES)


def greeting(rng: random.Random, facts: IdentityFacts) -> Row:
    language = _language(rng)
    pools = (
        (GREETINGS, GREETING_REPLIES),
        (THANKS, THANKS_REPLIES),
        (FAREWELLS, FAREWELL_REPLIES),
    )
    prompts, replies = _pick(rng, pools)
    reply = _pick(rng, replies[language]).format(name=facts.name)
    return _row("greeting", language, [(_pick(rng, prompts[language]), reply)])


def clarification(rng: random.Random, facts: IdentityFacts) -> Row:
    language = _language(rng)
    return _row("clarification", language, [_pick(rng, CLARIFY_REQUESTS[language])])


def unknown_knowledge(rng: random.Random, facts: IdentityFacts) -> Row:
    language = _language(rng)
    return _row("unknown_knowledge", language, [_pick(rng, UNKNOWN_REQUESTS[language])])


def unknown_entity(rng: random.Random, facts: IdentityFacts) -> Row:
    language = _language(rng)
    place = "".join(rng.choice(FAKE_SYLLABLES) for _ in range(3)).capitalize()
    question, answer = FAKE_TEMPLATES[language]
    return _row(
        "unknown_entity", language, [(question.format(place=place), answer.format(place=place))]
    )


def general_fact(rng: random.Random, facts: IdentityFacts) -> Row:
    language = _language(rng)
    fact = _pick(rng, lx.FACTS)
    return _row(
        "general_fact",
        language,
        [(fact.question.in_language(language), fact.answer.in_language(language))],
    )


def concept_row(rng: random.Random, concepts: list[lx.Concept], category: str, group: str) -> Row:
    language = _language(rng)
    concept = _pick(rng, concepts)
    name = concept.name.in_language(language)
    question = _pick(rng, CONCEPT_QUESTIONS[language]).format(c=name, C=name[:1].upper() + name[1:])
    answer = concept.definition.in_language(language) + _pick(rng, CONCEPT_CLOSERS[language])
    row = _row(category, language, [(question, answer)])
    row["group"] = group
    return row


def explain_concept(rng: random.Random, facts: IdentityFacts) -> Row:
    return concept_row(rng, lx.CONCEPTS, "explain_concept", "dialog")


def explain_code_concept(rng: random.Random, facts: IdentityFacts) -> Row:
    return concept_row(rng, lx.CODE_CONCEPTS, "explain_code_concept", "code")


def _person_line(language: str, name: str, job: lx.Term, city: lx.Term, age: int) -> str:
    title = job.in_language(language)
    place = city.in_language(language)
    if language == "en":
        article = "an" if title[0] in "aeiou" else "a"
        return f"{name} is {article} {title} from {place} and is {age} years old."
    if language == "ru":
        return f"{name} — {title} из города {place}, ему {age} лет."
    return f"{name} — {title}, {place} қаласында тұрады, {age} жаста."


def _attribute_qa(
    language: str, name: str, job: lx.Term, city: lx.Term, age: int, attribute: str
) -> tuple[str, str]:
    title, place = job.in_language(language), city.in_language(language)
    if language == "en":
        article = "an" if title[0] in "aeiou" else "a"
        return {
            "job": (f"What does {name} do for a living?", f"{name} works as {article} {title}."),
            "city": (f"Which city is {name} from?", f"{name} is from {place}."),
            "age": (f"How old is {name}?", f"{name} is {age} years old."),
        }[attribute]
    if language == "ru":
        return {
            "job": (f"Кем работает {name}?", f"{name} работает {lx.JOB_INSTRUMENTAL[title]}."),
            "city": (f"Из какого города {name}?", f"{name} из города {place}."),
            "age": (f"Сколько лет человеку по имени {name}?", f"Ему {age} лет."),
        }[attribute]
    return {
        "job": (f"{name} кім болып жұмыс істейді?", f"{name} {title} болып жұмыс істейді."),
        "city": (f"{name} қай қалада тұрады?", f"{name} {place} қаласында тұрады."),
        "age": (f"{name} неше жаста?", f"{name} {age} жаста."),
    }[attribute]


def context_qa(rng: random.Random, facts: IdentityFacts) -> Row:
    language = _language(rng)
    indexes = rng.sample(range(len(lx.PEOPLE[language])), 2)
    cities = rng.sample(lx.CITIES, 2)
    jobs = rng.sample(lx.JOBS, 2)
    ages = rng.sample(lx.AGES, 2)
    names = [lx.PEOPLE[language][i] for i in indexes]
    text = " ".join(_person_line(language, names[i], jobs[i], cities[i], ages[i]) for i in range(2))
    target = rng.randrange(2)
    if rng.random() < 0.15:
        question = MISSING_QUESTIONS[language].format(name=names[target])
        answer = NOT_STATED[language]
    else:
        attribute = _pick(rng, ("job", "city", "age"))
        question, answer = _attribute_qa(
            language, names[target], jobs[target], cities[target], ages[target], attribute
        )
    prompt = _pick(rng, CONTEXT_WRAPPERS[language]).format(text=text, question=question)
    return _row("context_qa", language, [(prompt, answer)])


def _split_sentences(text: str) -> int:
    return len([part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part])


def text_analysis(rng: random.Random, facts: IdentityFacts) -> Row:
    language = _language(rng)
    if rng.random() < 0.2:
        count = rng.randint(2, 3)
        paragraph = " ".join(rng.sample(lx.SENTENCES[language], count))
        question = PARAGRAPH_REQUEST[language].format(s=paragraph)
        answer = PARAGRAPH_ANSWER[language].format(n=_split_sentences(paragraph))
        return _row("text_analysis", language, [(question, answer)])
    sentence = _pick(rng, lx.SENTENCES[language])
    words = re.findall(r"[^\W\d_]+(?:-[^\W\d_]+)*", sentence)
    task = _pick(rng, tuple(SENTENCE_REQUESTS))
    question = _pick(rng, SENTENCE_REQUESTS[task][language]).format(s=sentence)
    if task == "uppercase":
        answer = sentence.upper()
    elif task == "reverse":
        answer = " ".join(reversed(words)) + "."
    elif task == "count_words":
        answer = SENTENCE_ANSWERS[task][language].format(n=len(words))
    elif task == "longest":
        answer = SENTENCE_ANSWERS[task][language].format(w=max(words, key=len))
    else:
        answer = SENTENCE_ANSWERS[task][language].format(a=words[0], b=words[-1])
    return _row("text_analysis", language, [(question, answer)])


def list_items(rng: random.Random, facts: IdentityFacts) -> Row:
    language = _language(rng)
    kind = _pick(rng, tuple(LIST_KINDS))
    pool = LIST_KINDS[kind]
    count = rng.randint(2, 3)
    chosen = rng.sample(pool, count)
    question = LIST_REQUEST_FORMS[language].format(
        n=LIST_COUNT_WORDS[language][count], kind=LIST_KIND_PLURAL[kind][language]
    )
    answer = "\n".join(f"{i}. {item.in_language(language)}" for i, item in enumerate(chosen, 1))
    return _row("list_items", language, [(question, answer)])


def translation(rng: random.Random, facts: IdentityFacts) -> Row:
    _language(rng)
    source, target = rng.sample(lx.LANGUAGES, 2)
    pool = _pick(rng, (lx.NOUNS, lx.NUMBERS, lx.COLORS, lx.PHRASES))
    term = _pick(rng, pool)
    target_name = TARGET_NAMES[source][target]
    if rng.random() < 0.35 and pool is lx.NOUNS:
        sentence = lx.SENTENCE_FRAMES[source].format(word=term.in_language(source))
        answer = lx.SENTENCE_FRAMES[target].format(word=term.in_language(target))
        question = _pick(rng, TRANSLATION_REQUESTS["sentence"][source]).format(
            tl=target_name, s=sentence
        )
        return _row("translation", source, [(question, answer)])
    question = _pick(rng, TRANSLATION_REQUESTS["word"][source]).format(
        w=term.in_language(source), tl=target_name
    )
    answer = TRANSLATION_ANSWERS[source].format(
        w=term.in_language(source),
        tl=TRANSLATION_ANSWER_NAMES[source][target],
        t=term.in_language(target),
    )
    return _row("translation", source, [(question, answer)])


def multi_turn_name(rng: random.Random, facts: IdentityFacts) -> Row:
    language = _language(rng)
    name = _pick(rng, lx.PEOPLE[language])
    intro, welcome, ask, answer = NAME_INTRO[language]
    turns = [(intro.format(name=name), welcome.format(name=name)), (ask, answer.format(name=name))]
    return _row("multi_turn_name", language, turns)


def multi_turn_math(rng: random.Random, facts: IdentityFacts) -> Row:
    language = _language(rng)
    a, b, k = rng.randint(2, 60), rng.randint(2, 60), rng.randint(2, 9)
    first_q, first_a, second_q, second_a = CARRY_REQUESTS[language]
    total = a + b
    turns = [
        (first_q.format(a=a, b=b), first_a.format(a=a, b=b, s=total)),
        (second_q.format(k=k), second_a.format(s=total, k=k, p=total * k)),
    ]
    return _row("multi_turn_math", language, turns)


def multi_turn_translation(rng: random.Random, facts: IdentityFacts) -> Row:
    language = _language(rng)
    term = _pick(rng, lx.NOUNS)
    source = language
    first, second = rng.sample([code for code in lx.LANGUAGES if code != source], 2)
    first_row = translation_pair(source, first, term)
    second_row = translation_pair(source, second, term)
    follow = TRANSLATE_FOLLOWUP[source].format(tl=TARGET_NAMES[source][second])
    return _row("multi_turn_translation", source, [first_row, (follow, second_row[1])])


def translation_pair(source: str, target: str, term: lx.Term) -> tuple[str, str]:
    question = TRANSLATION_REQUESTS["word"][source][0].format(
        w=term.in_language(source), tl=TARGET_NAMES[source][target]
    )
    answer = TRANSLATION_ANSWERS[source].format(
        w=term.in_language(source),
        tl=TRANSLATION_ANSWER_NAMES[source][target],
        t=term.in_language(target),
    )
    return question, answer


def multi_turn_identity(rng: random.Random, facts: IdentityFacts) -> Row:
    language = _language(rng)
    greeting_text = _pick(rng, GREETINGS[language])
    reply = _pick(rng, GREETING_REPLIES[language]).format(name=facts.name)
    intent = _pick(rng, tuple(QUESTIONS))
    follow = identity_conversation(rng, facts, language, intent)["messages"]
    turns = [
        (greeting_text, reply),
        (follow[0]["content"], follow[1]["content"]),
    ]
    row = _row("multi_turn_identity", language, turns)
    row["group"] = "identity"
    return row


CATEGORIES: dict[str, tuple[float, str, Generator]] = {
    "greeting": (0.05, "dialog", greeting),
    "clarification": (0.04, "dialog", clarification),
    "unknown_knowledge": (0.05, "dialog", unknown_knowledge),
    "unknown_entity": (0.03, "dialog", unknown_entity),
    "general_fact": (0.06, "dialog", general_fact),
    "explain_concept": (0.07, "dialog", explain_concept),
    "explain_code_concept": (0.02, "code", explain_code_concept),
    "multi_turn_name": (0.03, "dialog", multi_turn_name),
    "multi_turn_identity": (0.03, "identity", multi_turn_identity),
    "context_qa": (0.22, "skills", context_qa),
    "text_analysis": (0.12, "skills", text_analysis),
    "list_items": (0.05, "skills", list_items),
    "translation": (0.14, "skills", translation),
    "multi_turn_math": (0.06, "skills", multi_turn_math),
    "multi_turn_translation": (0.05, "skills", multi_turn_translation),
}
ATTEMPT_FACTOR = 40


def build_instruction_dataset(facts: IdentityFacts, count: int, seed: int) -> list[Row]:
    rng = random.Random(seed)
    rows: list[Row] = []
    for weight, group, generator in CATEGORIES.values():
        target = round(count * weight)
        seen: set[str] = set()
        produced = 0
        for _ in range(target * ATTEMPT_FACTOR):
            if produced >= target:
                break
            row = generator(rng, facts)
            key = json.dumps(row["messages"], ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            row["group"] = group
            rows.append(row)
            produced += 1
    rng.shuffle(rows)
    return rows
