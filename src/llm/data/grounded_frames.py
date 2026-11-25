from __future__ import annotations

import hashlib
from dataclasses import dataclass

from llm.data.grounded_text import (
    CAUSE_QUESTIONS,
    CAUSE_STATEMENTS,
    HOLDOUT_QUESTIONS,
    HOLDOUT_REVERSE,
    HOLDOUT_STATEMENTS,
    LANGUAGES,
    NEGATIONS,
    PRIORITY_STATEMENTS,
    QUESTIONS,
    REVERSE_QUESTIONS,
    SOURCE_CLAIMS,
    STATEMENTS,
    UPDATES,
)

LEGACY, DIVERSE = "legacy", "diverse"
TRAIN, DEV, TEST, HOLDOUT = "train", "dev", "test", "holdout"
TEMPLATE_SPLITS = (TRAIN, DEV, TEST, HOLDOUT)
LEGACY_TRAIN_COUNTS = {
    "statements": 4,
    "questions": 4,
    "reverse": 3,
    "negations": 3,
    "updates": 3,
    "cause_statements": 3,
    "cause_questions": 2,
    "priority": 2,
}
Tables = dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class Bank:
    train: tuple[str, ...]
    dev: tuple[str, ...]
    test: tuple[str, ...]
    holdout: tuple[str, ...] = ()

    def pool(self, split: str) -> tuple[str, ...]:
        chosen = getattr(self, split)
        return chosen if chosen else self.dev

    def pick(self, split: str, pick: int) -> str:
        pool = self.pool(split)
        return pool[pick % len(pool)]


def template_id(table: str, language: str, text: str) -> str:
    return f"{table}/{language}/{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


LEGACY_TABLES: dict[str, Tables] = {
    "statements": STATEMENTS,
    "questions": QUESTIONS,
    "reverse": REVERSE_QUESTIONS,
    "negations": NEGATIONS,
    "updates": UPDATES,
    "cause_statements": CAUSE_STATEMENTS,
    "cause_questions": CAUSE_QUESTIONS,
    "priority": PRIORITY_STATEMENTS,
}
HOLDOUTS: dict[str, Tables] = {
    "statements": HOLDOUT_STATEMENTS,
    "questions": HOLDOUT_QUESTIONS,
    "reverse": {language: (HOLDOUT_REVERSE[language],) for language in LANGUAGES},
}

DIVERSE_STATEMENTS: Tables = {
    "en": (
        "{X} is the {noun} of {E}.",
        "{E} has {X} as its {noun}.",
        "It is noted that the {noun} of {E} is {X}.",
        "Concerning {E}: the {noun} is {X}.",
        "The {noun} belonging to {E} is {X}.",
        "{E} — its {noun} is {X}, as stated here.",
        "You can see that {E} has the {noun} {X}.",
        "The text names {X} as the {noun} of {E}.",
    ),
    "ru": (
        "«{noun}» у {E} — {X}.",
        "{E} характеризуется так: «{noun}» — {X}.",
        "Известно, что у {E} «{noun}» — {X}.",
        "Про {E} сказано так: «{noun}» — {X}.",
        "{X} значится как «{noun}» у {E}.",
        "Что до «{noun}» у {E}, то это {X}.",
        "Здесь отмечено: «{noun}» для {E} — {X}.",
        "В тексте {X} назван «{noun}» для {E}.",
    ),
    "kk": (
        "{E} үшін «{noun}» — {X} болып табылады.",
        "{E} үшін «{noun}» ретінде {X} аталған.",
        "Мәтінде {E} үшін «{noun}» — {X} деп көрсетілген.",
        "{E} жайында: «{noun}» — {X}.",
        "{X} — {E} үшін «{noun}» болып саналады.",
        "{E} қонысының «{noun}» мәні — {X}.",
        "Бұл жерде {E} үшін «{noun}» ретінде {X} жазылған.",
        "«{noun}» дегенге келсек, {E} үшін ол — {X}.",
    ),
}
DIVERSE_QUESTIONS: Tables = {
    "en": (
        "Could you tell me the {noun} of {E}?",
        "I would like to know the {noun} of {E}.",
        "Please state the {noun} for {E}.",
        "Do you know what the {noun} of {E} is? Answer from the text.",
        "What {noun} does {E} have?",
        "Of {E}, what is the {noun}?",
        "The {noun} of {E} — what is it?",
        "Find the {noun} of {E} in the text.",
    ),
    "ru": (
        "Не могли бы вы сказать, каково «{noun}» у {E}?",
        "Мне нужно узнать «{noun}» для {E}.",
        "Укажите «{noun}» для {E}.",
        "Знаешь ли ты, чему равно «{noun}» у {E}? Ответь по тексту.",
        "Что является «{noun}» у {E}?",
        "Чему равно «{noun}» у {E}?",
        "«{noun}» у {E} — что это?",
        "Найди в тексте «{noun}» для {E}.",
    ),
    "kk": (
        "{E} үшін «{noun}» мәнін айтып бере аласыз ба?",
        "Маған {E} үшін «{noun}» мәнін білу керек.",
        "{E} үшін «{noun}» мәнін көрсетіңіз.",
        "{E} үшін «{noun}» неге тең екенін білесің бе? Мәтін бойынша жауап бер.",
        "{E} үшін «{noun}» не болып табылады?",
        "«{noun}» дегеніміз {E} үшін не?",
        "{E} үшін «{noun}» деген не?",
        "Мәтіннен {E} үшін «{noun}» мәнін тап.",
    ),
}
DIVERSE_REVERSE: Tables = {
    "en": (
        "To which settlement does the {noun} {X} belong?",
        "Name the settlement that has {X} as its {noun}.",
        "Which place has {X} as its {noun}?",
        "I am looking for the settlement with the {noun} {X}.",
        "Tell me which settlement lists {X} as its {noun}.",
        "Which place carries {X} as its {noun}?",
    ),
    "ru": (
        "Какому поселению принадлежит {X} как «{noun}»?",
        "Назови поселение, для которого «{noun}» — {X}.",
        "Какое поселение имеет {X} в качестве «{noun}»?",
        "Я ищу поселение, у которого «{noun}» равно {X}.",
        "Подскажи, какое поселение связано с {X} как с «{noun}».",
        "У какого места «{noun}» названо {X}?",
    ),
    "kk": (
        "{X} қай қонысқа «{noun}» ретінде тиесілі?",
        "«{noun}» ретінде {X} аталған қоныстың атын айт.",
        "Қай қоныста «{noun}» — {X}?",
        "Мен «{noun}» мәні {X} болатын қонысты іздеп жүрмін.",
        "Қай қоныстың «{noun}» мәні {X} деп көрсетілген?",
        "«{noun}» деп {X} саналатын қоныс қайсы?",
    ),
}
DIVERSE_NEGATIONS: Tables = {
    "en": (
        "It is a mistake to say that the {noun} of {E} is {W}; it is {X}.",
        "{E} does not have {W} as its {noun}; the {noun} is {X}.",
        "Not {W} but {X} is the {noun} of {E}.",
        "Do not confuse: {W} is not the {noun} of {E}, {X} is.",
        "For {E} the {noun} is {X}, and certainly not {W}.",
        "{W} would be wrong as the {noun} of {E}. The correct one is {X}.",
    ),
    "ru": (
        "Неверно, что «{noun}» у {E} — {W}; верное значение — {X}.",
        "У {E} нет «{noun}» {W}: оно равно {X}.",
        "Не {W}, а {X} является «{noun}» у {E}.",
        "Не путайте: для {E} «{noun}» — это {X}, а не {W}.",
        "Для {E} «{noun}» — {X}, и точно не {W}.",
        "{W} в качестве «{noun}» для {E} было бы ошибкой. Верно: {X}.",
    ),
    "kk": (
        "{E} үшін «{noun}» — {W} деу қате; дұрысы — {X}.",
        "{E} үшін «{noun}» мәні {W} емес, ол — {X}.",
        "{E} үшін «{noun}» — {W} емес, нақты {X}.",
        "Шатастырмаңыз: {E} үшін «{noun}» — {X}, {W} емес.",
        "{E} үшін «{noun}» — {X}, ал {W} мүлде емес.",
        "{W} деп көрсету {E} үшін «{noun}» ретінде қате болар еді. Дұрысы: {X}.",
    ),
}
DIVERSE_UPDATES: Tables = {
    "en": (
        "Newer information: {E} now has {X} as its {noun}.",
        "The {noun} of {E} has changed and is {X} today.",
        "Revised entry: {E}, {noun} {X}.",
        "After a recent check the {noun} of {E} was given as {X}.",
        "{E} was re-registered, and its {noun} became {X}.",
        "This replaces the old record: the {noun} of {E} is {X}.",
    ),
    "ru": (
        "Новые сведения: у {E} теперь «{noun}» — {X}.",
        "«{noun}» у {E} изменилось и сейчас равно {X}.",
        "Исправленная запись: {E}, {noun} {X}.",
        "После недавней проверки «{noun}» у {E} указали как {X}.",
        "{E} перерегистрировали, и его «{noun}» стало {X}.",
        "Эта запись заменяет прежнюю: «{noun}» у {E} — {X}.",
    ),
    "kk": (
        "Жаңа мәлімет: {E} үшін «{noun}» енді {X} болды.",
        "{E} үшін «{noun}» өзгеріп, бүгін {X} болып тұр.",
        "Түзетілген жазба: {E}, {noun} {X}.",
        "Жуырдағы тексеруден кейін {E} үшін «{noun}» {X} деп көрсетілді.",
        "{E} қайта тіркелді, оның «{noun}» мәні {X} болды.",
        "Бұл жазба бұрынғысын ауыстырады: {E} үшін «{noun}» — {X}.",
    ),
}
DIVERSE_CAUSE_STATEMENTS: Tables = {
    "en": (
        "{C} forced the market in {E} to close.",
        "The market in {E} did not open because of {C}.",
        "Trading stopped in {E}; the cause was {C}.",
        "Because of {C}, the market of {E} was closed.",
        "In {E} the market shut its doors owing to {C}.",
        "The closure of the market in {E} was caused by {C}.",
    ),
    "ru": (
        "Из-за такой причины рынок в {E} пришлось закрыть: {C}.",
        "Рынок в {E} не открылся по причине: {C}.",
        "В {E} торговля прекратилась; причина — {C}.",
        "Причиной закрытия рынка в {E} назван такой факт: {C}.",
        "В {E} рынок закрыл свои двери. Причина была такой: {C}.",
        "Закрытие рынка в {E} объясняется так: {C}.",
    ),
    "kk": (
        "{E} қонысындағы базарды жабуға мынадай себеп болды: {C}.",
        "{E} қонысында базар ашылмады; себебі — {C}.",
        "{E} қонысында сауда тоқтады; себеп — {C}.",
        "{E} қонысында базардың жабылуының басты себебі былай аталды: {C}.",
        "{E} қонысында базар есігін жапты. Себебі мынау: {C}.",
        "{E} қонысындағы базардың жабылуы былай түсіндіріледі: {C}.",
    ),
}
DIVERSE_CAUSE_QUESTIONS: Tables = {
    "en": (
        "Can you tell me why the market in {E} closed?",
        "For what reason was the market of {E} shut?",
        "I would like to know what closed the market in {E}.",
        "Give the cause of the market closure in {E}.",
    ),
    "ru": (
        "Можешь объяснить, почему в {E} закрыли рынок?",
        "По какой причине рынок в {E} был закрыт?",
        "Мне хотелось бы узнать, что закрыло рынок в {E}.",
        "Укажи, чем объясняется закрытие рынка в {E}.",
    ),
    "kk": (
        "{E} қонысындағы базардың неге жабылғанын айтып бере аласың ба?",
        "{E} қонысындағы базар қандай себеппен жабылды?",
        "{E} қонысында базарды не жапқанын білгім келеді.",
        "{E} қонысындағы базардың жабылу себебін көрсет.",
    ),
}
DIVERSE_PRIORITY: Tables = {
    "en": (
        "{Low} is less reliable than {High}.",
        "When {High} and {Low} disagree, {High} is the one to follow.",
        "Believe {High} rather than {Low}.",
        "Of the two sources, {High} has the better record.",
    ),
    "ru": (
        "{High} заслуживает больше доверия, чем {Low}.",
        "Если сведения расходятся, верным считается {High}, а не {Low}.",
        "Доверие вызывает {High}, но не {Low}.",
        "Из двух источников больше доверия у {High}.",
    ),
    "kk": (
        "Екі дерек қайшы келсе, дұрысы — {High}, {Low} емес.",
        "Сенімге лайық дерек — {High}; {Low} емес.",
        "{High} көбірек сенім тудырады, {Low} аз.",
        "Қайшылық болса, {High} айтқанына сену керек, {Low} айтқанына емес.",
    ),
}
DIVERSE_CLAIMS: Tables = {
    "en": (
        "{Scap} says the {noun} of {E} is {V}.",
        "Source ({S}): {E}, {noun} {V}.",
    ),
    "ru": (
        "{Snom}: «{noun}» у {E} — {V}.",
        "Запись из источника ({Snom}): {E}, «{noun}» — {V}.",
    ),
    "kk": (
        "{Snom}: {E} үшін «{noun}» — {V}.",
        "Дереккөз ({Snom}) бойынша {E} үшін «{noun}» — {V}.",
    ),
}
DIVERSE_ADDITIONS: dict[str, Tables] = {
    "statements": DIVERSE_STATEMENTS,
    "questions": DIVERSE_QUESTIONS,
    "reverse": DIVERSE_REVERSE,
    "negations": DIVERSE_NEGATIONS,
    "updates": DIVERSE_UPDATES,
    "cause_statements": DIVERSE_CAUSE_STATEMENTS,
    "cause_questions": DIVERSE_CAUSE_QUESTIONS,
    "priority": DIVERSE_PRIORITY,
}
PROFILE_ITEMS = {"en": "{noun}: {X}", "ru": "«{noun}» — {X}", "kk": "«{noun}» — {X}"}
PROFILE_FRAMES: Tables = {
    "en": ("{E} — {items}.", "Notes on {E}: {items}.", "What we know about {E}: {items}."),
    "ru": ("{E} — {items}.", "Сведения о {E}: {items}.", "Что известно про {E}: {items}."),
    "kk": (
        "{E} — {items}.",
        "{E} туралы мәліметтер: {items}.",
        "{E} жайында белгілі: {items}.",
    ),
}
TABLE_HEADERS = {
    "en": "Entity | Field | Value",
    "ru": "Объект | Поле | Значение",
    "kk": "Нысан | Өріс | Мән",
}
LABELS = {"en": ("Text", "Question"), "ru": ("Текст", "Вопрос"), "kk": ("Мәтін", "Сұрақ")}


def legacy_bank(table: str, language: str) -> Bank:
    texts = LEGACY_TABLES[table][language]
    cut = LEGACY_TRAIN_COUNTS[table]
    held = HOLDOUTS.get(table, {}).get(language, ())
    return Bank(texts[:cut], texts[cut:], texts[cut:], held)


def diverse_bank(table: str, language: str) -> Bank:
    base = legacy_bank(table, language)
    return Bank(
        (*base.train, *DIVERSE_ADDITIONS[table][language]), base.dev, base.test, base.holdout
    )


def build_banks(mode: str) -> dict[str, dict[str, Bank]]:
    make = legacy_bank if mode == LEGACY else diverse_bank
    banks = {
        table: {language: make(table, language) for language in LANGUAGES}
        for table in LEGACY_TABLES
    }
    banks["claims"] = {
        language: Bank(
            (SOURCE_CLAIMS[language], *(DIVERSE_CLAIMS[language] if mode == DIVERSE else ())),
            (SOURCE_CLAIMS[language],),
            (SOURCE_CLAIMS[language],),
        )
        for language in LANGUAGES
    }
    return banks


BANKS = {LEGACY: build_banks(LEGACY), DIVERSE: build_banks(DIVERSE)}


def split_ids(mode: str) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {split: set() for split in TEMPLATE_SPLITS}
    for table, languages in BANKS[mode].items():
        for language, bank in languages.items():
            for split in TEMPLATE_SPLITS:
                found[split].update(template_id(table, language, t) for t in getattr(bank, split))
    return found
