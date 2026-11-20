from __future__ import annotations

import re

DIALOGUE = "dialogue"
INFORMATION = "information"
MATH = "math"
LOGIC = "logic"
CODE = "code"
ANALYSIS = "analysis"
TRANSLATION = "translation"
GENERAL = "general"

INTENTS = (DIALOGUE, INFORMATION, MATH, LOGIC, CODE, ANALYSIS, TRANSLATION)
VERIFIED_KINDS = {MATH, LOGIC, CODE}

FLAGS = re.IGNORECASE
CODE_HINTS = re.compile(
    r"```|\b(code|function|def|class|sql|query|python|java|javascript|typescript|algorithm|"
    r"функци\w*|код\w*|программ\w*|алгоритм\w*|бағдарлама\w*|функция\w*)\b",
    FLAGS,
)
MATH_HINTS = re.compile(
    r"\d\s*[-+*/^=×÷]\s*\d|\b(solve|compute|calculate|integral|derivative|probability|mean|"
    r"determinant|equation|percent\w*|вычисли\w*|реши\w*|посчитай|уравнени\w*|процент\w*|"
    r"есепте\w*|шеш\w*|теңдеу\w*|пайыз\w*)\b",
    FLAGS,
)
QUANTITY_QUESTION = re.compile(
    r"\b(how many|how much|total|in all|left|remain\w*|сколько|осталось|всего|"
    r"неше|қанша|қалды)\b",
    FLAGS,
)
LOGIC_HINTS = re.compile(
    r"\b(taller|shorter|older|younger|heavier|faster|tallest|shortest|oldest|youngest|"
    r"выше|ниже|старше|младше|самый (?:высокий|низкий|старший|младший)|"
    r"кто самый|кто выше|кто старше|биік|аласа|ең (?:аласа|биік|үлкен|кіші)|"
    r"all \w+ are|every \w+ is|все \w+ являются|барлық \w+|yes or no|да или нет)\b",
    FLAGS,
)
TRANSLATION_HINTS = re.compile(
    r"\b(translate|translation|how do you say|переведи\w*|перевод\w*|как будет|"
    r"аудар\w*|қалай айтылады)\b|in (?:russian|english|kazakh)\b|на (?:русский|английский|казахский)"
    r"|(?:орыс|ағылшын|қазақ) тіліне",
    FLAGS,
)
ANALYSIS_HINTS = re.compile(
    r"\b(summari[sz]e|summary|analy[sz]e|extract|how many (?:words|sentences)|count the words|"
    r"основн\w* мысл\w*|кратко перескажи|перескажи|проанализируй|выдели|"
    r"сколько (?:слов|предложений)|неше (?:сөз|сөйлем)|талда|қысқаша айтып бер)\b|"
    r"\b(?:text|текст|мәтін)\s*:",
    FLAGS,
)
DIALOGUE_HINTS = re.compile(
    r"^\W*(hi|hello|hey|good (?:morning|evening|night)|thanks?|thank you|bye|goodbye|see you|"
    r"привет\w*|здравствуй\w*|добр\w+ (?:день|утро|вечер)|спасибо|благодарю|пока|до свидания|"
    r"сәлем\w*|рахмет|сау бол\w*|қайырлы \w+)\b|"
    r"\b(who are you|your name|what can you do|what are you|about yourself|кто ты|как тебя зовут|"
    r"что ты умеешь|что ты можешь|расскажи о себе|сен кімсің|атың кім|не істей аласың)\b",
    FLAGS,
)
QUESTION_HINTS = re.compile(
    r"\?|^\W*(what|who|why|how|when|where|which|explain|describe|define|что|кто|почему|как|"
    r"когда|где|какой|объясни\w*|расскажи|опиши|не|неге|қашан|қайда|қандай|түсіндір\w*)\b",
    FLAGS,
)


def classify_intent(question: str) -> str:
    if CODE_HINTS.search(question):
        return CODE
    if TRANSLATION_HINTS.search(question):
        return TRANSLATION
    if LOGIC_HINTS.search(question):
        return LOGIC
    if MATH_HINTS.search(question):
        return MATH
    if re.search(r"\d", question) and QUANTITY_QUESTION.search(question):
        return MATH
    if ANALYSIS_HINTS.search(question):
        return ANALYSIS
    if DIALOGUE_HINTS.search(question):
        return DIALOGUE
    if QUESTION_HINTS.search(question):
        return INFORMATION
    return DIALOGUE


def classify_task(question: str) -> str:
    intent = classify_intent(question)
    return intent if intent in VERIFIED_KINDS else GENERAL
