from __future__ import annotations

LANGUAGES = ("en", "ru", "kk")
TRAIN_FRAMES = 4
SPECIAL_TRAIN_FRAMES = 3
CAUSE_SPLITS = {"train": (0, 1, 2, 3), "dev": (4, 5), "test": (6, 7)}
PLACE, PERSON, NUMBER = "place", "person", "number"
TRAIN_RELATIONS = ("river", "mountain", "lake", "founder", "mayor", "year")
HELD_OUT_RELATIONS = ("bridge", "festival", "headmaster")
KINDS = {
    "river": PLACE,
    "mountain": PLACE,
    "lake": PLACE,
    "bridge": PLACE,
    "festival": PLACE,
    "founder": PERSON,
    "mayor": PERSON,
    "headmaster": PERSON,
    "year": NUMBER,
}
NOUNS: dict[str, dict[str, str]] = {
    "en": {
        "river": "river",
        "mountain": "mountain",
        "lake": "lake",
        "founder": "founder",
        "mayor": "mayor",
        "year": "founding year",
        "bridge": "bridge",
        "festival": "festival",
        "headmaster": "headmaster",
    },
    "ru": {
        "river": "река",
        "mountain": "гора",
        "lake": "озеро",
        "founder": "основатель",
        "mayor": "мэр",
        "year": "год основания",
        "bridge": "мост",
        "festival": "фестиваль",
        "headmaster": "директор школы",
    },
    "kk": {
        "river": "өзен",
        "mountain": "тау",
        "lake": "көл",
        "founder": "негізін қалаушы",
        "mayor": "әкім",
        "year": "құрылған жылы",
        "bridge": "көпір",
        "festival": "мейрам",
        "headmaster": "мектеп директоры",
    },
}
STATEMENTS: dict[str, tuple[str, ...]] = {
    "en": (
        "The {noun} of {E} is {X}.",
        "{E}: {noun} — {X}.",
        "As for {E}, its {noun} is {X}.",
        "Records say that {X} is the {noun} of {E}.",
        "Entry: {E}. Field: {noun}. Value: {X}.",
        "{X} is listed as the {noun} for {E} in the archive.",
    ),
    "ru": (
        "У {E} {noun} — {X}.",
        "{E}: {noun} — {X}.",
        "Что касается {E}, то его «{noun}» — {X}.",
        "По записям, {X} — это «{noun}» для {E}.",
        "Запись: {E}. Поле: {noun}. Значение: {X}.",
        "В архиве для {E} в качестве «{noun}» указано: {X}.",
    ),
    "kk": (
        "{E} үшін {noun} — {X}.",
        "{E}: {noun} — {X}.",
        "{E} туралы айтсақ, оның «{noun}» мәні — {X}.",
        "Жазбаларға сәйкес, {X} — {E} үшін «{noun}».",
        "Жазба: {E}. Өріс: {noun}. Мән: {X}.",
        "Мұрағатта {E} үшін «{noun}» ретінде {X} көрсетілген.",
    ),
}
QUESTIONS: dict[str, tuple[str, ...]] = {
    "en": (
        "What is the {noun} of {E}?",
        "Tell me the {noun} for {E}.",
        "According to the text, what is the {noun} of {E}?",
        "{E}: {noun}?",
        "Can you find the {noun} that belongs to {E} in the passage?",
        "Look at {E} and report its {noun}.",
    ),
    "ru": (
        "Какое значение «{noun}» у {E}?",
        "Назови «{noun}» для {E}.",
        "Согласно тексту, что является «{noun}» у {E}?",
        "{E}: {noun}?",
        "Можешь найти в отрывке «{noun}», относящееся к {E}?",
        "Посмотри на {E} и сообщи его «{noun}».",
    ),
    "kk": (
        "{E} үшін «{noun}» қандай?",
        "{E} үшін «{noun}» мәнін ата.",
        "Мәтін бойынша {E} үшін «{noun}» не?",
        "{E}: {noun}?",
        "Үзіндіден {E} үшін «{noun}» мәнін таба аласың ба?",
        "{E} жазбасын қарап, оның «{noun}» мәнін хабарла.",
    ),
}
REVERSE_QUESTIONS: dict[str, tuple[str, ...]] = {
    "en": (
        "Which settlement has the {noun} {X}?",
        "Whose {noun} is {X}?",
        "Find the settlement whose {noun} is {X}.",
        "{X} is the {noun} of which settlement?",
    ),
    "ru": (
        "У какого поселения «{noun}» — {X}?",
        "Чьё «{noun}» — {X}?",
        "Найди поселение, у которого «{noun}» — {X}.",
        "{X} — это «{noun}» какого поселения?",
    ),
    "kk": (
        "«{noun}» — {X} болатын қоныс қайсы?",
        "Қай қоныстың «{noun}» мәні — {X}?",
        "«{noun}» мәні {X} болатын қонысты тап.",
        "{X} — қай қоныстың «{noun}» мәні?",
    ),
}
REVERSE_ANSWERS: dict[str, tuple[str, ...]] = {
    "en": ("It is {E}.", "{E}.", "The settlement is {E}."),
    "ru": ("Это {E}.", "{E}.", "Это поселение — {E}."),
    "kk": ("Бұл — {E}.", "{E}.", "Бұл қоныс — {E}."),
}
ALTERNATIVES = {
    "en": "Is the {noun} of {E} {A} or {B}?",
    "ru": "«{noun}» у {E} — {A} или {B}?",
    "kk": "{E} үшін «{noun}» — {A} немесе {B}?",
}
NEGATIONS: dict[str, tuple[str, ...]] = {
    "en": (
        "The {noun} of {E} is not {W}. It is {X}.",
        "Contrary to some claims, {E} has {X} as its {noun}, not {W}.",
        "{E}: {noun} — not {W}, but {X}.",
        "People say {W}, but the {noun} of {E} is actually {X}.",
    ),
    "ru": (
        "«{noun}» у {E} — не {W}, а {X}.",
        "Вопреки слухам, у {E} «{noun}» — {X}, а не {W}.",
        "{E}: {noun} — не {W}, но {X}.",
        "Говорят, {W}, но на самом деле «{noun}» у {E} — {X}.",
    ),
    "kk": (
        "{E} үшін «{noun}» — {W} емес, {X}.",
        "Қауесетке қарамастан, {E} үшін «{noun}» — {W} емес, {X}.",
        "{E}: {noun} — {W} емес, бірақ {X}.",
        "{W} дейді, бірақ {E} үшін «{noun}» шын мәнінде — {X}.",
    ),
}
UPDATES: dict[str, tuple[str, ...]] = {
    "en": (
        "Update: the {noun} of {E} is now {X}.",
        "Later it was corrected: {E} has {X} as its {noun}.",
        "Correction — {noun} of {E}: {X}.",
        "The earlier entry about {E} was wrong; its {noun} is {X}.",
    ),
    "ru": (
        "Обновление: «{noun}» у {E} теперь — {X}.",
        "Позже уточнили: у {E} «{noun}» — {X}.",
        "Поправка — {noun} у {E}: {X}.",
        "Прежняя запись о {E} была неверной; его «{noun}» — {X}.",
    ),
    "kk": (
        "Жаңарту: {E} үшін «{noun}» енді — {X}.",
        "Кейін нақтылады: {E} үшін «{noun}» — {X}.",
        "Түзету — {E} үшін {noun}: {X}.",
        "{E} туралы бұрынғы жазба қате болған; оның «{noun}» мәні — {X}.",
    ),
}
SOURCE_CLAIMS = {
    "en": "According to {S}, the {noun} of {E} is {V}.",
    "ru": "Согласно {S}, «{noun}» у {E} — {V}.",
    "kk": "{S} бойынша, {E} үшін «{noun}» — {V}.",
}
SOURCES: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "en": (
        (
            "the old chronicle",
            "the old chronicle",
            "the official register",
            "the official register",
        ),
        ("a travel leaflet", "a travel leaflet", "the census", "the census"),
        ("a rumour", "a rumour", "the court record", "the court record"),
    ),
    "ru": (
        ("старой хронике", "Старая хроника", "официальному реестру", "Официальный реестр"),
        ("туристическому буклету", "Туристический буклет", "переписи", "Перепись"),
        ("слуху", "Слух", "судебной записи", "Судебная запись"),
    ),
    "kk": (
        ("ескі шежіре", "Ескі шежіре", "ресми тізілім", "Ресми тізілім"),
        ("туристік буклет", "Туристік буклет", "санақ", "Санақ"),
        ("қауесет", "Қауесет", "сот жазбасы", "Сот жазбасы"),
    ),
}
PRIORITY_STATEMENTS: dict[str, tuple[str, ...]] = {
    "en": (
        "{High} is more reliable than {Low}.",
        "Trust {High}, not {Low}.",
        "Only {High} should be believed over {Low}.",
    ),
    "ru": (
        "{High} надёжнее, чем {Low}.",
        "Верить следует: {High}, а не {Low}.",
        "Из двух источников верен только {High}, а не {Low}.",
    ),
    "kk": (
        "Сенімді дерек — {High}, {Low} емес.",
        "Сену керек: {High}, {Low} емес.",
        "Екі дерекнің ішінде тек {High} дұрыс, {Low} емес.",
    ),
}
CAUSE_STATEMENTS: dict[str, tuple[str, ...]] = {
    "en": (
        "In {E} the market closed. The reason: {C}.",
        "The market of {E} was shut because of {C}.",
        "{E}: market closed; cause — {C}.",
        "{C} is why the market in {E} stopped trading.",
    ),
    "ru": (
        "В {E} рынок закрылся. Причина: {C}.",
        "Рынок в {E} закрыли из-за такой причины: {C}.",
        "{E}: рынок закрыт; причина — {C}.",
        "Из-за такой причины в {E} рынок перестал работать: {C}.",
    ),
    "kk": (
        "{E} қонысында базар жабылды. Себебі: {C}.",
        "{E} қонысындағы базар жабылды, себеп — {C}.",
        "{E}: базар жабық; себебі — {C}.",
        "{E} қонысында базардың тоқтауының себебі — {C}.",
    ),
}
CAUSE_QUESTIONS: dict[str, tuple[str, ...]] = {
    "en": (
        "Why did the market close in {E}?",
        "What caused the market in {E} to close?",
        "State the reason the {E} market stopped.",
    ),
    "ru": (
        "Почему в {E} закрылся рынок?",
        "Что стало причиной закрытия рынка в {E}?",
        "Назови причину, по которой в {E} остановился рынок.",
    ),
    "kk": (
        "{E} қонысында базар неліктен жабылды?",
        "{E} қонысында базардың жабылуына не себеп болды?",
        "{E} қонысындағы базар тоқтауының себебін ата.",
    ),
}
CAUSE_ANSWERS: dict[str, tuple[str, ...]] = {
    "en": ("The market in {E} closed because of {C}.", "Because of {C}.", "{C}."),
    "ru": ("Рынок в {E} закрылся из-за такой причины: {C}.", "Причина: {C}.", "{C}."),
    "kk": ("{E} қонысындағы базар жабылды, себебі: {C}.", "Себебі: {C}.", "{C}."),
}
CAUSES: dict[str, tuple[tuple[str, str], ...]] = {
    "en": (
        ("heavy snow", "snow"),
        ("a flood", "flood"),
        ("a power failure", "power"),
        ("a fire", "fire"),
        ("a strike", "strike"),
        ("an epidemic", "epidemic"),
        ("a bridge collapse", "collapse"),
        ("a storm", "storm"),
    ),
    "ru": (
        ("сильный снегопад", "снегопад"),
        ("наводнение", "наводнен"),
        ("отключение электричества", "электричеств"),
        ("пожар", "пожар"),
        ("забастовка", "забастовк"),
        ("эпидемия", "эпидеми"),
        ("обрушение моста", "обрушени"),
        ("шторм", "шторм"),
    ),
    "kk": (
        ("қатты қар", "қар"),
        ("су тасқыны", "тасқын"),
        ("электр жарығының өшуі", "электр"),
        ("өрт", "өрт"),
        ("ереуіл", "ереуіл"),
        ("індет", "індет"),
        ("көпірдің құлауы", "құлау"),
        ("дауыл", "дауыл"),
    ),
}
ANSWERS: dict[str, tuple[str, ...]] = {
    "en": (
        "The {noun} of {E} is {X}.",
        "{X}.",
        "It is {X}.",
        "According to the text, it is {X}.",
    ),
    "ru": (
        "У {E} {noun} — {X}.",
        "{X}.",
        "Это {X}.",
        "Согласно тексту, это {X}.",
    ),
    "kk": (
        "{E} үшін {noun} — {X}.",
        "{X}.",
        "Бұл — {X}.",
        "Мәтін бойынша, бұл — {X}.",
    ),
}
MISSING: dict[str, tuple[str, ...]] = {
    "en": (
        "The text does not say.",
        "The text does not mention the {noun} of {E}.",
        "The document does not say what the {noun} of {E} is.",
        "The passage does not mention this.",
    ),
    "ru": (
        "В тексте об этом не говорится.",
        "В тексте не говорится, каково «{noun}» у {E}.",
        "В тексте не говорится про «{noun}» у {E}.",
        "В тексте об этом не говорится, поэтому ответить нельзя.",
    ),
    "kk": (
        "Мәтінде бұл туралы айтылмаған.",
        "Мәтінде бұл туралы айтылмаған: {E} үшін «{noun}» көрсетілмеген.",
        "Мәтінде бұл туралы айтылмаған, сондықтан жауап бере алмаймын.",
        "Мәтінде бұл туралы айтылмаған, «{noun}» белгісіз.",
    ),
}
GENERIC_MISSING = (0, 3)
FILLERS: dict[str, tuple[str, ...]] = {
    "en": (
        "The weather was mild that week.",
        "Merchants arrive early in the morning.",
        "The road to the north is long and dusty.",
        "Children play in the square after lessons.",
        "A small bakery opens before sunrise.",
        "The old gate has been repainted twice.",
    ),
    "ru": (
        "Погода в ту неделю была мягкой.",
        "Торговцы приезжают рано утром.",
        "Дорога на север длинная и пыльная.",
        "Дети играют на площади после уроков.",
        "Маленькая пекарня открывается до рассвета.",
        "Старые ворота красили уже дважды.",
    ),
    "kk": (
        "Ол аптада ауа райы жұмсақ болды.",
        "Саудагерлер таңертең ерте келеді.",
        "Солтүстікке баратын жол ұзақ әрі шаңды.",
        "Балалар сабақтан кейін алаңда ойнайды.",
        "Шағын наубайхана күн шығардан бұрын ашылады.",
        "Ескі қақпа екі рет бояғалы тұр.",
    ),
}
CONSONANTS = "bdgklmnprstvz"
VOWELS = "aeiou"
SUFFIXES = ("a", "en", "is", "ar", "on", "ith", "ova")
