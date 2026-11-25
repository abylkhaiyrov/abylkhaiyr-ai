from __future__ import annotations

Tables = dict[str, tuple[str, ...]]

INSTRUCTIONS_TRAIN: Tables = {
    "en": (
        "Repeat exactly: {X}",
        "Copy this: {X}",
        "Say it back: {X}",
        "Write the following exactly as given: {X}",
        "Please repeat: {X}",
        "Echo: {X}",
        "Reproduce this text: {X}",
        "Just repeat {X}",
        "Type exactly this: {X}",
        "Output the same: {X}",
        "Repeat after me: {X}",
        "Copy and paste: {X}",
        "Give back this text unchanged: {X}",
        "Retype: {X}",
    ),
    "ru": (
        "Повтори точно: {X}",
        "Скопируй это: {X}",
        "Скажи то же самое: {X}",
        "Запиши в точности как дано: {X}",
        "Пожалуйста, повтори: {X}",
        "Эхо: {X}",
        "Воспроизведи текст: {X}",
        "Просто повтори {X}",
        "Напиши в точности так: {X}",
        "Выведи то же самое: {X}",
        "Повторяй за мной: {X}",
        "Скопируй и вставь: {X}",
        "Верни этот текст без изменений: {X}",
        "Перепечатай: {X}",
    ),
    "kk": (
        "Дәл қайтала: {X}",
        "Мынаны көшіріп жаз: {X}",
        "Дәл солай айт: {X}",
        "Берілгендей дәл жаз: {X}",
        "Өтінемін, қайтала: {X}",
        "Жаңғырық: {X}",
        "Мәтінді қайта шығар: {X}",
        "Тек қайтала {X}",
        "Дәл осылай теріп жаз: {X}",
        "Сол мәтінді шығар: {X}",
        "Менен кейін қайтала: {X}",
        "Көшіріп қой: {X}",
        "Бұл мәтінді өзгертпей қайтар: {X}",
        "Қайта басып шығар: {X}",
    ),
}
INSTRUCTIONS_DEV: Tables = {
    "en": ("Return this string as it is: {X}", "Duplicate: {X}", "Respond with the same text: {X}"),
    "ru": ("Верни строку как есть: {X}", "Продублируй: {X}", "Ответь тем же текстом: {X}"),
    "kk": (
        "Жолды сол күйінде қайтар: {X}",
        "Көшірмесін жаса: {X}",
        "Дәл сол мәтінмен жауап бер: {X}",
    ),
}
INSTRUCTIONS_HOLDOUT: Tables = {
    "en": ("Mirror this exactly: {X}", "Replicate the following: {X}", "Send back: {X}"),
    "ru": ("Отзеркаль в точности: {X}", "Воспроизведи следующее: {X}", "Отправь обратно: {X}"),
    "kk": ("Дәл айна сияқты қайтала: {X}", "Мынаны қайта жаса: {X}", "Кері жібер: {X}"),
}
FRAMES_TRAIN: Tables = {
    "en": (
        "The {noun} is called {X}.",
        "{X} is the name of the {noun}.",
        "People call the {noun} {X}.",
        "Our {noun} bears the name {X}.",
        "Records list {X} as the {noun}.",
        "The name of the {noun} is {X}.",
        "Here the {noun} goes by {X}.",
        "We know the {noun} as {X}.",
    ),
    "ru": (
        "{Noun} называется {X}.",
        "{X} — так называется {noun}.",
        "Название: {X}. Это «{noun}».",
        "Здесь есть «{noun}»: {X}.",
        "Имя для «{noun}»: {X}.",
        "Согласно записям, «{noun}» — {X}.",
        "«{noun}»: {X}.",
        "Для «{noun}» указано название {X}.",
    ),
    "kk": (
        "«{noun}» — {X}.",
        "{X} — осы «{noun}» атауы.",
        "Атауы: {X}. Бұл — «{noun}».",
        "Мұнда «{noun}» бар: {X}.",
        "«{noun}» үшін атау: {X}.",
        "Жазбаларға сәйкес, «{noun}» — {X}.",
        "«{noun}»: {X}.",
        "«{noun}» үшін {X} атауы көрсетілген.",
    ),
}
FRAMES_DEV: Tables = {
    "en": ("The {noun} has the name {X}.", "{X}: that is how the {noun} is named."),
    "ru": ("У «{noun}» есть название {X}.", "{X}: именно так зовут «{noun}»."),
    "kk": ("«{noun}» атауы — {X}.", "{X}: дәл осылай «{noun}» аталады."),
}
FRAMES_HOLDOUT: Tables = {
    "en": (
        "Locals refer to the {noun} as {X}.",
        "Known as {X}, the {noun} lies nearby.",
    ),
    "ru": (
        "Местные зовут «{noun}» {X}.",
        "{X} — вот как называют «{noun}» местные.",
    ),
    "kk": (
        "Жергілікті халық «{noun}» дегенді {X} дейді.",
        "{X} — «{noun}» осылай аталады.",
    ),
}
QUESTIONS_TRAIN: Tables = {
    "en": (
        "Name the {noun}.",
        "What is the {noun} called?",
        "Which name does the {noun} have?",
        "Tell me the name of the {noun}.",
        "What name is given to the {noun}?",
        "Give the name of the {noun}.",
        "How is the {noun} named?",
        "State the name of the {noun}.",
    ),
    "ru": (
        "Назови «{noun}».",
        "Как называется «{noun}»?",
        "Какое название у «{noun}»?",
        "Скажи название «{noun}».",
        "Какое имя дано «{noun}»?",
        "Дай название «{noun}».",
        "Как зовут «{noun}»?",
        "Укажи название «{noun}».",
    ),
    "kk": (
        "«{noun}» қалай аталады?",
        "«{noun}» атауын ата.",
        "«{noun}» үшін атау қандай?",
        "«{noun}» атауын айт.",
        "«{noun}» қандай атпен аталады?",
        "«{noun}» үшін атауды жаз.",
        "«{noun}» атауы қандай?",
        "«{noun}» атауын көрсет.",
    ),
}
QUESTIONS_DEV: Tables = {
    "en": ("What do people call the {noun}?", "Please say how the {noun} is called."),
    "ru": ("Как люди называют «{noun}»?", "Скажи, пожалуйста, как называется «{noun}»."),
    "kk": ("Адамдар «{noun}» дегенді қалай атайды?", "«{noun}» қалай аталатынын айтып бер."),
}
QUESTIONS_HOLDOUT: Tables = {
    "en": ("I need the name of the {noun}.", "Could you tell me what the {noun} is called?"),
    "ru": ("Мне нужно название «{noun}».", "Не подскажешь название «{noun}»?"),
    "kk": ("Маған «{noun}» атауы керек.", "«{noun}» атауын айта аласыз ба?"),
}
ANSWERS: Tables = {
    "en": ("The {noun} is called {X}.", "It is {X}.", "{X}.", "The name is {X}."),
    "ru": ("«{noun}» называется {X}.", "Это {X}.", "{X}.", "Название — {X}."),
    "kk": ("«{noun}» — {X}.", "Бұл — {X}.", "{X}.", "Атауы — {X}."),
}
NOUNS: Tables = {
    "en": (
        "river",
        "mountain",
        "city",
        "teacher",
        "town",
        "lake",
        "book",
        "island",
        "bridge",
        "doctor",
    ),
    "ru": (
        "река",
        "гора",
        "город",
        "учитель",
        "посёлок",
        "озеро",
        "книга",
        "остров",
        "мост",
        "врач",
    ),
    "kk": (
        "өзен",
        "тау",
        "қала",
        "мұғалім",
        "елді мекен",
        "көл",
        "кітап",
        "арал",
        "көпір",
        "дәрігер",
    ),
}
LABELS = {"en": ("Text", "Question"), "ru": ("Текст", "Вопрос"), "kk": ("Мәтін", "Сұрақ")}
LAYOUTS = ("context_first", "question_first", "labeled", "text_inline")
