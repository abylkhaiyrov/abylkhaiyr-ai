from __future__ import annotations

from dataclasses import dataclass

LANGUAGES = ("ru", "en", "kk")
LANGUAGE_NAMES = {
    "ru": {"ru": "русский", "en": "английский", "kk": "казахский"},
    "en": {"ru": "Russian", "en": "English", "kk": "Kazakh"},
    "kk": {"ru": "орыс тілі", "en": "ағылшын тілі", "kk": "қазақ тілі"},
}


@dataclass(frozen=True)
class Term:
    en: str
    ru: str
    kk: str

    def in_language(self, language: str) -> str:
        return getattr(self, language)


NOUNS = [
    Term("book", "книга", "кітап"),
    Term("table", "стол", "үстел"),
    Term("house", "дом", "үй"),
    Term("water", "вода", "су"),
    Term("sun", "солнце", "күн"),
    Term("moon", "луна", "ай"),
    Term("tree", "дерево", "ағаш"),
    Term("flower", "цветок", "гүл"),
    Term("cat", "кошка", "мысық"),
    Term("dog", "собака", "ит"),
    Term("horse", "лошадь", "жылқы"),
    Term("bird", "птица", "құс"),
    Term("fish", "рыба", "балық"),
    Term("apple", "яблоко", "алма"),
    Term("bread", "хлеб", "нан"),
    Term("milk", "молоко", "сүт"),
    Term("school", "школа", "мектеп"),
    Term("teacher", "учитель", "мұғалім"),
    Term("friend", "друг", "дос"),
    Term("mother", "мама", "ана"),
    Term("father", "папа", "әке"),
    Term("city", "город", "қала"),
    Term("river", "река", "өзен"),
    Term("mountain", "гора", "тау"),
    Term("road", "дорога", "жол"),
    Term("window", "окно", "терезе"),
    Term("door", "дверь", "есік"),
    Term("car", "машина", "көлік"),
    Term("train", "поезд", "пойыз"),
    Term("pen", "ручка", "қалам"),
    Term("paper", "бумага", "қағаз"),
    Term("sky", "небо", "аспан"),
    Term("star", "звезда", "жұлдыз"),
    Term("rain", "дождь", "жаңбыр"),
    Term("snow", "снег", "қар"),
    Term("wind", "ветер", "жел"),
    Term("fire", "огонь", "от"),
    Term("stone", "камень", "тас"),
    Term("tea", "чай", "шай"),
    Term("bridge", "мост", "көпір"),
    Term("garden", "сад", "бақ"),
    Term("market", "рынок", "базар"),
    Term("hand", "рука", "қол"),
    Term("eye", "глаз", "көз"),
    Term("heart", "сердце", "жүрек"),
    Term("song", "песня", "ән"),
    Term("night", "ночь", "түн"),
    Term("morning", "утро", "таң"),
]

COLORS = [
    Term("red", "красный", "қызыл"),
    Term("blue", "синий", "көк"),
    Term("green", "зелёный", "жасыл"),
    Term("yellow", "жёлтый", "сары"),
    Term("white", "белый", "ақ"),
    Term("black", "чёрный", "қара"),
]

SEASONS = [
    Term("winter", "зима", "қыс"),
    Term("spring", "весна", "көктем"),
    Term("summer", "лето", "жаз"),
    Term("autumn", "осень", "күз"),
]

ANIMALS = [
    Term("cat", "кошка", "мысық"),
    Term("dog", "собака", "ит"),
    Term("horse", "лошадь", "жылқы"),
    Term("bird", "птица", "құс"),
    Term("fish", "рыба", "балық"),
    Term("camel", "верблюд", "түйе"),
    Term("sheep", "овца", "қой"),
    Term("wolf", "волк", "қасқыр"),
]

NUMBERS = [
    Term("one", "один", "бір"),
    Term("two", "два", "екі"),
    Term("three", "три", "үш"),
    Term("four", "четыре", "төрт"),
    Term("five", "пять", "бес"),
    Term("six", "шесть", "алты"),
    Term("seven", "семь", "жеті"),
    Term("eight", "восемь", "сегіз"),
    Term("nine", "девять", "тоғыз"),
    Term("ten", "десять", "он"),
]

PHRASES = [
    Term("hello", "привет", "сәлем"),
    Term("thank you", "спасибо", "рахмет"),
    Term("goodbye", "до свидания", "сау болыңыз"),
    Term("good morning", "доброе утро", "қайырлы таң"),
    Term("good night", "спокойной ночи", "қайырлы түн"),
    Term("please", "пожалуйста", "өтінемін"),
    Term("sorry", "извините", "кешіріңіз"),
    Term("yes", "да", "иә"),
    Term("no", "нет", "жоқ"),
    Term("how are you", "как дела", "қалың қалай"),
    Term("welcome", "добро пожаловать", "қош келдіңіз"),
    Term("good luck", "удачи", "сәттілік тілеймін"),
]

SENTENCE_FRAMES = {
    "en": "This is a {word}.",
    "ru": "Это {word}.",
    "kk": "Бұл — {word}.",
}


@dataclass(frozen=True)
class Concept:
    name: Term
    definition: Term


CONCEPTS = [
    Concept(
        Term("photosynthesis", "фотосинтез", "фотосинтез"),
        Term(
            "Photosynthesis is the process by which plants use sunlight, water and carbon "
            "dioxide to make sugar and release oxygen.",
            "Фотосинтез — это процесс, при котором растения используют солнечный свет, воду "
            "и углекислый газ, чтобы создавать сахар и выделять кислород.",
            "Фотосинтез — өсімдіктердің күн сәулесін, суды және көмірқышқыл газын пайдаланып, "
            "қант түзіп, оттегін бөлетін үдерісі.",
        ),
    ),
    Concept(
        Term("gravity", "гравитация", "гравитация"),
        Term(
            "Gravity is the force that pulls objects toward each other; it keeps us on the "
            "ground and the planets in orbit around the Sun.",
            "Гравитация — это сила, с которой тела притягиваются друг к другу; она удерживает "
            "нас на земле, а планеты — на орбитах вокруг Солнца.",
            "Гравитация — денелердің бір-біріне тартылу күші; ол бізді жерде, ал планеталарды "
            "Күнді айнала орбитада ұстап тұрады.",
        ),
    ),
    Concept(
        Term("an algorithm", "алгоритм", "алгоритм"),
        Term(
            "An algorithm is a clear sequence of steps that solves a problem or completes a task.",
            "Алгоритм — это чёткая последовательность шагов, которая решает задачу или "
            "помогает выполнить какое-либо действие.",
            "Алгоритм — есепті шешуге немесе тапсырманы орындауға арналған нақты қадамдар тізбегі.",
        ),
    ),
    Concept(
        Term("an atom", "атом", "атом"),
        Term(
            "An atom is the smallest unit of a chemical element; it consists of a nucleus made "
            "of protons and neutrons and electrons around it.",
            "Атом — это мельчайшая частица химического элемента; он состоит из ядра с "
            "протонами и нейтронами и электронов вокруг него.",
            "Атом — химиялық элементтің ең кішкентай бөлшегі; ол протондар мен "
            "нейтрондардан тұратын ядродан және оның айналасындағы электрондардан тұрады.",
        ),
    ),
    Concept(
        Term("a prime number", "простое число", "жай сан"),
        Term(
            "A prime number is a natural number greater than 1 that has no divisors other than "
            "1 and itself, for example 2, 3, 5 and 7.",
            "Простое число — это натуральное число больше 1, которое делится только на 1 и на "
            "само себя, например 2, 3, 5 и 7.",
            "Жай сан — тек 1-ге және өзіне бөлінетін, 1-ден үлкен натурал сан, мысалы 2, 3, 5 "
            "және 7.",
        ),
    ),
    Concept(
        Term("a neural network", "нейронная сеть", "нейрондық желі"),
        Term(
            "A neural network is a computer model made of connected layers of simple units "
            "that learns patterns from data.",
            "Нейронная сеть — это компьютерная модель из связанных слоёв простых элементов, "
            "которая учится находить закономерности в данных.",
            "Нейрондық желі — қарапайым элементтердің өзара байланысқан қабаттарынан тұратын, "
            "деректерден заңдылықтарды үйренетін компьютерлік модель.",
        ),
    ),
    Concept(
        Term("the Internet", "интернет", "интернет"),
        Term(
            "The Internet is a global network of computers that exchange information using "
            "shared rules called protocols.",
            "Интернет — это глобальная сеть компьютеров, которые обмениваются информацией по "
            "общим правилам, называемым протоколами.",
            "Интернет — протокол деп аталатын ортақ ережелер арқылы ақпарат алмасатын "
            "компьютерлердің жаһандық желісі.",
        ),
    ),
    Concept(
        Term("electricity", "электричество", "электр тогы"),
        Term(
            "Electricity is the flow of electric charge, usually electrons moving through a wire.",
            "Электричество — это движение электрических зарядов, обычно электронов по проводу.",
            "Электр тогы — электр зарядтарының, әдетте сым бойымен қозғалатын электрондардың "
            "ағыны.",
        ),
    ),
    Concept(
        Term("a volcano", "вулкан", "жанартау"),
        Term(
            "A volcano is an opening in the Earth's crust through which hot lava, ash and "
            "gases can escape.",
            "Вулкан — это отверстие в земной коре, через которое могут выходить раскалённая "
            "лава, пепел и газы.",
            "Жанартау — жер қыртысындағы, одан қайнаған лава, күл және газдар шығатын тесік.",
        ),
    ),
    Concept(
        Term("the water cycle", "круговорот воды", "судың айналымы"),
        Term(
            "The water cycle is the continuous movement of water: it evaporates, forms clouds, "
            "falls as rain or snow and returns to rivers and seas.",
            "Круговорот воды — это непрерывное движение воды: она испаряется, образует "
            "облака, выпадает дождём или снегом и возвращается в реки и моря.",
            "Судың айналымы — судың үздіксіз қозғалысы: ол буланады, бұлт түзеді, жаңбыр "
            "немесе қар болып жауады да, өзендер мен теңіздерге қайта оралады.",
        ),
    ),
    Concept(
        Term("a variable", "переменная", "айнымалы"),
        Term(
            "A variable in programming is a named place in memory that stores a value which "
            "can change.",
            "Переменная в программировании — это именованная область памяти, в которой "
            "хранится значение, способное меняться.",
            "Бағдарламалаудағы айнымалы — өзгеруі мүмкін мәнді сақтайтын жадтағы аты бар орын.",
        ),
    ),
    Concept(
        Term("inflation", "инфляция", "инфляция"),
        Term(
            "Inflation is a general rise in prices over time, which reduces how much money "
            "can buy.",
            "Инфляция — это общий рост цен со временем, из-за которого на те же деньги можно "
            "купить меньше.",
            "Инфляция — уақыт өте баға деңгейінің жалпы өсуі, соның салдарынан сол ақшаға "
            "азырақ зат сатып алуға болады.",
        ),
    ),
]


@dataclass(frozen=True)
class Fact:
    question: Term
    answer: Term


FACTS = [
    Fact(
        Term(
            "What is the capital of Kazakhstan?",
            "Какая столица Казахстана?",
            "Қазақстанның астанасы қай қала?",
        ),
        Term(
            "The capital of Kazakhstan is Astana.",
            "Столица Казахстана — Астана.",
            "Қазақстанның астанасы — Астана.",
        ),
    ),
    Fact(
        Term("How many days are in a week?", "Сколько дней в неделе?", "Бір аптада неше күн бар?"),
        Term("There are seven days in a week.", "В неделе семь дней.", "Бір аптада жеті күн бар."),
    ),
    Fact(
        Term(
            "At what temperature does water boil at sea level?",
            "При какой температуре кипит вода на уровне моря?",
            "Теңіз деңгейінде су неше градуста қайнайды?",
        ),
        Term(
            "Water boils at 100 degrees Celsius at sea level.",
            "Вода кипит при 100 градусах Цельсия на уровне моря.",
            "Теңіз деңгейінде су 100 градус Цельсийде қайнайды.",
        ),
    ),
    Fact(
        Term(
            "Which planet is the largest in the Solar System?",
            "Какая планета самая большая в Солнечной системе?",
            "Күн жүйесіндегі ең үлкен планета қайсы?",
        ),
        Term(
            "Jupiter is the largest planet in the Solar System.",
            "Юпитер — самая большая планета Солнечной системы.",
            "Күн жүйесіндегі ең үлкен планета — Юпитер.",
        ),
    ),
    Fact(
        Term("How many months are in a year?", "Сколько месяцев в году?", "Бір жылда неше ай бар?"),
        Term(
            "There are twelve months in a year.",
            "В году двенадцать месяцев.",
            "Бір жылда он екі ай бар.",
        ),
    ),
    Fact(
        Term(
            "Who wrote the Book of Words (Qara sozder)?",
            "Кто написал «Слова назидания»?",
            "«Қара сөздер» кітабын кім жазған?",
        ),
        Term(
            "Abai Kunanbayev wrote the Book of Words.",
            "«Слова назидания» написал Абай Кунанбаев.",
            "«Қара сөздер» кітабын Абай Құнанбайұлы жазған.",
        ),
    ),
    Fact(
        Term(
            "What is the chemical formula of water?",
            "Какая химическая формула воды?",
            "Судың химиялық формуласы қандай?",
        ),
        Term(
            "The chemical formula of water is H2O.",
            "Химическая формула воды — H2O.",
            "Судың химиялық формуласы — H2O.",
        ),
    ),
    Fact(
        Term(
            "How many sides does a triangle have?",
            "Сколько сторон у треугольника?",
            "Үшбұрыштың неше қабырғасы бар?",
        ),
        Term(
            "A triangle has three sides.",
            "У треугольника три стороны.",
            "Үшбұрыштың үш қабырғасы бар.",
        ),
    ),
    Fact(
        Term(
            "Which ocean is the largest?",
            "Какой океан самый большой?",
            "Ең үлкен мұхит қайсы?",
        ),
        Term(
            "The Pacific Ocean is the largest ocean.",
            "Тихий океан — самый большой океан.",
            "Ең үлкен мұхит — Тынық мұхиты.",
        ),
    ),
    Fact(
        Term(
            "Is the Sun a planet or a star?",
            "Солнце — это планета или звезда?",
            "Күн — планета ма, әлде жұлдыз ба?",
        ),
        Term("The Sun is a star.", "Солнце — это звезда.", "Күн — жұлдыз."),
    ),
    Fact(
        Term(
            "How many minutes are in an hour?",
            "Сколько минут в часе?",
            "Бір сағатта неше минут бар?",
        ),
        Term(
            "There are 60 minutes in an hour.",
            "В часе 60 минут.",
            "Бір сағатта 60 минут бар.",
        ),
    ),
    Fact(
        Term(
            "What gas do humans breathe in to live?",
            "Каким газом люди дышат, чтобы жить?",
            "Адамдар өмір сүру үшін қандай газбен дем алады?",
        ),
        Term(
            "Humans need to breathe oxygen to live.",
            "Чтобы жить, люди дышат кислородом.",
            "Адамдар өмір сүру үшін оттегімен дем алады.",
        ),
    ),
    Fact(
        Term("What is the dombra?", "Что такое домбра?", "Домбыра деген не?"),
        Term(
            "The dombra is a traditional Kazakh two-stringed musical instrument.",
            "Домбра — это казахский традиционный двухструнный музыкальный инструмент.",
            "Домбыра — қазақтың дәстүрлі екі ішекті музыкалық аспабы.",
        ),
    ),
    Fact(
        Term(
            "How many legs does a spider have?",
            "Сколько ног у паука?",
            "Өрмекшінің неше аяғы бар?",
        ),
        Term("A spider has eight legs.", "У паука восемь ног.", "Өрмекшінің сегіз аяғы бар."),
    ),
    Fact(
        Term(
            "Which country is the largest in the world by area?",
            "Какая страна самая большая в мире по площади?",
            "Аумағы бойынша әлемдегі ең үлкен ел қайсы?",
        ),
        Term(
            "Russia is the largest country in the world by area.",
            "Россия — самая большая страна мира по площади.",
            "Аумағы бойынша әлемдегі ең үлкен ел — Ресей.",
        ),
    ),
]

SENTENCES = {
    "en": [
        "The cat sleeps on the warm sofa.",
        "Our school is near the river.",
        "It rained all day yesterday.",
        "She reads a book every evening.",
        "The train arrives at noon.",
        "Tea is better with lemon.",
        "Children play in the garden.",
        "We cook dinner together.",
        "My friend lives in a small town.",
        "The market opens early in the morning.",
        "Birds fly south in autumn.",
        "The old bridge is very long.",
    ],
    "ru": [
        "Кошка спит на тёплом диване.",
        "Наша школа находится рядом с рекой.",
        "Вчера целый день шёл дождь.",
        "Она читает книгу каждый вечер.",
        "Поезд прибывает в полдень.",
        "Чай вкуснее с лимоном.",
        "Дети играют в саду.",
        "Мы вместе готовим ужин.",
        "Мой друг живёт в маленьком городе.",
        "Рынок открывается рано утром.",
        "Осенью птицы улетают на юг.",
        "Старый мост очень длинный.",
    ],
    "kk": [
        "Мысық жылы диванда ұйықтап жатыр.",
        "Біздің мектеп өзеннің жанында орналасқан.",
        "Кеше күні бойы жаңбыр жауды.",
        "Ол күн сайын кешке кітап оқиды.",
        "Пойыз түсте келеді.",
        "Лимон қосқан шай дәмдірек.",
        "Балалар бақта ойнап жүр.",
        "Біз кешкі асты бірге дайындаймыз.",
        "Менің досым шағын қалада тұрады.",
        "Базар таңертең ерте ашылады.",
        "Күзде құстар оңтүстікке ұшады.",
        "Ескі көпір өте ұзын.",
    ],
}

PEOPLE = {
    "en": ["Timur", "Marat", "Ruslan", "Nurlan", "Yerlan", "Alibek", "Sanzhar", "Azamat", "Kairat"],
    "ru": ["Тимур", "Марат", "Руслан", "Нурлан", "Ерлан", "Алибек", "Санжар", "Азамат", "Кайрат"],
    "kk": ["Тимур", "Марат", "Руслан", "Нұрлан", "Ерлан", "Әлібек", "Санжар", "Азамат", "Қайрат"],
}

CITIES = [
    Term("Karaganda", "Караганда", "Қарағанды"),
    Term("Shymkent", "Шымкент", "Шымкент"),
    Term("Aktobe", "Актобе", "Ақтөбе"),
    Term("Pavlodar", "Павлодар", "Павлодар"),
    Term("Semey", "Семей", "Семей"),
    Term("Taraz", "Тараз", "Тараз"),
    Term("Atyrau", "Атырау", "Атырау"),
    Term("Kostanay", "Костанай", "Қостанай"),
    Term("Turkistan", "Туркестан", "Түркістан"),
    Term("Kyzylorda", "Кызылорда", "Қызылорда"),
]

JOBS = [
    Term("teacher", "учитель", "мұғалім"),
    Term("doctor", "врач", "дәрігер"),
    Term("engineer", "инженер", "инженер"),
    Term("driver", "водитель", "жүргізуші"),
    Term("painter", "художник", "суретші"),
    Term("cook", "повар", "аспаз"),
]

JOB_INSTRUMENTAL = {
    "учитель": "учителем",
    "врач": "врачом",
    "инженер": "инженером",
    "водитель": "водителем",
    "художник": "художником",
    "повар": "поваром",
}

AGES = (25, 26, 27, 28, 29, 30, 35, 36, 37, 38, 39, 40)
YEARS = tuple(range(1950, 2005))
