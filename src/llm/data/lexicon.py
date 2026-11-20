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


CODE_CONCEPTS = [
    Concept(
        Term("SOLID", "SOLID", "SOLID"),
        Term(
            "SOLID is a set of five object-oriented design principles: single responsibility, open/closed, Liskov substitution, interface segregation and dependency inversion.",
            "SOLID — это пять принципов объектно-ориентированного проектирования: единственная ответственность, открытость/закрытость, подстановка Лисков, разделение интерфейсов и инверсия зависимостей.",
            "SOLID — объектіге бағытталған жобалаудың бес принципі: жалғыз жауапкершілік, ашықтық/жабықтық, Лисков алмастыруы, интерфейстерді бөлу және тәуелділікті инверсиялау.",
        ),
    ),
    Concept(
        Term(
            "the single responsibility principle",
            "принцип единственной ответственности",
            "жалғыз жауапкершілік принципі",
        ),
        Term(
            "The single responsibility principle says that a class or function should have only one reason to change.",
            "Принцип единственной ответственности гласит, что у класса или функции должна быть только одна причина для изменения.",
            "Жалғыз жауапкершілік принципі бойынша класс немесе функцияның өзгеруіне тек бір ғана себеп болуы керек.",
        ),
    ),
    Concept(
        Term(
            "the open/closed principle",
            "принцип открытости/закрытости",
            "ашықтық/жабықтық принципі",
        ),
        Term(
            "The open/closed principle says that code should be open for extension but closed for modification.",
            "Принцип открытости/закрытости: код должен быть открыт для расширения, но закрыт для изменения.",
            "Ашықтық/жабықтық принципі: код кеңейтуге ашық, бірақ өзгертуге жабық болуы керек.",
        ),
    ),
    Concept(
        Term(
            "the Liskov substitution principle",
            "принцип подстановки Лисков",
            "Лисков алмастыру принципі",
        ),
        Term(
            "The Liskov substitution principle says that objects of a subclass must be usable wherever objects of the base class are expected without breaking the program.",
            "Принцип подстановки Лисков: объекты подкласса должны использоваться везде, где ожидаются объекты базового класса, не нарушая работу программы.",
            "Лисков алмастыру принципі: туынды класс объектілерін базалық класс күтілетін жерде бағдарламаны бұзбай қолдануға болуы керек.",
        ),
    ),
    Concept(
        Term(
            "the interface segregation principle",
            "принцип разделения интерфейсов",
            "интерфейстерді бөлу принципі",
        ),
        Term(
            "The interface segregation principle says that clients should not be forced to depend on methods they do not use, so interfaces should be small and focused.",
            "Принцип разделения интерфейсов: клиенты не должны зависеть от методов, которыми не пользуются, поэтому интерфейсы должны быть небольшими и узкоспециализированными.",
            "Интерфейстерді бөлу принципі: клиенттер өздері қолданбайтын әдістерге тәуелді болмауы керек, сондықтан интерфейстер шағын әрі нақты болуы тиіс.",
        ),
    ),
    Concept(
        Term(
            "the dependency inversion principle",
            "принцип инверсии зависимостей",
            "тәуелділікті инверсиялау принципі",
        ),
        Term(
            "The dependency inversion principle says that high-level modules should depend on abstractions rather than on concrete low-level implementations.",
            "Принцип инверсии зависимостей: модули высокого уровня должны зависеть от абстракций, а не от конкретных низкоуровневых реализаций.",
            "Тәуелділікті инверсиялау принципі: жоғары деңгейлі модульдер нақты төмен деңгейлі іске асыруларға емес, абстракцияларға тәуелді болуы керек.",
        ),
    ),
    Concept(
        Term("clean code", "чистый код", "таза код"),
        Term(
            "Clean code is code that is easy to read, understand and change: it has clear names, small functions and no unnecessary duplication.",
            "Чистый код — это код, который легко читать, понимать и изменять: понятные имена, небольшие функции и отсутствие лишнего дублирования.",
            "Таза код — оқуға, түсінуге және өзгертуге оңай код: түсінікті атаулар, шағын функциялар және артық қайталанусыз.",
        ),
    ),
    Concept(
        Term("the DRY principle", "принцип DRY", "DRY принципі"),
        Term(
            "DRY means don't repeat yourself: every piece of knowledge should have a single representation in the code.",
            "DRY означает «не повторяйся»: каждый фрагмент знания должен иметь в коде единственное представление.",
            "DRY «қайталанба» дегенді білдіреді: әрбір білім бөлігінің кодта бір ғана көрінісі болуы керек.",
        ),
    ),
    Concept(
        Term("the KISS principle", "принцип KISS", "KISS принципі"),
        Term(
            "KISS means keep it simple: prefer the simplest solution that works.",
            "KISS означает «делай проще»: выбирай самое простое решение, которое работает.",
            "KISS «қарапайым ұста» дегенді білдіреді: жұмыс істейтін ең қарапайым шешімді таңда.",
        ),
    ),
    Concept(
        Term("a unit test", "модульный тест", "модульдік тест"),
        Term(
            "A unit test is a small automatic check that verifies that one piece of code, such as a function, behaves as expected.",
            "Модульный тест — это небольшая автоматическая проверка того, что отдельный фрагмент кода, например функция, работает как ожидается.",
            "Модульдік тест — функция сияқты код бөлігінің күтілгендей жұмыс істейтінін тексеретін шағын автоматты тексеру.",
        ),
    ),
    Concept(
        Term("refactoring", "рефакторинг", "рефакторинг"),
        Term(
            "Refactoring is changing the structure of code to make it cleaner without changing what it does.",
            "Рефакторинг — это изменение структуры кода для улучшения его качества без изменения его поведения.",
            "Рефакторинг — кодтың жұмысын өзгертпей, оның құрылымын жақсарту.",
        ),
    ),
    Concept(
        Term("recursion", "рекурсия", "рекурсия"),
        Term(
            "Recursion is a technique where a function solves a problem by calling itself on smaller inputs until it reaches a base case.",
            "Рекурсия — это приём, при котором функция решает задачу, вызывая саму себя для меньших входных данных, пока не дойдёт до базового случая.",
            "Рекурсия — функция есепті өзін кішірек деректермен шақыру арқылы, базалық жағдайға жеткенше шешетін әдіс.",
        ),
    ),
    Concept(
        Term("Big O notation", "O-большое", "O-үлкен белгілеуі"),
        Term(
            "Big O notation describes how the running time or memory of an algorithm grows as the input size grows.",
            "O-большое описывает, как время работы или память алгоритма растут с увеличением размера входных данных.",
            "O-үлкен белгілеуі кіріс деректер өлшемі өскенде алгоритмнің жұмыс уақыты немесе жады қалай өсетінін сипаттайды.",
        ),
    ),
    Concept(
        Term("a stack", "стек", "стек"),
        Term(
            "A stack is a data structure that works on the last in, first out principle: the last element added is the first one removed.",
            "Стек — это структура данных, работающая по принципу «последним пришёл — первым вышел».",
            "Стек — «соңғы кірген — бірінші шығады» принципімен жұмыс істейтін деректер құрылымы.",
        ),
    ),
    Concept(
        Term("a queue", "очередь", "кезек"),
        Term(
            "A queue is a data structure that works on the first in, first out principle: the first element added is the first one removed.",
            "Очередь — это структура данных, работающая по принципу «первым пришёл — первым вышел».",
            "Кезек — «бірінші кірген — бірінші шығады» принципімен жұмыс істейтін деректер құрылымы.",
        ),
    ),
    Concept(
        Term("a hash table", "хеш-таблица", "хэш-кесте"),
        Term(
            "A hash table stores key-value pairs and uses a hash function to find a value by its key in constant time on average.",
            "Хеш-таблица хранит пары «ключ — значение» и с помощью хеш-функции в среднем находит значение по ключу за постоянное время.",
            "Хэш-кесте «кілт — мән» жұптарын сақтайды және хэш-функция арқылы мәнді кілті бойынша орта есеппен тұрақты уақытта табады.",
        ),
    ),
    Concept(
        Term("a linked list", "связный список", "байланыстырылған тізім"),
        Term(
            "A linked list is a data structure made of nodes, where each node stores a value and a reference to the next node.",
            "Связный список — это структура данных из узлов, где каждый узел хранит значение и ссылку на следующий узел.",
            "Байланыстырылған тізім — түйіндерден тұратын деректер құрылымы, мұнда әр түйін мәнді және келесі түйінге сілтемені сақтайды.",
        ),
    ),
    Concept(
        Term("polymorphism", "полиморфизм", "полиморфизм"),
        Term(
            "Polymorphism lets objects of different classes be used through the same interface, each responding in its own way.",
            "Полиморфизм позволяет использовать объекты разных классов через один интерфейс, причём каждый отвечает по-своему.",
            "Полиморфизм әртүрлі класс объектілерін бір интерфейс арқылы қолдануға мүмкіндік береді, әрқайсысы өзінше жауап береді.",
        ),
    ),
    Concept(
        Term("encapsulation", "инкапсуляция", "инкапсуляция"),
        Term(
            "Encapsulation means hiding the internal state of an object and exposing it only through methods.",
            "Инкапсуляция — это сокрытие внутреннего состояния объекта и доступ к нему только через методы.",
            "Инкапсуляция — объектінің ішкі күйін жасырып, оған тек әдістер арқылы қол жеткізу.",
        ),
    ),
    Concept(
        Term("inheritance", "наследование", "мұрагерлік"),
        Term(
            "Inheritance lets a class reuse the fields and methods of another class and extend or change them.",
            "Наследование позволяет классу повторно использовать поля и методы другого класса, расширяя или изменяя их.",
            "Мұрагерлік класқа басқа таптың өрістері мен әдістерін қайта пайдалануға, оларды кеңейтуге немесе өзгертуге мүмкіндік береді.",
        ),
    ),
    Concept(
        Term("binary search", "бинарный поиск", "бинарлық іздеу"),
        Term(
            "Binary search finds an element in a sorted list by repeatedly halving the search range, which takes logarithmic time.",
            "Бинарный поиск находит элемент в отсортированном списке, многократно деля диапазон поиска пополам, и работает за логарифмическое время.",
            "Бинарлық іздеу сұрыпталған тізімнен элементті іздеу ауқымын қайта-қайта екіге бөлу арқылы табады және логарифмдік уақытта жұмыс істейді.",
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
