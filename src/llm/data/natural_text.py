from __future__ import annotations

from typing import NamedTuple


class Passage(NamedTuple):
    language: str
    text: str
    original: str
    changed: str
    question: str
    missing: str


NATURAL: tuple[Passage, ...] = (
    Passage(
        "en",
        "The community library on Mill Street is open from {V} on weekdays. On Saturdays it closes at noon. Visitors may borrow up to five books for three weeks.",
        "9:30",
        "8:00",
        "When does the library open on weekdays?",
        "How much is the annual membership fee?",
    ),
    Passage(
        "en",
        "Dr. Elena Moreau gave the opening lecture at the regional geology conference. She spoke for {V} minutes about sandstone formations in the northern valley. The audience asked questions afterwards.",
        "40",
        "55",
        "How long did Dr. Moreau speak?",
        "Which university does Dr. Moreau work at?",
    ),
    Passage(
        "en",
        "The bakery on the corner raised the price of its large loaf to {V} euros last month. Small rolls cost the same as before. Customers mostly buy bread in the morning.",
        "2.80",
        "3.10",
        "What is the price of the large loaf?",
        "How much do the small rolls cost?",
    ),
    Passage(
        "en",
        "The ferry to the island leaves from pier {V} at seven in the morning. The crossing takes about forty minutes, and tickets are sold at the kiosk.",
        "4",
        "7",
        "From which pier does the ferry leave?",
        "What is the price of a ferry ticket?",
    ),
    Passage(
        "en",
        "After three days of rain, the river near the village rose to {V} meters on Tuesday. The mayor asked residents on the lower street to move their cars. The water fell again on Thursday.",
        "3.2",
        "4.1",
        "How high did the river rise on Tuesday?",
        "What is the name of the mayor?",
    ),
    Passage(
        "en",
        "Maria is planning a trip to the mountains with her brother Tomas. They will take the train on {V} and stay in a small guesthouse near the lake. Tomas will bring the tent just in case.",
        "Friday",
        "Sunday",
        "On which day will Maria and Tomas take the train?",
        "How many nights will they stay at the guesthouse?",
    ),
    Passage(
        "ru",
        "Районная библиотека на улице Мельничной по будням открывается в {V}. По субботам она закрывается в полдень. Читатели могут брать до пяти книг на три недели.",
        "9:30",
        "8:00",
        "Во сколько библиотека открывается по будням?",
        "Сколько стоит годовой читательский билет?",
    ),
    Passage(
        "ru",
        "Доктор Елена Моро открыла региональную геологическую конференцию лекцией. Она говорила {V} минут о пластах песчаника в северной долине. После лекции слушатели задавали вопросы.",
        "40",
        "55",
        "Сколько минут говорила доктор Моро?",
        "В каком университете работает доктор Моро?",
    ),
    Passage(
        "ru",
        "Булочная на углу подняла цену большого батона до {V} евро в прошлом месяце. Маленькие булочки стоят столько же, сколько раньше. Покупатели чаще всего приходят за хлебом утром.",
        "2,80",
        "3,10",
        "Сколько стоит большой батон?",
        "Сколько стоят маленькие булочки?",
    ),
    Passage(
        "ru",
        "Паром на остров отходит от причала номер {V} в семь утра. Переправа занимает около сорока минут, билеты продаются в киоске.",
        "4",
        "7",
        "От какого причала отходит паром?",
        "Сколько стоит билет на паром?",
    ),
    Passage(
        "ru",
        "После трёх дней дождя река возле села поднялась во вторник до {V} метров. Староста попросил жителей нижней улицы переставить машины. В четверг вода снова спала.",
        "3,2",
        "4,1",
        "До какой высоты поднялась река во вторник?",
        "Как зовут старосту?",
    ),
    Passage(
        "ru",
        "Мария собирается в горы с братом Томашем. Они поедут на поезде в {V} и остановятся в небольшой гостинице у озера. Томаш на всякий случай возьмёт палатку.",
        "пятницу",
        "воскресенье",
        "В какой день Мария и Томаш поедут на поезде?",
        "Сколько ночей они проведут в гостинице?",
    ),
    Passage(
        "kk",
        "Диірмен көшесіндегі аудандық кітапхана жұмыс күндері сағат {V}-да ашылады. Сенбіде түсте жабылады. Оқырмандар бес кітапты үш аптаға алып кете алады.",
        "9:30",
        "8:00",
        "Кітапхана жұмыс күндері сағат нешеде ашылады?",
        "Жылдық оқырман билеті қанша тұрады?",
    ),
    Passage(
        "kk",
        "Доктор Елена Моро аймақтық геология конференциясын дәрісімен ашты. Ол солтүстік алқаптағы құмтас қабаттары туралы {V} минут сөйледі. Дәрістен кейін тыңдаушылар сұрақ қойды.",
        "40",
        "55",
        "Доктор Моро неше минут сөйледі?",
        "Доктор Моро қай университетте жұмыс істейді?",
    ),
    Passage(
        "kk",
        "Бұрыштағы наубайхана өткен айда үлкен нанның бағын {V} еуроға көтерді. Шағын тоқаштардың бағы бұрынғыдай. Сатып алушылар көбіне нанды таңертең алады.",
        "2,80",
        "3,10",
        "Үлкен нан қанша тұрады?",
        "Шағын тоқаштар қанша тұрады?",
    ),
    Passage(
        "kk",
        "Аралға баратын паром таңғы жетіде {V}-нөмірлі айлақтан жүзеді. Өту шамамен қырық минутқа созылады, билеттер киоскіде сатылады.",
        "4",
        "7",
        "Паром қай айлақтан жүзеді?",
        "Паромның билеті қанша тұрады?",
    ),
    Passage(
        "kk",
        "Үш күн жауған жаңбырдан кейін ауыл маңындағы өзен сейсенбіде {V} метрге дейін көтерілді. Ауыл старшынасы төменгі көшенің тұрғындарынан көліктерін ауыстыруды сұрады. Бейсенбіде су қайта қайтты.",
        "3,2",
        "4,1",
        "Өзен сейсенбіде қанша метрге көтерілді?",
        "Старшинаның аты кім?",
    ),
    Passage(
        "kk",
        "Мария ағасы Томашпен бірге тауға барғалы жатыр. Олар {V} пойызбен барып, көл жағасындағы шағын қонақүйде аялдайды. Томаш болмаса дегенде шатырды алып барады.",
        "жұма күні",
        "жексенбі күні",
        "Мария мен Томаш қай күні пойызбен барады?",
        "Олар қонақүйде неше түн болады?",
    ),
)
