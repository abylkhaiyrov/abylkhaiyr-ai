from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Iterator

from datasets import load_dataset

from llm.data.cleaning import clean_document

OASST_DATASET = "OpenAssistant/oasst1"
OASST_LICENSE = "Apache-2.0"
ROLE_MAP = {"prompter": "user", "assistant": "assistant"}
OASST_LANGUAGES = ("en", "ru")


def load_oasst_messages(languages: tuple[str, ...] = OASST_LANGUAGES) -> list[dict]:
    rows = []
    for split in ("train", "validation"):
        for row in load_dataset(OASST_DATASET, split=split, streaming=True):
            if row["deleted"] or row["lang"] not in languages or row["review_result"] is False:
                continue
            rows.append(row)
    return rows


class MessageTree:
    def __init__(self, rows: Iterable[dict]) -> None:
        self.by_id = {row["message_id"]: row for row in rows}
        self.children: dict[str, list[dict]] = defaultdict(list)
        for row in self.by_id.values():
            if row["parent_id"] in self.by_id:
                self.children[row["parent_id"]].append(row)

    def path(self, message_id: str) -> list[dict]:
        chain: list[dict] = []
        current: dict | None = self.by_id.get(message_id)
        while current is not None:
            chain.append(current)
            current = self.by_id.get(current["parent_id"])
        return chain[::-1]

    def to_messages(self, chain: list[dict]) -> list[dict[str, str]]:
        return [
            {"role": ROLE_MAP[row["role"]], "content": clean_document(row["text"], False)}
            for row in chain
        ]

    def ranked_children(self, message_id: str) -> list[dict]:
        return sorted(
            (c for c in self.children[message_id] if c.get("rank") is not None),
            key=lambda c: c["rank"],
        )


def conversations(tree: MessageTree) -> Iterator[dict]:
    for message_id, row in tree.by_id.items():
        if row["role"] != "assistant" or row.get("rank") != 0 or tree.children[message_id]:
            continue
        chain = tree.path(message_id)
        if chain[0]["role"] == "prompter" and all(
            c["role"] == ROLE_MAP_INVERSE[i % 2] for i, c in enumerate(chain)
        ):
            yield {
                "source": "oasst1",
                "group": "chat",
                "language": row["lang"],
                "messages": tree.to_messages(chain),
            }


ROLE_MAP_INVERSE = ("prompter", "assistant")


def preference_pairs(tree: MessageTree) -> Iterator[dict]:
    for message_id, row in tree.by_id.items():
        if row["role"] != "prompter":
            continue
        replies = tree.ranked_children(message_id)
        if len(replies) < 2:
            continue
        prompt = tree.to_messages(tree.path(message_id))
        best, worst = replies[0], replies[-1]
        yield {
            "source": "oasst1",
            "language": row["lang"],
            "chosen": prompt + tree.to_messages([best]),
            "rejected": prompt + tree.to_messages([worst]),
        }


REFUSALS = {
    "en": [
        (
            "What is the weather in Almaty right now?",
            "I can't see real-time data, so I don't know the current weather in Almaty. A weather service or app will give you an accurate forecast.",
        ),
        (
            "What did I have for breakfast today?",
            "I have no access to your personal life, so I can't know what you ate. If you tell me, I can help you plan meals.",
        ),
        (
            "What will the exchange rate of the dollar be next year?",
            "Nobody can predict future exchange rates reliably, and I don't have the data to try. I can explain the factors that move exchange rates if that helps.",
        ),
        (
            "What is the phone number of my neighbour?",
            "I don't have personal contact information about people, so I can't tell you that.",
        ),
        (
            "Who won yesterday's football match?",
            "I don't have access to current news or results, so I can't say who won. A sports website will have the latest score.",
        ),
        (
            "What is the password of my wifi?",
            "I can't know your password because I have no access to your devices. You can usually find it on the router or in its settings.",
        ),
        (
            "What is the exact population of my city today?",
            "I don't know the exact current figure, and it changes constantly. Official statistics offices publish the most reliable numbers.",
        ),
        (
            "Tell me the secret ingredient of a restaurant's private recipe.",
            "I don't have access to private recipes, so I can't tell you that. I can suggest similar recipes that are publicly known.",
        ),
    ],
    "ru": [
        (
            "Какая сейчас погода в Алматы?",
            "У меня нет доступа к данным в реальном времени, поэтому я не знаю текущую погоду в Алматы. Точный прогноз покажет метеосервис.",
        ),
        (
            "Что я ел сегодня на завтрак?",
            "У меня нет доступа к вашей личной жизни, поэтому я не могу знать, что вы ели. Если расскажете, я помогу составить меню.",
        ),
        (
            "Каким будет курс доллара через год?",
            "Курс валют нельзя надёжно предсказать, и у меня нет для этого данных. Могу объяснить, какие факторы влияют на курс.",
        ),
        (
            "Какой номер телефона у моего соседа?",
            "У меня нет личных контактных данных людей, поэтому я не могу вам это сказать.",
        ),
        (
            "Кто выиграл вчерашний матч?",
            "У меня нет доступа к свежим новостям и результатам, поэтому я не знаю, кто выиграл. Актуальный счёт есть на спортивных сайтах.",
        ),
        (
            "Какой пароль от моего вайфая?",
            "Я не могу знать ваш пароль, потому что у меня нет доступа к вашим устройствам. Обычно он указан на роутере или в его настройках.",
        ),
    ],
    "kk": [
        (
            "Алматыда қазір ауа райы қандай?",
            "Менде нақты уақыттағы деректерге қолжетімділік жоқ, сондықтан Алматыдағы қазіргі ауа райын білмеймін. Дәл болжамды метеорология қызметі көрсетеді.",
        ),
        (
            "Мен бүгін таңғы асқа не жедім?",
            "Менде сіздің жеке өміріңізге қолжетімділік жоқ, сондықтан не жегеніңізді білмеймін. Айтсаңыз, мәзір құруға көмектесемін.",
        ),
        (
            "Келесі жылы доллар бағамы қандай болады?",
            "Валюта бағамын сенімді болжау мүмкін емес, менде мұндай деректер де жоқ. Бағамға әсер ететін факторларды түсіндіре аламын.",
        ),
        (
            "Көршімнің телефон нөмірі қандай?",
            "Менде адамдардың жеке байланыс деректері жоқ, сондықтан мұны айта алмаймын.",
        ),
        (
            "Кеше матчта кім жеңді?",
            "Менде соңғы жаңалықтар мен нәтижелерге қолжетімділік жоқ, сондықтан кім жеңгенін білмеймін. Нақты есепті спорт сайттарынан көруге болады.",
        ),
    ],
}


def refusal_examples(seed: int, repeats: int = 3) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for language, pairs in REFUSALS.items():
        for question, answer in pairs:
            for _ in range(repeats):
                rows.append(
                    {
                        "source": "synthetic_refusal",
                        "group": "chat",
                        "language": language,
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ],
                    }
                )
    rng.shuffle(rows)
    return rows
