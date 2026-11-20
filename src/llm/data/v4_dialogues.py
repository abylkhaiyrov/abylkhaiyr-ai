from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from llm.data.binary import read_jsonl
from llm.data.dedup import stable_hash
from llm.data.v4_curated import kazsandra_rows
from llm.data.v4_quality import LANGUAGES, CyrillicLanguageFilter, clean_v4_text
from llm.data.v4_sources import parquet_rows
from llm.tokenizer.bpe import BPETokenizer, Message

OASST_LABELS = ("spam", "fails_task", "lang_mismatch", "pii", "not_appropriate", "hate_speech")
ROLE_NAMES = {"prompter": "user", "assistant": "assistant"}
LANGUAGE_CODES = {"eng": "en", "rus": "ru", "kaz": "kk"}
PROMPT_WORDS = re.compile(r"[^\W\d_]+")
TRANSLATE = {
    "en": "Translate the following into English.",
    "ru": "Переведи следующий текст на русский.",
    "kk": "Мына мәтінді қазақшаға аудар.",
}
INTENT_PATTERNS = (
    ("translation", r"\b(translate|переведи|переведите|аудар|аударыңыз)\b"),
    ("causal", r"\b(why|почему|зачем|неге|неліктен)\b"),
    ("comparison", r"\b(compare|difference|сравни|разница|салыстыр|айырмашылығы)\b"),
    ("summarization", r"\b(summarize|summary|кратко|резюме|қорытындыла)\b"),
    ("explanation", r"\b(explain|объясни|объясните|түсіндір|түсіндіріңіз)\b"),
    ("greeting", r"^(hi|hello|hey|привет|здравствуйте|сәлем|сәлеметсіз бе)[.!? ]*$"),
    ("code", r"\b(python|javascript|sql|java|код|функцию|бағдарлама)\b"),
    ("math", r"\b(calculate|solve|вычисли|реши|есепте)\b"),
    ("knowledge", r"\b(what is|who is|where is|что такое|кто такой|қайда|деген не)\b"),
)


@dataclass(frozen=True)
class Dialogue:
    source: str
    source_id: str
    family: str
    language: str
    category: str
    messages: tuple[Message, ...]

    @property
    def key(self) -> str:
        return f"{self.source}/{self.source_id}"

    @property
    def prompt(self) -> str:
        return "\n".join(message.content for message in self.messages[:-1])

    def as_row(self) -> dict:
        return {"id": self.key, **asdict(self)}


def topic(prompt: str, turns: int) -> str:
    if turns > 2:
        return "multi_turn"
    lowered = prompt.lower().strip()
    for category, pattern in INTENT_PATTERNS:
        if re.search(pattern, lowered):
            return category
    return "instruction"


def oasst_reviewed(row: dict) -> bool:
    if (
        row["deleted"]
        or row["synthetic"]
        or row["model_name"]
        or not row["review_result"]
        or row["tree_state"] != "ready_for_export"
    ):
        return False
    if row["role"] == "assistant" and row["rank"] != 0:
        return False
    labels = row.get("labels") or {}
    scores = dict(zip(labels.get("name") or [], labels.get("value") or [], strict=True))
    return all((scores.get(label) or 0) < 0.5 for label in OASST_LABELS)


def oasst_path(
    leaf: dict, messages: dict[str, dict], quality: CyrillicLanguageFilter
) -> tuple[Message, ...]:
    language = leaf["lang"]
    chain: list[Message] = []
    node = leaf
    visited: set[str] = set()
    while True:
        identifier = node["message_id"]
        role = ROLE_NAMES.get(node["role"])
        if (
            identifier in visited
            or role is None
            or node["lang"] != language
            or node["message_tree_id"] != leaf["message_tree_id"]
        ):
            return ()
        visited.add(identifier)
        text = clean_v4_text(node["text"], 2)
        if text is None or not quality.accepts(text, language, allow_short=True):
            return ()
        chain.append(Message(role, text))
        parent = node["parent_id"]
        if parent is None:
            break
        if parent not in messages:
            return ()
        node = messages[parent]
    chain.reverse()
    if len(chain) < 2 or len(chain) % 2:
        return ()
    if any(
        message.role != ("user" if index % 2 == 0 else "assistant")
        for index, message in enumerate(chain)
    ):
        return ()
    return tuple(chain)


def oasst_dialogues(
    source: str, path: Path, quality: CyrillicLanguageFilter, tokenizer: BPETokenizer
) -> Iterator[Dialogue]:
    messages = {
        row["message_id"]: row
        for row in parquet_rows(path)
        if row["lang"] in LANGUAGES and oasst_reviewed(row)
    }
    by_tree: dict[str, list[Dialogue]] = defaultdict(list)
    for row in messages.values():
        if row["role"] != "assistant":
            continue
        path_messages = oasst_path(row, messages, quality)
        if not path_messages or len(tokenizer.encode_conversation(list(path_messages))) > 512:
            continue
        family = row["message_tree_id"]
        by_tree[family].append(
            Dialogue(
                source,
                row["message_id"],
                f"oasst1/{family}",
                row["lang"],
                topic(path_messages[-2].content, len(path_messages)),
                path_messages,
            )
        )
    for candidates in by_tree.values():
        yield min(candidates, key=lambda item: (-len(item.messages), stable_hash(item.key)))


def instruction_pair(
    source: str,
    source_id: str,
    family: str,
    language: str,
    category: str,
    prompt: str,
    answer: str,
    quality: CyrillicLanguageFilter,
    tokenizer: BPETokenizer,
) -> Dialogue | None:
    question = clean_v4_text(prompt, 2)
    response = clean_v4_text(answer, 2)
    if (
        question is None
        or response is None
        or not quality.accepts(question, language, allow_short=True)
        or not quality.accepts(response, language, allow_short=True)
    ):
        return None
    messages = (Message("user", question), Message("assistant", response))
    if len(tokenizer.encode_conversation(list(messages))) > 512:
        return None
    return Dialogue(source, source_id, family, language, category, messages)


def dolly_dialogues(
    path: Path, quality: CyrillicLanguageFilter, tokenizer: BPETokenizer
) -> Iterator[Dialogue]:
    for row in read_jsonl(path):
        instruction, context = row["instruction"], row["context"]
        identifier = f"{stable_hash(instruction + chr(10) + context):016x}"
        category = {
            "classification": "classification",
            "information_extraction": "extraction",
            "summarization": "summarization",
            "closed_qa": "reading",
            "open_qa": "knowledge",
            "general_qa": "explanation",
        }.get(row["category"], "instruction")
        family = f"dolly/{stable_hash(context):016x}" if context else f"dolly/{identifier}"
        prompt = f"{instruction}\n\n{context}" if context else instruction
        dialogue = instruction_pair(
            "dolly",
            identifier,
            family,
            "en",
            category,
            prompt,
            row["response"],
            quality,
            tokenizer,
        )
        if dialogue is not None:
            yield dialogue


def aya_dialogues(
    path: Path, quality: CyrillicLanguageFilter, tokenizer: BPETokenizer
) -> Iterator[Dialogue]:
    for row in parquet_rows(path):
        language = LANGUAGE_CODES.get(row["language_code"])
        if not language or row["annotation_type"] != "original-annotations":
            continue
        prompt = row["inputs"]
        category = topic(prompt, 2)
        if category == "translation":
            continue
        identifier = f"{stable_hash(prompt):016x}"
        dialogue = instruction_pair(
            "aya_train",
            identifier,
            f"aya/{identifier}",
            language,
            category,
            prompt,
            row["targets"],
            quality,
            tokenizer,
        )
        if dialogue is not None:
            yield dialogue


def support_dialogues(
    path: Path, quality: CyrillicLanguageFilter, tokenizer: BPETokenizer
) -> Iterator[Dialogue]:
    for row in read_jsonl(path):
        if row["slots"]:
            continue
        dialogue = instruction_pair(
            "kk_support",
            row["id"],
            f"kk_support/{row['intent']}",
            "kk",
            "instruction",
            row["question"],
            row["answer"],
            quality,
            tokenizer,
        )
        if dialogue is not None:
            yield dialogue


def squad_dialogues(
    source: str, path: Path, quality: CyrillicLanguageFilter, tokenizer: BPETokenizer
) -> Iterator[Dialogue]:
    for row in parquet_rows(path):
        context, answers = row["context"], row["answers"]
        valid = [
            text
            for start, text in zip(answers["answer_start"], answers["text"], strict=True)
            if context[start : start + len(text)] == text
        ]
        if not valid:
            continue
        dialogue = instruction_pair(
            source,
            row["id"],
            f"squad/{row['title']}",
            "en",
            "reading",
            f"{context}\n\n{row['question']}",
            valid[0],
            quality,
            tokenizer,
        )
        if dialogue is not None:
            yield dialogue


def belebele_dialogues(
    source: str, path: Path, language: str, quality: CyrillicLanguageFilter, tokenizer: BPETokenizer
) -> Iterator[Dialogue]:
    for row in read_jsonl(path):
        passage = row["flores_passage"]
        question = row["question"]
        family = f"web/{stable_hash(row['link']):016x}"
        identifier = f"{row['ds']}/{row['question_number']}/{stable_hash(passage):016x}"
        options = [row[f"mc_answer{index}"] for index in range(1, 5)]
        if not quality.accepts(passage, language) or not quality.accepts(
            question, language, allow_short=True
        ):
            continue
        if any(clean_v4_text(option, 1) is None for option in options):
            continue
        lines = "\n".join(f"{index}. {option}" for index, option in enumerate(options, 1))
        response = options[int(row["correct_answer_num"]) - 1]
        dialogue = instruction_pair(
            source,
            identifier,
            family,
            language,
            "reading",
            f"{passage}\n\n{question}\n{lines}",
            response,
            quality,
            tokenizer,
        )
        if dialogue is not None:
            yield dialogue


def kazsandra_dialogues(
    path: Path, quality: CyrillicLanguageFilter, tokenizer: BPETokenizer
) -> Iterator[Dialogue]:
    polarity = {"0": "Жағымсыз.", "1": "Жағымды."}
    for row in kazsandra_rows(path, semantic=True):
        review, label = row["text"], row["label"]
        if label not in polarity or len(review) < 40:
            continue
        identifier = row["custom_id"]
        dialogue = instruction_pair(
            "kazsandra",
            identifier,
            f"kazsandra/{identifier}",
            "kk",
            "classification",
            f"{review}\n\nПікірдің реңкі жағымды ма, жағымсыз ба?",
            polarity[label],
            quality,
            tokenizer,
        )
        if dialogue is not None:
            yield dialogue
