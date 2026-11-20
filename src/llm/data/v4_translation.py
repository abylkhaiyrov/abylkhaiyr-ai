from __future__ import annotations

import csv
import tarfile
from collections.abc import Iterator
from pathlib import Path

from llm.data.dedup import stable_hash
from llm.data.v4_dialogues import TRANSLATE, Dialogue
from llm.data.v4_quality import CyrillicLanguageFilter, clean_v4_text
from llm.data.v4_sources import fetch_snapshot
from llm.tokenizer.bpe import BPETokenizer, Message

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_LICENSE_URL = "https://huggingface.co/datasets/facebook/flores"
FLORES_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
FLORES_CODES = {"en": "eng_Latn", "ru": "rus_Cyrl", "kk": "kaz_Cyrl"}
TRANSLATION_SOURCE = {"en": "ru", "ru": "en", "kk": "en"}


def fetch_flores(root: Path) -> Path:
    result = root / "flores200.tar.gz"
    fetch_snapshot(FLORES_URL, result, FLORES_SHA256)
    return result


def flores_dialogues(
    path: Path,
    split: str,
    language: str,
    quality: CyrillicLanguageFilter,
    tokenizer: BPETokenizer,
) -> Iterator[Dialogue]:
    origin = TRANSLATION_SOURCE[language]
    texts = {}
    with tarfile.open(path) as archive:
        for lang in (origin, language):
            code = FLORES_CODES[lang]
            name = f"./flores200_dataset/{split}/{code}.{split}"
            data = archive.extractfile(name)
            if data is None:
                raise ValueError(f"missing FLORES member: {name}")
            texts[lang] = data.read().decode("utf-8").splitlines()
        metadata = archive.extractfile(f"./flores200_dataset/metadata_{split}.tsv")
        if metadata is None:
            raise ValueError(f"missing FLORES metadata: {split}")
        urls = [
            row["URL"]
            for row in csv.DictReader(
                metadata.read().decode("utf-8").splitlines(),
                delimiter="\t",
            )
        ]
    parallel = zip(texts[origin], texts[language], urls, strict=True)
    for index, (source, target, url) in enumerate(parallel):
        text = clean_v4_text(source, 12)
        answer = clean_v4_text(target, 12)
        if not text or not answer or text.casefold() == answer.casefold():
            continue
        length_ratio = len(text) / len(answer)
        if not 0.38 <= length_ratio <= 2.65:
            continue
        if not quality.accepts(text, origin) or not quality.accepts(answer, language):
            continue
        messages = (
            Message("user", f"{TRANSLATE[language]}\n\n{text}"),
            Message("assistant", answer),
        )
        if len(tokenizer.encode_conversation(list(messages))) > 512:
            continue
        yield Dialogue(
            "flores200",
            f"{split}/{index}/{language}",
            f"web/{stable_hash(url):016x}",
            language,
            "translation",
            messages,
        )
