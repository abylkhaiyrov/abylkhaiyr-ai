import pytest

from llm.core.errors import TokenizerError
from llm.tokenizer.bpe import SPECIAL_TOKENS, BPETokenizer, Message

TEXTS = [
    "Hello, world!",
    "Привет, мир! Это тест.",
    "Сәлем, әлем! Қазақстан Республикасы — ұлы ел.",
    "def f(x):\n\treturn x ** 2  # square\n",
    "∫ sin(x) dx = −cos(x) + C; ∑ᵢ aᵢ ≤ π",
    "emoji 🙂 and 日本語 and \x00 odd  spaces\n\n",
    "",
]


def test_roundtrip_preserves_text_exactly(tokenizer):
    for text in TEXTS:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_unknown_characters_never_fail(tokenizer):
    text = "𓀀𓀁 ꙮ"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_exist_and_are_unique(tokenizer):
    ids = {tokenizer.backend.token_to_id(t) for t in SPECIAL_TOKENS}
    assert None not in ids and len(ids) == len(SPECIAL_TOKENS)


def test_special_token_text_is_not_injected_from_user_content(tokenizer):
    ids = tokenizer.encode("<eos> <user>")
    assert tokenizer.eos_id not in ids and tokenizer.role_ids["user"] not in ids


def test_save_and_load_give_identical_encoding(tokenizer, tmp_path):
    tokenizer.save(tmp_path)
    loaded = BPETokenizer.load(tmp_path)
    assert loaded.version == tokenizer.version
    assert loaded.encode(TEXTS[2]) == tokenizer.encode(TEXTS[2])


def test_load_missing_raises(tmp_path):
    with pytest.raises(TokenizerError):
        BPETokenizer.load(tmp_path / "missing")


def test_conversation_labels_mask_everything_except_assistant(tokenizer):
    messages = [
        Message("system", "be brief"),
        Message("user", "hi"),
        Message("assistant", "hello"),
        Message("user", "bye"),
        Message("assistant", "ciao"),
    ]
    ids, labels = tokenizer.encode_conversation_with_labels(messages, -100)
    assert len(ids) == len(labels)
    trained = [i for i, label in zip(ids, labels) if label != -100]
    assert trained == [
        *tokenizer.encode("hello"),
        tokenizer.eos_id,
        *tokenizer.encode("ciao"),
        tokenizer.eos_id,
    ]
    assert labels[0] == -100
    assert ids == tokenizer.encode_conversation(messages)


def test_generation_prompt_ends_with_assistant_token(tokenizer):
    ids = tokenizer.encode_conversation([Message("user", "hi")], add_generation_prompt=True)
    assert ids[-1] == tokenizer.role_ids["assistant"]


def test_unsupervised_assistant_turns_are_context_only(tokenizer):
    from llm.data.binary import IGNORE_INDEX
    from llm.tokenizer.bpe import Message

    messages = [
        Message("user", "What is 17 + 25?"),
        Message("assistant", "It is 41.", supervised=False),
        Message("user", "That is wrong, please correct it."),
        Message("assistant", "You are right, it is 42."),
    ]
    ids, labels = tokenizer.encode_conversation_with_labels(messages, IGNORE_INDEX)
    supervised = [label for label in labels if label != IGNORE_INDEX]
    assert supervised == tokenizer.encode("You are right, it is 42.") + [tokenizer.eos_id]
    wrong = tokenizer.encode("It is 41.")
    assert all(label == IGNORE_INDEX for label in labels[: len(ids) - len(supervised) - 1])
    assert wrong[0] in ids
