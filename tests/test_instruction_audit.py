import pytest
import torch

from llm.data.binary import IGNORE_INDEX
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.model.transformer import CausalLM
from llm.tokenizer.bpe import Message
from llm.training.losses import causal_lm_loss_sum

CPU = torch.device("cpu")

HISTORY = [
    Message("system", "You are a helper."),
    Message("user", "hi"),
    Message("assistant", "hello"),
    Message("user", "what is 2 + 2?"),
]


@pytest.fixture()
def engine(model, tokenizer):
    torch.manual_seed(0)
    return InferenceEngine(CausalLM(model.config), tokenizer, CPU)


def test_generation_prompt_is_exact_prefix_of_training_sequence(tokenizer):
    full = [*HISTORY, Message("assistant", "four")]
    ids, labels = tokenizer.encode_conversation_with_labels(full, IGNORE_INDEX)
    prompt = tokenizer.encode_conversation(HISTORY, add_generation_prompt=True)
    assert ids[: len(prompt)] == prompt
    assert prompt[-1] == tokenizer.role_ids["assistant"]
    assert labels[len(prompt) - 1] == IGNORE_INDEX
    assert all(label != IGNORE_INDEX for label in labels[len(prompt) :])
    assert ids[-1] == tokenizer.eos_id and labels[-1] == tokenizer.eos_id


def test_training_targets_are_next_tokens_of_assistant_reply_only(tokenizer):
    ids, labels = tokenizer.encode_conversation_with_labels(
        [Message("user", "hi"), Message("assistant", "hello")], IGNORE_INDEX
    )
    inputs, targets = ids[:-1], labels[1:]
    answer = tokenizer.encode("hello") + [tokenizer.eos_id]
    supervised = [t for t in targets if t != IGNORE_INDEX]
    assert supervised == answer
    for position, target in enumerate(targets):
        if target != IGNORE_INDEX:
            assert target == ids[position + 1]
    assert inputs[0] == tokenizer.bos_id


def test_padding_does_not_change_loss_of_the_shorter_sequence(model, tokenizer):
    ids, labels = tokenizer.encode_conversation_with_labels(
        [Message("user", "hi"), Message("assistant", "ok")], IGNORE_INDEX
    )
    long_ids, long_labels = tokenizer.encode_conversation_with_labels(
        [Message("user", "hi there my friend"), Message("assistant", "ok then, good")],
        IGNORE_INDEX,
    )
    width = max(len(ids), len(long_ids))
    pad = tokenizer.pad_id

    def batch(rows: list[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.full((len(rows), width), pad)
        targets = torch.full((len(rows), width), IGNORE_INDEX)
        for index, (row_ids, row_labels) in enumerate(rows):
            inputs[index, : len(row_ids)] = torch.tensor(row_ids)
            targets[index, : len(row_labels)] = torch.tensor(row_labels)
        return inputs[:, :-1], targets[:, 1:]

    alone_inputs, alone_targets = batch([(ids, labels)])
    mixed_inputs, mixed_targets = batch([(ids, labels), (long_ids, long_labels)])
    with torch.no_grad():
        alone = causal_lm_loss_sum(model(alone_inputs), alone_targets)[0]
        mixed_logits = model(mixed_inputs)
        first = causal_lm_loss_sum(mixed_logits[:1], mixed_targets[:1])[0]
    assert torch.allclose(alone, first, atol=1e-5)


def test_repetition_penalty_ignores_prompt_tokens_by_default(engine, monkeypatch):
    seen: list[list[int]] = []
    original = engine._next_token

    def spy(logits, history, params, generator):
        seen.append(list(history))
        return original(logits, history, params, generator)

    monkeypatch.setattr(engine, "_next_token", spy)
    prompt = engine.tokenizer.encode("alpha beta gamma", add_bos=True)
    list(engine.stream_ids(prompt, SamplingParams(greedy=True, max_new_tokens=3)))
    assert seen[0] == []
    seen.clear()
    params = SamplingParams(greedy=True, max_new_tokens=3, penalize_prompt=True)
    list(engine.stream_ids(prompt, params))
    assert seen[0] == prompt


def test_chat_context_drops_whole_oldest_turns_and_keeps_latest_request(engine):
    long_history = [Message("system", "rules")]
    for index in range(12):
        long_history.append(Message("user", f"question number {index} about a topic"))
        long_history.append(Message("assistant", f"answer number {index} with some detail"))
    long_history.append(Message("user", "the final question"))
    ids = engine._chat_ids(long_history, 8)
    assert len(ids) <= engine.max_positions - 8
    assert ids[0] == engine.tokenizer.bos_id
    assert ids[1] == engine.tokenizer.role_ids["system"]
    assert ids[-1] == engine.tokenizer.role_ids["assistant"]
    text = engine.tokenizer.decode(ids)
    assert "the final question" in text
    assert "question number 0 " not in text
    user_roles = [i for i, t in enumerate(ids) if t == engine.tokenizer.role_ids["user"]]
    assistant_roles = [i for i, t in enumerate(ids) if t == engine.tokenizer.role_ids["assistant"]]
    assert len(assistant_roles) == len(user_roles)


def test_short_chat_is_not_modified(engine):
    ids = engine._chat_ids(HISTORY[:2], 4)
    assert ids == engine.tokenizer.encode_conversation(HISTORY[:2], add_generation_prompt=True)


def test_controlled_overfit_reproduces_training_answers_only(model_config, tokenizer):
    torch.manual_seed(0)
    network = CausalLM(model_config)
    pairs = [
        ("hi", "hello"),
        ("who are you", "a model"),
        ("2 + 2", "4"),
        ("color of sky", "blue"),
        ("say yes", "yes"),
        ("add 1 and 2", "3"),
    ]
    rows = [
        tokenizer.encode_conversation_with_labels(
            [Message("user", question), Message("assistant", answer)], IGNORE_INDEX
        )
        for question, answer in pairs
    ]
    width = max(len(ids) for ids, _ in rows)
    inputs = torch.full((len(rows), width), tokenizer.pad_id)
    targets = torch.full((len(rows), width), IGNORE_INDEX)
    for index, (ids, labels) in enumerate(rows):
        inputs[index, : len(ids)] = torch.tensor(ids)
        targets[index, : len(labels)] = torch.tensor(labels)
    optimizer = torch.optim.AdamW(network.parameters(), lr=5e-3)
    network.train()
    first = last = 0.0
    for step in range(400):
        loss_sum, count = causal_lm_loss_sum(network(inputs[:, :-1]), targets[:, 1:])
        loss = loss_sum / count
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        first = loss.item() if step == 0 else first
        last = loss.item()
    assert last < first * 0.05
    engine = InferenceEngine(network, tokenizer, CPU)
    params = SamplingParams(greedy=True, max_new_tokens=8)
    hits = sum(
        engine.chat([Message("user", question)], params).text.strip() == answer
        for question, answer in pairs
    )
    assert hits >= 5
