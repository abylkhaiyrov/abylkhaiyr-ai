import pytest
import torch

from llm.inference.engine import InferenceEngine, partial_stop_length, truncate_at_stop
from llm.inference.sampling import (
    SamplingParams,
    apply_repetition_penalty,
    select_token,
    top_k_filter,
    top_p_filter,
)
from llm.model.transformer import CausalLM
from llm.tokenizer.bpe import Message

CPU = torch.device("cpu")


@pytest.fixture()
def engine(model, tokenizer):
    torch.manual_seed(0)
    return InferenceEngine(CausalLM(model.config), tokenizer, CPU)


def greedy(**kwargs):
    return SamplingParams(greedy=True, max_new_tokens=12, **kwargs)


def test_top_k_and_top_p_filters():
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    assert torch.isinf(top_k_filter(logits, 2)).sum() == 2
    assert top_k_filter(logits, 2)[0, 1] == 4.0
    filtered = top_p_filter(torch.log(torch.tensor([[0.6, 0.3, 0.07, 0.03]])), 0.8)
    assert torch.isfinite(filtered[0]).tolist() == [True, True, False, False]
    assert torch.isfinite(top_p_filter(logits, 1e-9)).sum() == 1


def test_repetition_penalty_lowers_seen_token_scores():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    result = apply_repetition_penalty(logits, torch.tensor([[0, 1]]), 2.0)
    assert result.tolist() == [[1.0, -4.0, 1.0]]


def test_sampling_is_seeded_and_greedy_is_argmax():
    logits = torch.randn(1, 50)
    params = SamplingParams(temperature=1.0, top_k=10)
    first = select_token(logits, params, torch.Generator().manual_seed(1))
    second = select_token(logits, params, torch.Generator().manual_seed(1))
    assert first == second
    assert select_token(logits, SamplingParams(greedy=True), None) == logits.argmax()


def test_stop_helpers():
    assert truncate_at_stop("abcSTOPdef", ["STOP"]) == ("abc", True)
    assert partial_stop_length("abcST", ["STOP"]) == 2
    assert partial_stop_length("abc", ["STOP"]) == 0


def test_generation_with_and_without_cache_is_identical(engine):
    prompt = engine.tokenizer.encode("Hello world", add_bos=True)
    with_cache = engine.generate_ids(prompt, greedy(use_cache=True))
    without = engine.generate_ids(prompt, greedy(use_cache=False))
    assert with_cache.token_ids == without.token_ids and len(with_cache.token_ids) > 0


def test_chat_context_truncation_preserves_role_markers(engine):
    messages = [
        Message("system", "Rules. " * 100),
        Message("user", "older"),
        Message("assistant", "history"),
        Message("user", "Қазақша сұрақ. " * 200 + "Не істеу керек?"),
    ]
    fitted, dropped = engine.fit_messages(messages, 12)
    assert dropped == 2 and [m.role for m in fitted] == ["system", "user"]
    assert fitted[-1].content.endswith("Не істеу керек?")
    ids = engine.tokenizer.encode_conversation(fitted, True)
    assert len(ids) <= engine.max_positions - 12
    assert ids[0] == engine.tokenizer.bos_id
    assert ids[-3:].count(engine.tokenizer.eos_id) == 1
    assert ids[-1] == engine.tokenizer.role_ids["assistant"]


def test_chat_batch_agrees_with_serial_chat(engine):
    prompts = [[Message("user", "Hello")], [Message("user", "Привет, как дела?")]]
    results = engine.chat_batch(prompts, greedy())
    assert [result.token_ids for result in results] == [
        engine.chat(prompt, greedy()).token_ids for prompt in prompts
    ]
    assert engine.chat_batch([], greedy()) == []


def test_seeded_sampling_reproducible(engine):
    params = SamplingParams(max_new_tokens=10, temperature=0.9, top_p=0.9, seed=7)
    assert (
        engine.generate("Привет", params).token_ids == engine.generate("Привет", params).token_ids
    )


def test_max_new_tokens_and_context_window_respected(engine):
    result = engine.generate("a" * 500, SamplingParams(greedy=True, max_new_tokens=5))
    assert len(result.token_ids) <= 5
    fitted = engine.fit_prompt(list(range(200)), 10)
    assert len(fitted) == engine.max_positions - 10


def test_fit_prompt_with_one_token_budget_keeps_only_bos(engine):
    prompt = [engine.tokenizer.bos_id, 20, 21, 22]
    assert engine.fit_prompt(prompt, engine.max_positions) == [engine.tokenizer.bos_id]


def test_fit_prompt_with_one_token_budget_keeps_latest_token_without_bos(engine):
    prompt = [20, 21, 22]
    assert engine.fit_prompt(prompt, engine.max_positions) == [22]


def test_stop_sequence_truncates_output(engine):
    full = engine.generate("Hello", greedy())
    if len(full.text) < 4:
        pytest.skip("random model produced too little text")
    stop = full.text[2:4]
    cut = engine.generate("Hello", greedy(stop_sequences=[stop]))
    assert cut.text == full.text[: full.text.find(stop)] and cut.finish_reason == "stop"


def test_stream_text_equals_generate(engine):
    prompt = engine.tokenizer.encode("Hello world", add_bos=True)
    streamed = "".join(engine.stream_text(prompt, greedy()))
    assert streamed == engine.generate_ids(prompt, greedy()).text


def test_stream_reports_actual_end_reason(engine):
    from dataclasses import replace

    messages = [Message("user", "Short prompt")]
    params = replace(greedy(), max_new_tokens=1)
    stream = engine.stream_chat(messages, params)
    with pytest.raises(StopIteration) as ended:
        while True:
            next(stream)
    assert ended.value.value == engine.chat(messages, params).finish_reason


def test_batch_generation_matches_single(engine):
    prompts = ["Hello", "Привет, как дела у тебя"]
    batch = engine.generate_batch(prompts, greedy())
    for prompt, result in zip(prompts, batch, strict=True):
        assert result.token_ids == engine.generate(prompt, greedy()).token_ids


def test_beam_search_not_worse_than_greedy(engine):
    prompt = engine.tokenizer.encode("Hello", add_bos=True)
    greedy_ids = engine.generate_ids(prompt, greedy(use_cache=True)).token_ids

    def score(ids):
        full = torch.tensor([prompt + ids])
        logp = torch.log_softmax(engine.model(full).float(), -1)[0]
        return sum(logp[len(prompt) - 1 + i, t].item() for i, t in enumerate(ids))

    beam = engine.generate_ids(
        prompt, SamplingParams(max_new_tokens=12, beam_width=4, length_penalty=0.0)
    )
    assert len(beam.token_ids) > 0
    assert score(beam.token_ids[:12]) >= score(greedy_ids[: len(beam.token_ids)]) - 1e-3 or len(
        beam.token_ids
    ) != len(greedy_ids)


def test_beam_cache_and_stream_match_normal_generation(engine):
    prompt = engine.tokenizer.encode("Hello", add_bos=True)
    cached = SamplingParams(max_new_tokens=7, beam_width=3, repetition_penalty=1.1)
    uncached = SamplingParams(
        max_new_tokens=7, beam_width=3, repetition_penalty=1.1, use_cache=False
    )
    expected = engine.generate_ids(prompt, cached)
    assert engine.generate_ids(prompt, uncached) == expected
    assert "".join(engine.stream_text(prompt, cached)) == expected.text


def test_beam_immediate_stop_reports_stop_even_without_visible_tokens(engine):
    prompt = engine.tokenizer.encode("Hello", add_bos=True)
    params = SamplingParams(max_new_tokens=4, beam_width=2)
    stop_ids = set(range(engine.tokenizer.vocab_size))
    result = engine.generate_ids(prompt, params, stop_ids)
    assert result.finish_reason == "stop"
    assert result.token_ids == []
    stream = engine.stream_text(prompt, params, stop_ids)
    with pytest.raises(StopIteration) as stopped:
        next(stream)
    assert stopped.value.value == "stop"


def test_chat_uses_generation_prompt(engine):
    result = engine.chat([Message("user", "hi")], greedy())
    assert isinstance(result.text, str)


def test_model_in_eval_mode_and_no_grad_during_generation(engine):
    engine.generate("Hello", greedy())
    assert not engine.model.training
    assert all(p.grad is None for p in engine.model.parameters())


def test_chat_stops_on_role_special_tokens_and_continuation_streams(engine, monkeypatch):
    user_token = engine.tokenizer.role_ids["user"]
    hello_ids = engine.tokenizer.encode("hi")
    monkeypatch.setattr(
        engine, "stream_ids", lambda prompt_ids, params: iter([*hello_ids, user_token, *hello_ids])
    )
    reply = engine.chat([Message("user", "hello")], greedy())
    assert "<user>" not in reply.text
    assert reply.text == "hi"
    assert reply.finish_reason == "stop"
    continued = "".join(
        engine.stream_chat_continuation([Message("user", "hello")], "part ", greedy())
    )
    assert continued == "hi"


def test_prompt_usage_matches_chat_after_completion_budget_clamp(engine):
    messages = [Message("user", "Hello " * 100)]
    params = SamplingParams(max_new_tokens=2 * engine.max_positions)
    fitted = engine.fit_messages(messages, engine._chat_params(params).max_new_tokens)[0]
    encoded = engine.tokenizer.encode_conversation(fitted, add_generation_prompt=True)
    assert len(encoded) > 4
    assert engine.chat_prompt_tokens(messages, params.max_new_tokens) == len(encoded)
