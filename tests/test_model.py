import pytest
import torch
from pydantic import ValidationError

from llm.core.config import ModelConfig
from llm.core.profiles import PROFILES, build_model_config, count_parameters
from llm.model.attention import (
    build_attention_mask,
    reference_attention,
    repeat_kv,
    sdpa_attention,
)
from llm.model.transformer import CausalLM
from llm.training.checkpoint import load_model_weights, save_model_weights


def test_config_rejects_incompatible_parameters():
    base = {
        "vocab_size": 10,
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "max_position_embeddings": 16,
    }
    ModelConfig(**base)
    with pytest.raises(ValidationError):
        ModelConfig(**{**base, "hidden_size": 30})
    with pytest.raises(ValidationError):
        ModelConfig(**{**base, "num_key_value_heads": 3})
    with pytest.raises(ValidationError):
        ModelConfig(**{**base, "hidden_size": 12, "num_attention_heads": 4})


def test_causal_mask_is_lower_triangular_with_offset():
    mask = build_attention_mask(3, 3, torch.device("cpu"))[0, 0]
    assert torch.equal(mask, torch.tril(torch.ones(3, 3, dtype=torch.bool)))
    cached = build_attention_mask(1, 4, torch.device("cpu"))[0, 0]
    assert cached.all()


def test_sdpa_matches_reference_attention():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
    mask = build_attention_mask(6, 6, torch.device("cpu"))
    assert torch.allclose(
        reference_attention(q, k, v, mask), sdpa_attention(q, k, v, mask), atol=1e-5
    )


def test_reference_attention_gradcheck():
    q, k, v = (torch.randn(1, 2, 4, 4, dtype=torch.double, requires_grad=True) for _ in range(3))
    mask = build_attention_mask(4, 4, torch.device("cpu"))
    assert torch.autograd.gradcheck(lambda a, b, c: reference_attention(a, b, c, mask), (q, k, v))


def test_repeat_kv_expands_groups_in_order():
    x = torch.arange(2 * 2 * 1 * 1, dtype=torch.float).view(1, 2, 2, 1)
    out = repeat_kv(x, 2)
    assert out.shape == (1, 4, 2, 1)
    assert torch.equal(out[:, 0], out[:, 1]) and torch.equal(out[:, 2], out[:, 3])


def test_forward_shape_and_weight_tying(model, model_config):
    ids = torch.randint(0, model_config.vocab_size, (3, 10))
    assert model(ids).shape == (3, 10, model_config.vocab_size)
    assert model.lm_head.weight is model.embedding.weight


def test_parameter_count_formula_matches_model(model, model_config):
    assert model.num_parameters() == count_parameters(model_config)


def test_profile_configs_are_valid_and_in_range():
    for name, profile in PROFILES.items():
        config = build_model_config(name, vocab_size=8192)
        assert profile.min_params * 0.4 <= count_parameters(config) <= profile.max_params * 1.5


def test_no_information_leak_from_future_tokens(model, model_config):
    torch.manual_seed(1)
    a = torch.randint(0, model_config.vocab_size, (1, 12))
    b = a.clone()
    b[:, 7:] = torch.randint(0, model_config.vocab_size, (1, 5))
    with torch.no_grad():
        la, lb = model(a), model(b)
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-5)


def test_reference_and_sdpa_models_agree(model_config):
    torch.manual_seed(0)
    fast = CausalLM(model_config).eval()
    slow = CausalLM(model_config, use_reference_attention=True).eval()
    slow.load_state_dict(fast.state_dict())
    ids = torch.randint(0, model_config.vocab_size, (2, 9))
    with torch.no_grad():
        assert torch.allclose(fast(ids), slow(ids), atol=1e-4)


def test_kv_cache_matches_full_forward(model, model_config):
    ids = torch.randint(0, model_config.vocab_size, (2, 12))
    with torch.no_grad():
        full = model(ids)
        cache = model.new_cache()
        parts = [model(ids[:, :5], cache=cache)]
        parts += [model(ids[:, i : i + 1], cache=cache) for i in range(5, 12)]
    assert torch.allclose(full, torch.cat(parts, dim=1), atol=1e-4)


def test_gradients_flow_to_every_parameter(model, model_config):
    model.train()
    ids = torch.randint(0, model_config.vocab_size, (2, 8))
    model(ids).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_activation_checkpointing_gives_same_gradients(model_config):
    torch.manual_seed(0)
    plain = CausalLM(model_config).train()
    checkpointed = CausalLM(model_config).train()
    checkpointed.load_state_dict(plain.state_dict())
    checkpointed.activation_checkpointing = True
    ids = torch.randint(0, model_config.vocab_size, (2, 8))
    plain(ids).square().mean().backward()
    checkpointed(ids).square().mean().backward()
    for a, b in zip(plain.parameters(), checkpointed.parameters(), strict=True):
        assert torch.allclose(a.grad, b.grad, atol=1e-5)


def test_save_and_load_roundtrip(model, model_config, tmp_path):
    path = tmp_path / "model.safetensors"
    save_model_weights(model, path)
    clone = CausalLM(model_config).eval()
    load_model_weights(clone, path)
    ids = torch.randint(0, model_config.vocab_size, (1, 6))
    with torch.no_grad():
        assert torch.allclose(model(ids), clone(ids))


def test_native_gqa_and_causal_fast_path_match_reference_forward_and_gradients(model_config):
    import copy

    torch.manual_seed(1)
    base = CausalLM(model_config)
    fast = copy.deepcopy(base)
    fast.configure_attention(native_gqa=True, fast_causal=True)
    ids = torch.randint(0, model_config.vocab_size, (3, 24))
    assert torch.allclose(base(ids), fast(ids), atol=1e-5)
    base(ids).square().mean().backward()
    fast(ids).square().mean().backward()
    for (name, left), (_, right) in zip(
        base.named_parameters(), fast.named_parameters(), strict=True
    ):
        assert torch.allclose(left.grad, right.grad, atol=1e-5), name


def test_fast_path_is_not_used_with_padding_or_cache(model_config):
    torch.manual_seed(2)
    model = CausalLM(model_config).eval()
    model.configure_attention(native_gqa=True, fast_causal=True)
    ids = torch.randint(0, model_config.vocab_size, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.long)
    mask[0, :3] = 0
    padded = model(ids, attention_mask=mask)
    plain = CausalLM(model_config).eval()
    plain.load_state_dict(model.state_dict())
    assert torch.allclose(padded, plain(ids, attention_mask=mask), atol=1e-5)
    cache = model.new_cache()
    full = model(ids)
    step = [model(ids[:, i : i + 1], cache=cache) for i in range(10)]
    assert torch.allclose(full, torch.cat(step, dim=1), atol=1e-4)


def test_preallocated_cache_is_stable_under_growth_truncate_and_reorder(model_config):
    from llm.model.cache import LayerKVCache

    cache = LayerKVCache(capacity=4)
    keys = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).view(2, 2, 3, 4)
    first_keys, _ = cache.update(keys, keys + 100)
    assert cache.length == 3 and torch.equal(first_keys, keys)
    more = torch.full((2, 2, 3, 4), -1.0)
    grown_keys, grown_values = cache.update(more, more)
    assert cache.length == 6 and grown_keys.shape[2] == 6
    assert torch.equal(grown_keys[:, :, :3], keys) and torch.equal(
        grown_values[:, :, :3], keys + 100
    )
    cache.truncate(2)
    assert cache.length == 2
    resumed, _ = cache.update(more[:, :, :1], more[:, :, :1])
    assert resumed.shape[2] == 3 and torch.equal(resumed[:, :, :2], keys[:, :, :2])
    cache.reorder(torch.tensor([1, 0]))
    assert torch.equal(cache.keys[0, :, :2], keys[1, :, :2])


def test_packed_segments_equal_separate_sequences_in_loss_and_isolation(model_config):
    from llm.model.transformer import segment_positions
    from llm.training.losses import causal_lm_loss_sum

    torch.manual_seed(3)
    model = CausalLM(model_config)
    first = torch.randint(1, model_config.vocab_size, (1, 9))
    second = torch.randint(1, model_config.vocab_size, (1, 7))
    packed = torch.cat([first, second, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    segments = torch.tensor([[1] * 9 + [2] * 7 + [0] * 4])
    assert segment_positions(segments).tolist() == [list(range(9)) + list(range(7)) + [0, 1, 2, 3]]
    logits = model(packed, segment_ids=segments)
    assert torch.allclose(logits[:, :9], model(first), atol=1e-5)
    assert torch.allclose(logits[:, 9:16], model(second), atol=1e-5)
    targets = torch.randint(0, model_config.vocab_size, packed.shape)
    targets[segments == 0] = -100
    targets[:, 8] = -100
    packed_loss, _ = causal_lm_loss_sum(logits, targets)
    separate = (
        causal_lm_loss_sum(model(first), targets[:, :9])[0]
        + causal_lm_loss_sum(model(second), targets[:, 9:16])[0]
    )
    assert torch.allclose(packed_loss, separate, atol=1e-4)
    changed = packed.clone()
    changed[0, :9] = torch.randint(1, model_config.vocab_size, (9,))
    assert torch.allclose(model(changed, segment_ids=segments)[:, 9:16], logits[:, 9:16], atol=1e-5)


def test_fully_masked_attention_rows_match_sdpa_outputs_and_gradients():
    q, k, v = [torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    mask = torch.tensor([[False, False, False], [True, False, False], [True, True, False]])
    actual = reference_attention(q, k, v, mask)
    expected = sdpa_attention(q, k, v, mask)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.square().sum(), (q, k, v))
    expected_grads = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_packed_decoder_forward_and_backward_agree_with_cpu(model_config, device):
    available = torch.backends.mps.is_available() if device == "mps" else torch.cuda.is_available()
    if not available:
        pytest.skip(f"{device} is unavailable")
    cpu = CausalLM(model_config)
    accelerated = CausalLM(model_config).to(device)
    accelerated.load_state_dict(cpu.state_dict())
    ids = torch.tensor([[7, 8, 9, 10, 11], [12, 13, 14, 15, 16]])
    segments = torch.tensor([[0, 0, 0, 1, 1], [0, 1, 1, 2, 2]])
    reference = cpu(ids, segment_ids=segments)
    actual = accelerated(ids.to(device), segment_ids=segments.to(device))
    torch.testing.assert_close(actual.cpu(), reference, rtol=5e-4, atol=2e-5)
    reference.square().mean().backward()
    actual.square().mean().backward()
    for expected, observed in zip(cpu.parameters(), accelerated.parameters(), strict=True):
        torch.testing.assert_close(observed.grad.cpu(), expected.grad, rtol=2e-3, atol=2e-5)
