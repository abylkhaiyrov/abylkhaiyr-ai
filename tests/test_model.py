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
