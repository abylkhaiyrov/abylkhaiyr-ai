from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from llm.core.config import ModelConfig
from llm.model.attention import build_attention_mask, reference_attention, repeat_kv, sdpa_attention
from llm.model.mlp import swiglu
from llm.model.norm import rms_norm
from llm.model.rope import RotaryEmbedding
from llm.model.transformer import CausalLM, segment_mask, segment_positions
from llm.training.losses import IGNORE_INDEX, causal_lm_loss_sum, dpo_loss

DOUBLE = torch.float64


def test_rope_scores_depend_only_on_relative_distance():
    rotary = RotaryEmbedding(8, 64, 10000.0)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)

    def score(query_position: int, key_position: int) -> float:
        rq, _ = rotary(q, k, torch.tensor([[query_position]]))
        _, rk = rotary(q, k, torch.tensor([[key_position]]))
        return float((rq * rk).sum())

    assert score(10, 7) == pytest.approx(score(40, 37), abs=1e-4)
    assert score(10, 7) != pytest.approx(score(10, 6), abs=1e-4)


def test_rope_offset_equals_slicing_the_full_rotation():
    rotary = RotaryEmbedding(8, 32, 10000.0)
    q = torch.randn(2, 3, 12, 8)
    k = torch.randn(2, 3, 12, 8)
    full_q, full_k = rotary(q, k, torch.arange(12).unsqueeze(0).expand(2, -1))
    part_q, part_k = rotary(
        q[:, :, 5:], k[:, :, 5:], torch.arange(5, 12).unsqueeze(0).expand(2, -1)
    )
    assert torch.allclose(full_q[:, :, 5:], part_q, atol=1e-6)
    assert torch.allclose(full_k[:, :, 5:], part_k, atol=1e-6)


def test_rope_preserves_vector_norms():
    rotary = RotaryEmbedding(16, 32, 10000.0)
    q = torch.randn(1, 2, 9, 16)
    rq, _ = rotary(q, q, torch.arange(9).unsqueeze(0))
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_rms_norm_gradients_match_torch_reference():
    x = torch.randn(3, 5, 16, dtype=DOUBLE, requires_grad=True)
    weight = torch.randn(16, dtype=DOUBLE, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    rms_norm(x, weight, 1e-6).pow(2).sum().backward()
    functional.rms_norm(reference_x, (16,), reference_weight, eps=1e-6).pow(2).sum().backward()
    assert x.grad is not None and weight.grad is not None
    assert torch.allclose(x.grad, reference_x.grad, atol=1e-10)
    assert torch.allclose(weight.grad, reference_weight.grad, atol=1e-10)


def test_swiglu_matches_the_closed_form_and_gradcheck():
    gate = torch.randn(4, 6, dtype=DOUBLE, requires_grad=True)
    value = torch.randn(4, 6, dtype=DOUBLE, requires_grad=True)
    assert torch.allclose(swiglu(gate, value), gate / (1 + torch.exp(-gate)) * value)
    assert torch.autograd.gradcheck(swiglu, (gate, value))


def test_causal_lm_loss_passes_gradcheck_and_ignores_masked_targets():
    logits = torch.randn(2, 5, 7, dtype=DOUBLE, requires_grad=True)
    targets = torch.tensor([[1, 2, IGNORE_INDEX, 3, 4], [IGNORE_INDEX, 0, 6, 5, 1]])
    assert torch.autograd.gradcheck(lambda x: causal_lm_loss_sum(x, targets)[0], (logits,))
    total, count = causal_lm_loss_sum(logits, targets)
    total.backward()
    assert count == 8
    assert logits.grad is not None
    assert float(logits.grad[0, 2].abs().sum()) == 0.0


def test_dpo_loss_equals_log_two_when_policy_equals_reference():
    values = torch.randn(6, dtype=DOUBLE)
    assert dpo_loss(values, values + 1, values, values + 1, 0.1).item() == pytest.approx(
        torch.log(torch.tensor(2.0, dtype=DOUBLE)).item()
    )


@pytest.mark.parametrize("heads_per_group", [1, 2, 3])
def test_native_gqa_equals_repeated_keys_and_values_with_any_mask(heads_per_group):
    batch, kv_heads, length, dim = 2, 2, 7, 8
    q = torch.randn(batch, kv_heads * heads_per_group, length, dim, dtype=DOUBLE)
    k = torch.randn(batch, kv_heads, length, dim, dtype=DOUBLE)
    v = torch.randn(batch, kv_heads, length, dim, dtype=DOUBLE)
    padding = torch.ones(batch, length, dtype=torch.bool)
    padding[1, :2] = False
    mask = build_attention_mask(length, length, torch.device("cpu"), padding)
    repeated = sdpa_attention(q, repeat_kv(k, heads_per_group), repeat_kv(v, heads_per_group), mask)
    native = sdpa_attention(q, k, v, mask, enable_gqa=True)
    expected = reference_attention(
        q, repeat_kv(k, heads_per_group), repeat_kv(v, heads_per_group), mask
    )
    visible = mask.any(dim=-1, keepdim=True)
    assert torch.allclose(
        native.masked_select(visible), repeated.masked_select(visible), atol=1e-10
    )
    assert torch.allclose(
        repeated.masked_select(visible), expected.masked_select(visible), atol=1e-10
    )


def test_repeat_kv_does_not_alias_distinct_groups():
    x = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).view(2, 2, 3, 4)
    repeated = repeat_kv(x, 3)
    assert repeated.shape == (2, 6, 3, 4)
    assert torch.equal(repeated[:, 0], repeated[:, 2]) and not torch.equal(
        repeated[:, 2], repeated[:, 3]
    )


def test_segment_positions_restart_and_masks_isolate_packed_documents():
    segments = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]])
    assert segment_positions(segments).tolist() == [[0, 1, 2, 0, 1, 0, 1, 2]]
    mask = segment_mask(segments)[0, 0]
    assert not mask[3, :3].any() and mask[3, 3] and not mask[5, :5].any()
    assert torch.equal(mask, mask.tril())


def test_tied_embeddings_share_one_parameter_and_accumulate_both_gradients(model_config):
    model = CausalLM(model_config)
    assert model.lm_head.weight is model.embedding.weight
    names = [name for name, _ in model.named_parameters()]
    assert "lm_head.weight" not in names
    ids = torch.randint(0, model_config.vocab_size, (2, 8))
    model(ids).pow(2).mean().backward()
    untied = CausalLM(model_config.model_copy(update={"tie_word_embeddings": False}))
    assert untied.num_parameters() - model.num_parameters() == (
        model_config.vocab_size * model_config.hidden_size
    )
    assert model.embedding.weight.grad is not None


def test_initialization_scales_residual_projections_by_depth(model_config):
    torch.manual_seed(0)
    config = model_config.model_copy(
        update={"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 4}
    )
    model = CausalLM(config)
    plain = model.blocks[0].attention.q_proj.weight.std().item()
    residual = model.blocks[0].attention.o_proj.weight.std().item()
    assert plain == pytest.approx(config.initializer_range, rel=0.1)
    assert residual == pytest.approx(config.initializer_range / (2 * 4) ** 0.5, rel=0.1)


def test_forward_is_deterministic_for_fixed_weights_and_inputs(model, model_config):
    model.eval()
    ids = torch.randint(0, model_config.vocab_size, (2, 10))
    with torch.no_grad():
        assert torch.equal(model(ids), model(ids))


@pytest.mark.parametrize("fast_causal", [False, True])
def test_token_by_token_decoding_matches_full_sequence_logits(
    model_config: ModelConfig, fast_causal
):
    torch.manual_seed(1)
    model = CausalLM(model_config).eval()
    model.configure_attention(native_gqa=fast_causal, fast_causal=fast_causal)
    ids = torch.randint(0, model_config.vocab_size, (2, 9))
    with torch.no_grad():
        full = model(ids)
        cache = model.new_cache()
        steps = [model(ids[:, i : i + 1], cache=cache) for i in range(9)]
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-4)


def test_float32_model_matches_float64_model_within_tolerance(model_config):
    torch.manual_seed(2)
    single = CausalLM(model_config).eval()
    double = CausalLM(model_config).double().eval()
    double.load_state_dict({name: value.double() for name, value in single.state_dict().items()})
    ids = torch.randint(0, model_config.vocab_size, (2, 12))
    with torch.no_grad():
        assert torch.allclose(single(ids).double(), double(ids), atol=1e-4)
