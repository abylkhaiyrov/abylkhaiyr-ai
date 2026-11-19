import math

import pytest
import torch

from llm.model.attention import stable_softmax
from llm.model.mlp import silu, swiglu
from llm.model.norm import rms_norm
from llm.model.rope import RotaryEmbedding
from llm.optimization.adamw import AdamW
from llm.optimization.grad import clip_grad_norm
from llm.optimization.schedule import learning_rate_at
from llm.training.losses import causal_lm_loss_sum


def test_stable_softmax_matches_torch_and_survives_large_values():
    x = torch.tensor([[1000.0, 1001.0, 1002.0], [-1000.0, 0.0, 1.0]])
    result = stable_softmax(x)
    assert torch.isfinite(result).all()
    assert torch.allclose(result, torch.softmax(x, dim=-1), atol=1e-6)
    assert torch.allclose(result.sum(-1), torch.ones(2))


def test_rms_norm_known_values():
    x = torch.tensor([[3.0, 4.0]])
    expected = x / math.sqrt((9 + 16) / 2)
    assert torch.allclose(rms_norm(x, torch.ones(2), 0.0), expected, atol=1e-6)
    assert torch.allclose(rms_norm(x, torch.tensor([2.0, 0.5]), 0.0), expected * torch.tensor([2.0, 0.5]))


def test_silu_and_swiglu_known_values():
    x = torch.tensor([0.0, 1.0, -1.0])
    assert torch.allclose(silu(x), torch.tensor([0.0, 0.7310586, -0.2689414]), atol=1e-6)
    assert torch.allclose(swiglu(x, torch.tensor([2.0, 2.0, 2.0])), silu(x) * 2)


def test_rope_position_zero_is_identity_and_preserves_norm():
    rope = RotaryEmbedding(head_dim=8, max_positions=16, theta=10000.0)
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    zero = torch.zeros(1, 5, dtype=torch.long)
    q0, k0 = rope(q, k, zero)
    assert torch.allclose(q0, q) and torch.allclose(k0, k)
    positions = torch.arange(5).unsqueeze(0)
    qr, kr = rope(q, k, positions)
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_rope_known_rotation():
    rope = RotaryEmbedding(head_dim=2, max_positions=4, theta=10000.0)
    q = torch.tensor([[[[1.0, 0.0]]]])
    out, _ = rope(q, q, torch.tensor([[1]]))
    assert torch.allclose(out, torch.tensor([[[[math.cos(1.0), math.sin(1.0)]]]]), atol=1e-6)


def test_rope_scores_depend_only_on_relative_distance():
    rope = RotaryEmbedding(head_dim=8, max_positions=32, theta=10000.0)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)

    def score(qp: int, kp: int) -> torch.Tensor:
        qr, _ = rope(q, q, torch.tensor([[qp]]))
        _, kr = rope(k, k, torch.tensor([[kp]]))
        return (qr * kr).sum()

    assert torch.allclose(score(5, 2), score(15, 12), atol=1e-4)


def test_cross_entropy_matches_manual_and_ignores_masked_tokens():
    logits = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    targets = torch.tensor([[0, 1, -100]])
    total, count = causal_lm_loss_sum(logits, targets)
    manual = -math.log(math.exp(2) / (math.exp(2) + math.e + 1)) - math.log(1 / 3)
    assert count == 2
    assert total.item() == pytest.approx(manual, rel=1e-5)


def test_adamw_matches_torch_reference():
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(4, 3))
    b = torch.nn.Parameter(a.detach().clone())
    mine = AdamW([a], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    ref = torch.optim.AdamW([b], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    for _ in range(10):
        grad = torch.randn(4, 3)
        a.grad, b.grad = grad.clone(), grad.clone()
        mine.step()
        ref.step()
    assert torch.allclose(a, b, atol=1e-6)


def test_learning_rate_schedule():
    args = dict(max_steps=100, base_lr=1.0, warmup_steps=10, min_lr_ratio=0.1)
    assert learning_rate_at(0, **args) == pytest.approx(0.1)
    assert learning_rate_at(9, **args) == pytest.approx(1.0)
    assert learning_rate_at(10, **args) == pytest.approx(1.0)
    assert learning_rate_at(55, **args) == pytest.approx(0.55, abs=1e-6)
    assert learning_rate_at(100, **args) == pytest.approx(0.1)


def test_gradient_clipping_scales_to_max_norm():
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([3.0, 4.0])
    norm = clip_grad_norm([p], 1.0)
    assert norm.item() == pytest.approx(5.0)
    assert p.grad.norm().item() == pytest.approx(1.0, abs=1e-4)
