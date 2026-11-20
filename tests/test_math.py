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
    assert torch.allclose(
        rms_norm(x, torch.tensor([2.0, 0.5]), 0.0), expected * torch.tensor([2.0, 0.5])
    )


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
    qr, _kr = rope(q, k, positions)
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
    args = {"max_steps": 100, "base_lr": 1.0, "warmup_steps": 10, "min_lr_ratio": 0.1}
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


@pytest.mark.parametrize("foreach", [True, False])
def test_adamw_multi_tensor_paths_match_torch_over_many_steps(foreach):
    torch.manual_seed(0)
    shapes = [(5, 3), (4,), (2, 2, 2)]
    ours = [torch.randn(*shape, dtype=torch.float64, requires_grad=True) for shape in shapes]
    theirs = [p.detach().clone().requires_grad_(True) for p in ours]
    own = AdamW(
        [{"params": ours[:2], "weight_decay": 0.1}, {"params": ours[2:], "weight_decay": 0.0}],
        lr=1e-2,
        foreach=foreach,
    )
    ref = torch.optim.AdamW(
        [{"params": theirs[:2], "weight_decay": 0.1}, {"params": theirs[2:], "weight_decay": 0.0}],
        lr=1e-2,
    )
    for step in range(20):
        for left, right in zip(ours, theirs, strict=True):
            grad = torch.randn_like(left)
            left.grad, right.grad = grad.clone(), grad.clone()
        ours[1].grad = None if step % 4 == 3 else ours[1].grad
        theirs[1].grad = None if step % 4 == 3 else theirs[1].grad
        own.step()
        ref.step()
    for left, right in zip(ours, theirs, strict=True):
        assert torch.allclose(left, right, atol=1e-10)


def test_rms_norm_matches_torch_reference_in_float64():
    x = torch.randn(3, 7, 16, dtype=torch.float64)
    weight = torch.randn(16, dtype=torch.float64)
    expected = torch.nn.functional.rms_norm(x, (16,), weight, eps=1e-6)
    assert torch.allclose(rms_norm(x, weight, 1e-6), expected, atol=1e-12)


def test_rope_matches_complex_rotation_reference():
    head_dim, positions = 8, 6
    rotary = RotaryEmbedding(head_dim, positions, 10000.0)
    q = torch.randn(1, 2, positions, head_dim)
    k = torch.randn(1, 2, positions, head_dim)
    rq, rk = rotary(q, k, torch.arange(positions).unsqueeze(0))
    frequencies = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(positions).float(), frequencies)
    rotation = torch.polar(torch.ones_like(angles), angles)

    def reference(x: torch.Tensor) -> torch.Tensor:
        half = head_dim // 2
        complex_x = torch.complex(x[..., :half], x[..., half:])
        rotated = complex_x * rotation
        return torch.cat((rotated.real, rotated.imag), dim=-1)

    assert torch.allclose(rq, reference(q), atol=1e-5)
    assert torch.allclose(rk, reference(k), atol=1e-5)


def test_gradient_clipping_matches_torch_utility():
    torch.manual_seed(0)
    mine = [torch.nn.Parameter(torch.randn(5, 4)) for _ in range(3)]
    theirs = [torch.nn.Parameter(p.detach().clone()) for p in mine]
    for left, right in zip(mine, theirs, strict=True):
        grad = torch.randn_like(left) * 5
        left.grad, right.grad = grad.clone(), grad.clone()
    norm = clip_grad_norm(mine, 1.0)
    reference = torch.nn.utils.clip_grad_norm_(theirs, 1.0)
    assert torch.allclose(norm, reference)
    for left, right in zip(mine, theirs, strict=True):
        assert torch.allclose(left.grad, right.grad, atol=1e-5)


def test_dpo_loss_matches_closed_form_and_has_correct_gradient_signs():
    from llm.training.losses import dpo_loss

    chosen = torch.tensor([-1.0, -2.0], dtype=torch.float64, requires_grad=True)
    rejected = torch.tensor([-3.0, -1.5], dtype=torch.float64, requires_grad=True)
    ref_chosen = torch.tensor([-1.5, -2.0], dtype=torch.float64)
    ref_rejected = torch.tensor([-2.5, -2.0], dtype=torch.float64)
    beta = 0.3
    margin = (chosen - rejected) - (ref_chosen - ref_rejected)
    expected = torch.log1p(torch.exp(-beta * margin)).mean()
    loss = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert (chosen.grad < 0).all() and (rejected.grad > 0).all()
    assert torch.autograd.gradcheck(
        lambda c, r: dpo_loss(c, r, ref_chosen, ref_rejected, beta), (chosen, rejected)
    )


def test_sequence_log_probs_ignore_masked_positions_and_match_manual_sum():
    from llm.training.losses import IGNORE_INDEX, sequence_log_probs

    logits = torch.randn(2, 4, 6, dtype=torch.float64)
    targets = torch.tensor([[1, 2, IGNORE_INDEX, 3], [IGNORE_INDEX, IGNORE_INDEX, 5, 0]])
    log_probs = torch.log_softmax(logits, dim=-1)
    manual = torch.stack(
        [
            log_probs[0, 0, 1] + log_probs[0, 1, 2] + log_probs[0, 3, 3],
            log_probs[1, 2, 5] + log_probs[1, 3, 0],
        ]
    )
    assert torch.allclose(sequence_log_probs(logits, targets), manual)


def test_sft_loss_is_invariant_to_how_a_step_is_split_into_micro_batches():
    logits = torch.randn(4, 5, 7, dtype=torch.float64)
    targets = torch.randint(0, 7, (4, 5))
    targets[0, :3] = -100
    full, count = causal_lm_loss_sum(logits, targets)
    parts = [causal_lm_loss_sum(logits[i : i + 2], targets[i : i + 2]) for i in (0, 2)]
    assert torch.allclose(full, sum(part[0] for part in parts))
    assert count == sum(part[1] for part in parts)
