from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from llm.optimization.adamw import AdamW
from llm.training.checkpoint import (
    CheckpointError,
    align_optimizer_state,
    optimizer_layout,
)


class Pair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4, bias=True)
        self.second = nn.Linear(4, 4, bias=True)


def build(model: Pair, swapped: bool) -> AdamW:
    decayed = [model.first.weight, model.second.weight]
    plain = [model.first.bias, model.second.bias]
    groups = [
        {"params": decayed, "weight_decay": 0.1},
        {"params": plain, "weight_decay": 0.0},
    ]
    return AdamW(groups[::-1] if swapped else groups, lr=1e-2)


def trained(swapped: bool) -> tuple[Pair, AdamW]:
    torch.manual_seed(0)
    model = Pair()
    optimizer = build(model, swapped)
    for _ in range(3):
        model.zero_grad()
        sum(p.pow(2).sum() * (index + 1) for index, p in enumerate(model.parameters())).backward()
        optimizer.step()
    return model, optimizer


@pytest.mark.parametrize("use_layout", [True, False])
def test_equal_size_groups_in_a_different_order_receive_their_own_moments(use_layout):
    model, saved_optimizer = trained(swapped=True)
    saved = saved_optimizer.state_dict()
    layout = optimizer_layout(model, saved_optimizer)
    expected = {
        name: saved_optimizer.state[p]["exp_avg"].clone() for name, p in model.named_parameters()
    }
    current = build(model, swapped=False)
    aligned = align_optimizer_state(
        saved, current, model if use_layout else None, layout if use_layout else None
    )
    current.load_state_dict(aligned)
    for name, parameter in model.named_parameters():
        assert torch.equal(current.state[parameter]["exp_avg"], expected[name])


def test_layout_remaps_parameters_even_when_shapes_are_identical():
    model, saved_optimizer = trained(swapped=False)
    layout = optimizer_layout(model, saved_optimizer)
    saved = saved_optimizer.state_dict()
    current = AdamW(
        [
            {"params": [model.second.weight, model.first.weight], "weight_decay": 0.1},
            {"params": [model.second.bias, model.first.bias], "weight_decay": 0.0},
        ],
        lr=1e-2,
    )
    current.load_state_dict(align_optimizer_state(saved, current, model, layout))
    for parameter in model.parameters():
        assert torch.equal(
            current.state[parameter]["exp_avg_sq"], saved_optimizer.state[parameter]["exp_avg_sq"]
        )


def test_shape_mismatch_without_layout_is_rejected():
    _model, saved_optimizer = trained(swapped=False)
    saved = saved_optimizer.state_dict()
    other = nn.Linear(3, 5)
    current = AdamW([{"params": [other.weight, other.bias]}], lr=1e-2)
    with pytest.raises(CheckpointError):
        align_optimizer_state(saved, current)


def test_unknown_parameter_name_is_rejected():
    model, saved_optimizer = trained(swapped=False)
    layout = optimizer_layout(model, saved_optimizer)
    saved = saved_optimizer.state_dict()
    layout[0][0] = "removed.weight"
    with pytest.raises(CheckpointError):
        align_optimizer_state(saved, build(model, swapped=False), model, layout)


def test_resumed_optimizer_continues_exactly_like_the_uninterrupted_one():
    model, optimizer = trained(swapped=False)
    clone = Pair()
    clone.load_state_dict(model.state_dict())
    resumed = build(clone, swapped=False)
    resumed.load_state_dict(
        align_optimizer_state(
            copy.deepcopy(optimizer.state_dict()),
            resumed,
            clone,
            optimizer_layout(model, optimizer),
        )
    )
    for _ in range(3):
        for subject, target in ((model, optimizer), (clone, resumed)):
            subject.zero_grad()
            sum(p.pow(2).sum() for p in subject.parameters()).backward()
            target.step()
    for left, right in zip(model.parameters(), clone.parameters(), strict=True):
        assert torch.equal(left, right)


def test_frozen_parameters_keep_no_state_and_do_not_move():
    torch.manual_seed(0)
    model = Pair()
    model.first.weight.requires_grad_(False)
    before = model.first.weight.detach().clone()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad).backward()
    optimizer.step()
    assert torch.equal(model.first.weight, before)
    assert model.first.weight not in optimizer.state
