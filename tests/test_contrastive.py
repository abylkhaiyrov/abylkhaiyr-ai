from __future__ import annotations

import copy

import pytest
import torch

from llm.core.config import ContrastiveConfig, ModelConfig, TrainConfig
from llm.model.transformer import CausalLM
from llm.training.losses import (
    IGNORE_INDEX,
    causal_lm_loss_sum,
    contrastive_pair_terms,
    divergent_masks,
)
from llm.training.objectives import CausalLMObjective, ContrastiveObjective

VOCAB = 11


def targets(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long)


def test_divergent_masks_skip_the_shared_prefix_and_the_context() -> None:
    n = IGNORE_INDEX
    chosen = targets([[n, n, 4, 5, 6, 7, n], [n, n, 4, 5, 6, 7, 8]])
    rejected = targets([[n, n, 4, 5, 9, 9, 9], [n, n, 4, 5, 6, 7, 8]])
    chosen_mask, rejected_mask = divergent_masks(chosen, rejected)
    assert chosen_mask.tolist() == [
        [False, False, False, False, True, True, False],
        [False] * 7,
    ]
    assert rejected_mask.tolist() == [
        [False, False, False, False, True, True, True],
        [False] * 7,
    ]


def test_identical_pairs_are_not_valid() -> None:
    logits = torch.randn(1, 5, VOCAB)
    same = targets([[IGNORE_INDEX, 3, 4, 5, 6]])
    terms = contrastive_pair_terms(logits, same, logits, same, 1.0, 0.0)
    assert not bool(terms.valid.any())


def test_gradient_matches_the_analytic_expression() -> None:
    torch.manual_seed(0)
    beta, margin = 1.7, 0.3
    chosen_logits = torch.randn(1, 5, VOCAB, dtype=torch.float64, requires_grad=True)
    rejected_logits = torch.randn(1, 5, VOCAB, dtype=torch.float64, requires_grad=True)
    chosen_targets = targets([[IGNORE_INDEX, 2, 3, 4, 5]])
    rejected_targets = targets([[IGNORE_INDEX, 2, 3, 7, 8]])
    terms = contrastive_pair_terms(
        chosen_logits, chosen_targets, rejected_logits, rejected_targets, beta, margin
    )
    terms.loss.sum().backward()
    z = beta * (terms.delta.item() - margin)
    sigma = torch.sigmoid(torch.tensor(-z, dtype=torch.float64)).item()
    expected_chosen = torch.zeros_like(chosen_logits)
    expected_rejected = torch.zeros_like(rejected_logits)
    for position in (3, 4):
        for logits, row, expected, sign in (
            (chosen_logits, chosen_targets, expected_chosen, -1.0),
            (rejected_logits, rejected_targets, expected_rejected, 1.0),
        ):
            probabilities = torch.softmax(logits[0, position].detach(), dim=-1)
            onehot = torch.zeros(VOCAB, dtype=torch.float64)
            onehot[row[0, position]] = 1.0
            expected[0, position] = sign * beta * sigma * 0.5 * (onehot - probabilities) * -1.0
    assert torch.allclose(chosen_logits.grad, expected_chosen * -1.0, atol=1e-10)
    assert torch.allclose(rejected_logits.grad, expected_rejected * -1.0, atol=1e-10)
    assert chosen_logits.grad[0, 1].abs().sum() == 0
    assert chosen_logits.grad[0, 2].abs().sum() == 0


def test_numerical_gradcheck() -> None:
    torch.manual_seed(1)
    chosen_targets = targets([[IGNORE_INDEX, 2, 3, 4], [IGNORE_INDEX, 6, 3, 5]])
    rejected_targets = targets([[IGNORE_INDEX, 2, 9, 4], [IGNORE_INDEX, 1, 3, 5]])

    def objective(chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
        terms = contrastive_pair_terms(chosen, chosen_targets, rejected, rejected_targets, 1.3, 0.2)
        return (terms.loss * terms.valid).sum()

    inputs = (
        torch.randn(2, 4, VOCAB, dtype=torch.float64, requires_grad=True),
        torch.randn(2, 4, VOCAB, dtype=torch.float64, requires_grad=True),
    )
    assert torch.autograd.gradcheck(objective, inputs)


def tiny_model() -> CausalLM:
    torch.manual_seed(3)
    return CausalLM(
        ModelConfig(
            vocab_size=VOCAB,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            max_position_embeddings=16,
        )
    )


def pair_batch() -> tuple[torch.Tensor, ...]:
    chosen = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 7, 4, 5, 6]])
    rejected = torch.tensor([[1, 2, 3, 8, 9, 10], [1, 2, 7, 9, 8, 10]])

    def split(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = rows[:, 1:].clone()
        target[:, :1] = IGNORE_INDEX
        return rows[:, :-1], target

    return (*split(chosen), *split(rejected))


def test_zero_weight_reduces_to_the_causal_lm_objective() -> None:
    model = tiny_model()
    batch = pair_batch()
    device, dtype = torch.device("cpu"), torch.float32
    plain = CausalLMObjective(device, dtype).loss_sum(model, batch[:2])
    zero = ContrastiveObjective(device, dtype, ContrastiveConfig(weight=0.0)).loss_sum(model, batch)
    assert torch.allclose(plain, zero)
    assert ContrastiveObjective(device, dtype, ContrastiveConfig()).weight(batch) == int(
        (batch[1] != IGNORE_INDEX).sum()
    )


def test_contrastive_term_adds_the_weighted_pair_loss() -> None:
    model = tiny_model()
    batch = pair_batch()
    device, dtype = torch.device("cpu"), torch.float32
    config = ContrastiveConfig(weight=0.5, beta=1.0, margin=0.0)
    objective = ContrastiveObjective(device, dtype, config)
    total = objective.loss_sum(model, batch)
    ce = causal_lm_loss_sum(model(batch[0]), batch[1])[0]
    terms = contrastive_pair_terms(model(batch[0]), batch[1], model(batch[2]), batch[3], 1.0, 0.0)
    expected = ce + 0.5 * objective.weight(batch) * terms.loss.mean()
    assert torch.allclose(total, expected, atol=1e-5)
    assert set(objective.drain_stats()) == {"contrastive/margin", "contrastive/accuracy"}


def margin_after_training(weight: float, steps: int = 6) -> float:
    model = tiny_model()
    batch = pair_batch()
    objective = ContrastiveObjective(
        torch.device("cpu"), torch.float32, ContrastiveConfig(weight=weight, beta=1.0)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        optimizer.zero_grad()
        (objective.loss_sum(model, batch) / objective.weight(batch)).backward()
        optimizer.step()
    terms = contrastive_pair_terms(model(batch[0]), batch[1], model(batch[2]), batch[3], 1.0, 0.0)
    return float(terms.delta.mean())


def test_contrastive_training_separates_correct_from_wrong_answers_more_than_ce() -> None:
    assert margin_after_training(2.0) > margin_after_training(0.0) + 0.1


def test_train_config_accepts_the_contrastive_block() -> None:
    config = TrainConfig.model_validate({"stage": "sft", "contrastive": {"weight": 0.3}})
    assert config.contrastive is not None and config.contrastive.weight == pytest.approx(0.3)
    assert copy.deepcopy(config) == config
    assert TrainConfig().contrastive is None
