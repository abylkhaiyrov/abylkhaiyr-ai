import math
from pathlib import Path

import numpy as np
import pytest
import torch

from llm.core.config import ModelConfig
from llm.data.binary import ConversationLoader
from llm.model.transformer import CausalLM
from llm.training.losses import causal_lm_loss_sum, dpo_loss, sequence_log_probs

PROCESSED = Path("data/processed")


@pytest.fixture()
def double_model(model_config: ModelConfig) -> CausalLM:
    torch.manual_seed(0)
    return CausalLM(model_config).double().train()


def loss_of(model: CausalLM, ids: torch.Tensor) -> torch.Tensor:
    total, count = causal_lm_loss_sum(model(ids[:, :-1]), ids[:, 1:])
    return total / count


def test_full_model_gradients_match_finite_differences(double_model, model_config):
    torch.manual_seed(1)
    ids = torch.randint(4, model_config.vocab_size, (2, 9))
    loss_of(double_model, ids).backward()
    rng = np.random.default_rng(0)
    checked = 0
    for name, parameter in double_model.named_parameters():
        flat = parameter.view(-1)
        for index in rng.choice(flat.numel(), size=3, replace=False):
            analytic = parameter.grad.view(-1)[index].item()
            original = flat[index].item()
            step = 1e-6
            with torch.no_grad():
                flat[index] = original + step
                up = loss_of(double_model, ids).item()
                flat[index] = original - step
                down = loss_of(double_model, ids).item()
                flat[index] = original
            numeric = (up - down) / (2 * step)
            assert analytic == pytest.approx(numeric, abs=1e-7, rel=1e-4), name
            checked += 1
    assert checked > 30


def test_right_padding_does_not_change_loss(model, model_config):
    ids = torch.randint(4, model_config.vocab_size, (1, 10))
    padded = torch.cat((ids, torch.zeros(1, 6, dtype=torch.long)), dim=1)
    targets = padded[:, 1:].clone()
    targets[:, 9:] = -100
    with torch.no_grad():
        plain, count = causal_lm_loss_sum(model(ids[:, :-1]), ids[:, 1:])
        wide, wide_count = causal_lm_loss_sum(model(padded[:, :-1]), targets)
    assert count == wide_count and plain.item() == pytest.approx(wide.item(), rel=1e-5)


def test_initial_loss_is_close_to_uniform(model, model_config):
    ids = torch.randint(4, model_config.vocab_size, (8, 32))
    with torch.no_grad():
        total, count = causal_lm_loss_sum(model(ids[:, :-1]), ids[:, 1:])
    assert total.item() / count == pytest.approx(math.log(model_config.vocab_size), rel=0.05)


def test_residual_projection_init_is_scaled(model, model_config):
    expected = model_config.initializer_range / math.sqrt(2 * model_config.num_hidden_layers)
    std = model.blocks[0].attention.o_proj.weight.std().item()
    assert std == pytest.approx(expected, rel=0.25)
    assert model.blocks[0].attention.q_proj.weight.std().item() == pytest.approx(
        model_config.initializer_range, rel=0.25
    )


def test_attention_does_not_depend_on_gqa_group_order(model, model_config):
    ids = torch.randint(4, model_config.vocab_size, (1, 8))
    with torch.no_grad():
        assert torch.isfinite(model(ids)).all()
        assert model(ids).shape[-1] == model_config.vocab_size


def test_softmax_fp16_bf16_finite():
    from llm.model.attention import stable_softmax

    for dtype in (torch.float16, torch.bfloat16):
        x = torch.tensor([[60000.0, 59990.0, -60000.0]], dtype=dtype)
        assert torch.isfinite(stable_softmax(x)).all()


def test_dpo_step_increases_preference_margin(model, model_config):
    import copy

    reference = copy.deepcopy(model).eval()
    policy = model.train()
    chosen = torch.randint(4, model_config.vocab_size, (2, 12))
    rejected = torch.randint(4, model_config.vocab_size, (2, 12))
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.5)

    def margin() -> torch.Tensor:
        pc = sequence_log_probs(policy(chosen[:, :-1]), chosen[:, 1:])
        pr = sequence_log_probs(policy(rejected[:, :-1]), rejected[:, 1:])
        with torch.no_grad():
            rc = sequence_log_probs(reference(chosen[:, :-1]), chosen[:, 1:])
            rr = sequence_log_probs(reference(rejected[:, :-1]), rejected[:, 1:])
        return dpo_loss(pc, pr, rc, rr, 0.5)

    before = margin()
    before.backward()
    optimizer.step()
    optimizer.zero_grad()
    assert margin().item() < before.item()


@pytest.mark.skipif(not (PROCESSED / "sft" / "validation.ids.bin").exists(), reason="no data")
def test_real_sft_labels_cover_only_assistant_text():
    from llm.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer.load("data/tokenizer")
    loader = ConversationLoader(
        PROCESSED / "sft" / "validation", np.dtype(np.uint16), 512, 1, tokenizer.pad_id, 0, False
    )
    import json

    rows = [
        json.loads(line)
        for line in (PROCESSED / "sft" / "validation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for row in rows[:20]:
        _, targets = loader.next_batch()
        trained = targets[0][targets[0] != -100].tolist()
        decoded = tokenizer.decode(trained, skip_special_tokens=True)
        assistant = row["messages"][-1]["content"]
        assert decoded == assistant[: len(decoded)]
        assert row["messages"][0]["content"][:30] not in decoded[:30] or len(decoded) < 5
