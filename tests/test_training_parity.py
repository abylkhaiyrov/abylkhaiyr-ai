from __future__ import annotations

import copy

import numpy as np
import torch
from torch.nn import functional

from llm.data.binary import PackedTokenLoader
from llm.model.transformer import CausalLM
from llm.optimization.schedule import learning_rate_at
from tests.test_training import build_trainer

STEPS = 4
MICRO = 4
ACCUMULATION = 2


def reference_run(trainer, model: CausalLM, source: PackedTokenLoader) -> list[CausalLM]:
    settings = trainer.config.optimizer
    decayed = [p for p in model.parameters() if p.ndim > 1]
    plain = [p for p in model.parameters() if p.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": settings.weight_decay},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=settings.learning_rate,
        betas=(settings.beta1, settings.beta2),
        eps=settings.eps,
    )
    snapshots = []
    for step in range(STEPS):
        rate = learning_rate_at(
            step,
            trainer.config.max_steps,
            settings.learning_rate,
            settings.warmup_steps,
            settings.min_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        batches = [source.next_batch() for _ in range(ACCUMULATION)]
        total = sum(int((targets != -100).sum()) for _, targets in batches)
        optimizer.zero_grad(set_to_none=True)
        for inputs, targets in batches:
            logits = model(inputs)
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"
            )
            (loss / total).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
        optimizer.step()
        snapshots.append(copy.deepcopy(model))
    return snapshots


def test_full_training_steps_match_a_plain_pytorch_loop(tmp_path, model_config):
    trainer = build_trainer(
        tmp_path, model_config, steps=STEPS, accumulation=ACCUMULATION, micro=MICRO, name="parity"
    )
    reference = copy.deepcopy(trainer.model).train()
    source = PackedTokenLoader(tmp_path / "train.bin", np.dtype(np.uint16), 16, MICRO, seed=0)
    expected = reference_run(trainer, reference, source)
    trainer.model.train()
    for step in range(STEPS):
        trainer.run_step()
        trainer.state.step += 1
        for ours, theirs in zip(
            trainer.model.parameters(), expected[step].parameters(), strict=True
        ):
            assert torch.allclose(ours, theirs, atol=1e-6), f"step {step}"
