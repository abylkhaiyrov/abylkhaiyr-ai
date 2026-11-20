from __future__ import annotations

from pathlib import Path

import torch

from llm.core.config import load_experiment_config
from llm.data.binary import IGNORE_INDEX
from llm.tokenizer.bpe import BPETokenizer
from llm.training import v4_study


def test_packed_exposure_separates_assistant_targets_and_replay(monkeypatch) -> None:
    config = load_experiment_config(Path("configs/tiny_v4_pretrain.yaml"))
    config.train.pack_sequences = True
    config.train.gradient_accumulation_steps = 2
    inputs = torch.tensor([[3, 4, 5, 0]])
    packed_targets = torch.tensor([[IGNORE_INDEX, 5, 6, IGNORE_INDEX]])
    segments = torch.tensor([[1, 1, 1, 0]])
    replay = torch.tensor([[5, 6, 7, 8]])

    class Loader:
        def __init__(self) -> None:
            self.batches = iter(((inputs, packed_targets, segments), (replay, replay)))

        def next_batch(self):
            return next(self.batches)

    monkeypatch.setattr(v4_study, "_build_loaders", lambda config, tokenizer: (Loader(), None))
    tokenizer = BPETokenizer.train(["An unrelated token stream."], config.tokenizer)
    exposure = v4_study.planned_exposure(config, tokenizer, 1)
    assert exposure.as_dict() == {
        "compute_slots": 8,
        "assistant_tokens": 2,
        "replay_target_tokens": 4,
        "input_tokens": 7,
        "assistant_micro_batches": 1,
        "replay_micro_batches": 1,
        "objective_tokens": 6,
    }
