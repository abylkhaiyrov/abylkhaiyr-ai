from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from llm.data.binary import PackedTokenLoader, token_dtype
from llm.data.v4_quality import LANGUAGES
from llm.inference.engine import InferenceEngine
from llm.training.losses import causal_lm_loss_sum


@torch.inference_mode()
def language_losses(
    engine: InferenceEngine, root: Path, batches: int, *, split: str = "validation"
) -> dict:
    vocab_size = engine.tokenizer.vocab_size
    dtype = token_dtype(vocab_size)
    results = {}
    for language in LANGUAGES:
        loader = PackedTokenLoader(
            root / f"{split}.{language}.bin",
            dtype,
            min(512, engine.max_positions),
            1,
            seed=0,
            shuffle=False,
        )
        tokens = np.memmap(root / f"train.{language}.bin", dtype=dtype, mode="r")
        counts = np.ones(vocab_size, dtype=np.int64)
        for start in range(0, len(tokens), 1 << 20):
            counts += np.bincount(tokens[start : start + (1 << 20)], minlength=vocab_size)
        log_probs = np.log(counts / counts.sum())
        total_loss = unigram_loss = 0.0
        for _ in range(min(batches, loader.num_chunks)):
            inputs, targets = loader.next_batch()
            loss, _ = causal_lm_loss_sum(
                engine.model(inputs.to(engine.device)), targets.to(engine.device)
            )
            total_loss += float(loss.item())
            unigram_loss -= float(log_probs[targets.numpy()].sum())
        n = loader.cursor * loader.sequence_length
        results[language] = {
            "tokens": n,
            "loss": total_loss / n,
            "perplexity": math.exp(total_loss / n),
            "unigram_perplexity": math.exp(unigram_loss / n),
        }
    return results
