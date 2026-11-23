from __future__ import annotations

from pathlib import Path

from llm.core.config import ExperimentConfig, OptimizerConfig, TokenizerConfig, TrainConfig
from llm.core.profiles import build_model_config
from llm.core.scaling import flops_per_token

ARCHITECTURES = ("tiny", "small")
SEEDS = (1, 2)
CONTEXT = 512
BATCH = 8
ACCUMULATION = 4
TOKENS_PER_STEP = CONTEXT * BATCH * ACCUMULATION
BUDGETS = (5_000_000, 20_000_000, 50_000_000, 100_000_000)
INITIAL_WALL_SECONDS = 3_300
MIN_DISK_GIB = 20.0


def budget_step(tokens: int) -> int:
    return max(1, round(tokens / TOKENS_PER_STEP))


def model_profile(profile: str):
    return build_model_config(profile, 8192).model_copy(update={"max_position_embeddings": CONTEXT})


def matched_step(profile: str, tiny_budget: int) -> int:
    tiny_cost = flops_per_token(model_profile("tiny"), CONTEXT)
    cost = flops_per_token(model_profile(profile), CONTEXT)
    return budget_step(round(budget_step(tiny_budget) * TOKENS_PER_STEP * tiny_cost / cost))


def capacity_config(
    profile: str,
    seed: int,
    output: Path,
    checkpoints: Path,
    corpus: Path,
    tokenizer: Path,
) -> ExperimentConfig:
    model = model_profile(profile)
    steps = sorted({budget_step(n) for n in BUDGETS} | {matched_step(profile, n) for n in BUDGETS})
    label = f"{profile}-seed-{seed}"
    return ExperimentConfig(
        name=f"stage17-{label}",
        model=model,
        tokenizer=TokenizerConfig(vocab_size=8192, path=str(tokenizer)),
        train=TrainConfig(
            stage="pretrain",
            data_dir=str(corpus),
            checkpoint_dir=str(checkpoints / label),
            output_dir=str(output / label),
            device="mps",
            precision="fp32",
            seed=seed,
            num_threads=4,
            micro_batch_size=BATCH,
            gradient_accumulation_steps=ACCUMULATION,
            min_micro_batch_size=2,
            dynamic_batch_size=True,
            sequence_length=CONTEXT,
            max_steps=budget_step(BUDGETS[-1]),
            eval_interval=61,
            eval_batches=12,
            log_interval=10,
            checkpoint_interval=61,
            protected_steps=steps,
            keep_checkpoints=2,
            fast_causal=True,
            max_ram_fraction=0.90,
            min_free_disk_gb=MIN_DISK_GIB,
            dataset_version="v4-expanded",
            optimizer=OptimizerConfig(
                learning_rate=5e-4,
                min_lr_ratio=1.0,
                warmup_steps=30,
                foreach=True,
            ),
        ),
    )
