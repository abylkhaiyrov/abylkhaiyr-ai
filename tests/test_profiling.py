import numpy as np

from llm.core.config import ExperimentConfig, ModelConfig, TokenizerConfig, TrainConfig
from llm.optimization.profiling import PHASES, Variant, measure_throughput, profile_phases


def make_config(tmp_path, tokenizer) -> ExperimentConfig:
    tokens = tmp_path / "tokens"
    tokens.mkdir()
    data = np.random.default_rng(0).integers(4, tokenizer.vocab_size, 4000).astype(np.uint16)
    data.tofile(tokens / "train.bin")
    data.tofile(tokens / "validation.bin")
    tokenizer.save(tmp_path / "tok")
    return ExperimentConfig(
        name="profile",
        model=ModelConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            max_position_embeddings=64,
        ),
        tokenizer=TokenizerConfig(vocab_size=tokenizer.vocab_size, path=str(tmp_path / "tok")),
        train=TrainConfig(
            stage="pretrain",
            data_dir=str(tmp_path),
            device="cpu",
            precision="fp32",
            sequence_length=16,
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=4,
        ),
    )


def test_throughput_and_phase_profile_cover_every_variant(tmp_path, tokenizer):
    config = make_config(tmp_path, tokenizer)
    variants = [
        Variant("baseline", micro_batch=2, accumulation=2, foreach=False),
        Variant("fast", micro_batch=2, accumulation=2, native_gqa=True, fast_causal=True),
    ]
    for variant in variants:
        result = measure_throughput(config, variant, steps=2)
        assert result.useful_tokens_per_second > 0
        assert 0 < result.useful_fraction <= 1
        phases = profile_phases(config, variant, steps=1)
        assert set(PHASES) <= set(phases) and phases["checkpoint_save"] > 0
        assert all(value >= 0 for value in phases.values())


def test_optimized_variants_follow_the_baseline_loss_trajectory(tmp_path, tokenizer):
    from llm.optimization.profiling import loss_trajectory, max_trajectory_gap

    config = make_config(tmp_path, tokenizer)
    base = Variant("base", micro_batch=2, accumulation=2, foreach=False)
    tuned = Variant("tuned", micro_batch=2, accumulation=2, native_gqa=True, fast_causal=True)
    reference = loss_trajectory(config, base, 5)
    assert max_trajectory_gap(reference, loss_trajectory(config, tuned, 5)) < 1e-4
