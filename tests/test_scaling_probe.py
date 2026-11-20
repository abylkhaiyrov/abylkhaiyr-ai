from pathlib import Path

from llm.core.config import load_experiment_config
from llm.core.profiles import count_parameters
from llm.training.scaling_probe import probe_config

ROOT = Path(__file__).resolve().parents[1]


def test_probe_changes_only_the_model_size_and_matches_the_token_budget(tmp_path):
    base = load_experiment_config(str(ROOT / "configs/tiny_v2.yaml"))
    sizes = {}
    for name in ("tiny", "small"):
        config = probe_config(base, name, 6_000_000, tmp_path)
        per_step = (
            config.train.micro_batch_size
            * config.train.gradient_accumulation_steps
            * config.train.sequence_length
        )
        assert abs(config.train.max_steps * per_step - 6_000_000) < per_step
        assert config.train.init_from is None
        assert config.train.optimizer.learning_rate == base.train.optimizer.learning_rate
        assert config.train.mixture == base.train.mixture
        sizes[name] = count_parameters(config.model)
    assert sizes["small"] > 3 * sizes["tiny"]
