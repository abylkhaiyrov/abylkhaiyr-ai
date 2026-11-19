import pytest
import torch

from llm.core.config import ModelConfig, TokenizerConfig
from llm.model.transformer import CausalLM
from llm.tokenizer.bpe import BPETokenizer

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog. 2 + 2 = 4.",
    "Быстрая коричневая лиса прыгает через ленивую собаку.",
    "Жылдам қоңыр түлкі еріншек итті секіріп өтеді. Қазақстан — біздің елімыз.",
    "def add(a, b):\n    return a + b\n\nprint(add(1, 2))",
    "x^2 + y^2 = z^2; integral of sin(x) dx = -cos(x) + C",
] * 40


@pytest.fixture(scope="session")
def tokenizer() -> BPETokenizer:
    return BPETokenizer.train(SAMPLE_TEXTS, TokenizerConfig(vocab_size=400, min_frequency=1))


@pytest.fixture()
def model_config(tokenizer: BPETokenizer) -> ModelConfig:
    return ModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )


@pytest.fixture()
def model(model_config: ModelConfig) -> CausalLM:
    torch.manual_seed(0)
    return CausalLM(model_config).eval()
