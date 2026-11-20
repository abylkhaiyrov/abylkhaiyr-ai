import torch

from llm.evaluation.stress import growth, inference_memory_profile
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM


def test_many_varied_chat_requests_do_not_accumulate_tensors_or_cache(model, tokenizer):
    engine = InferenceEngine(CausalLM(model.config), tokenizer, torch.device("cpu"))
    samples = inference_memory_profile(engine, requests=60, max_prompt_words=20, every=10)
    assert len(samples) == 6
    assert growth(samples, "live_tensors") == 0
    assert {sample.live_tensors for sample in samples} == {samples[0].live_tensors}
