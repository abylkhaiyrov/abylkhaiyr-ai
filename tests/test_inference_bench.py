import torch

from llm.evaluation.inference_bench import concat_cache, measure_latency
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM


def test_both_cache_kinds_give_identical_logits_and_positive_latencies(model, tokenizer):
    network = CausalLM(model.config).eval()
    engine = InferenceEngine(network, tokenizer, torch.device("cpu"))
    ids = torch.randint(1, model.config.vocab_size, (1, 12))
    with torch.inference_mode():
        reference = network(ids)
        for cache in (network.new_cache(), concat_cache(model.config.num_hidden_layers)):
            outputs = [network(ids[:, :8], cache=cache)]
            outputs += [network(ids[:, i : i + 1], cache=cache) for i in range(8, 12)]
            assert torch.allclose(torch.cat(outputs, dim=1), reference, atol=1e-4)
    for kind in ("concat", "preallocated"):
        result = measure_latency(engine, 8, 6, 2, kind)
        assert result.prefill_ms > 0 and result.decode_ms_per_token > 0
        assert result.first_token_ms >= result.prefill_ms


def test_kv_cache_memory_reports_allocation_and_used_fraction(model, tokenizer):
    engine = InferenceEngine(CausalLM(model.config), tokenizer, torch.device("cpu"))
    pre = measure_latency(engine, 8, 4, 1, "preallocated")
    cat = measure_latency(engine, 8, 4, 1, "concat")
    assert pre.kv_used_mb > 0 and pre.kv_allocated_mb >= pre.kv_used_mb
    assert abs(cat.kv_allocated_mb - cat.kv_used_mb) < 1e-9
    assert pre.kv_allocated_mb > cat.kv_allocated_mb
