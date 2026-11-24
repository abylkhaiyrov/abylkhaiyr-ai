from __future__ import annotations

import torch

from llm.data.grounded import eval_cases
from llm.evaluation.hidden import (
    answer_prefix,
    capture,
    centroid_accuracy,
    cosine_distance,
    diagnose,
    prompt_ids,
    readout,
)
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM


def engine_of(model_config, tokenizer) -> InferenceEngine:
    return InferenceEngine(CausalLM(model_config), tokenizer, device=torch.device("cpu"))


def test_capture_hooks_do_not_change_the_forward_pass_and_are_removed(model_config, tokenizer):
    engine = engine_of(model_config, tokenizer)
    case = eval_cases("dev")[0]
    ids = prompt_ids(engine, case)[: model_config.max_position_embeddings - 1]
    engine.model.eval()
    with torch.no_grad():
        before = engine.model(torch.tensor([ids]))
    states = capture(engine, ids)
    with torch.no_grad():
        after = engine.model(torch.tensor([ids]))
    assert torch.equal(before, after)
    assert len(states) == len(engine.model.blocks) + 1
    assert all(not block._forward_hooks for block in engine.model.blocks)
    assert torch.allclose(readout(engine, states[-1]).exp().sum(), torch.tensor(1.0), atol=1e-4)


def test_final_block_readout_matches_the_model_logits(model_config, tokenizer):
    engine = engine_of(model_config, tokenizer)
    engine.model.eval()
    ids = prompt_ids(engine, eval_cases("dev")[0])[: model_config.max_position_embeddings - 1]
    with torch.no_grad():
        logits = engine.model(torch.tensor([ids]))[0, -1].float()
    lens = readout(engine, capture(engine, ids)[-1])
    assert torch.allclose(torch.log_softmax(logits, -1), lens, atol=1e-4)


def test_distance_and_centroid_helpers_behave_on_known_vectors():
    assert cosine_distance(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])) < 1e-6
    assert abs(cosine_distance(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) - 1.0) < 1e-6
    train = [(torch.tensor([0.0, 0.0]), 0), (torch.tensor([4.0, 4.0]), 1)]
    test = [(torch.tensor([0.5, 0.2]), 0), (torch.tensor([3.0, 4.5]), 1)]
    assert centroid_accuracy(train, test) == 1.0


def test_prefix_is_the_style_that_precedes_the_value():
    assert answer_prefix("en") == "It is "
    assert answer_prefix("kk").startswith("Бұл")


def test_diagnosis_reports_every_layer(model_config, tokenizer):
    engine = engine_of(model_config.model_copy(update={"max_position_embeddings": 1024}), tokenizer)
    cases = [c for c in eval_cases("dev") if c.language == "en" and c.cell == "seen"]
    report = diagnose(engine, cases, limit=4)
    layers = len(engine.model.blocks) + 1
    assert report["layers"] == layers and report["links"] > 0
    assert len(report["lens_original"]) == layers
    assert len(report["answerability_decodable"]) == layers
    assert set(report["shift"]) >= {"changed", "missing"}
