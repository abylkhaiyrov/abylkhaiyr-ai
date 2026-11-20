from llm.training.promotion import (
    Candidate,
    production_checkpoint,
    promote,
    read_manifest,
    rollback,
    select_experimental,
)


def candidate(stage: str, rate: float, repetition: float = 0.1, name: str | None = None):
    metrics = {"conv_pass_rate": rate, "conv_repetition_ratio": repetition}
    return Candidate(name or f"checkpoints/{stage}/step-1", stage, metrics)


def test_selection_ignores_step_numbers_and_prefers_measured_quality():
    items = [
        candidate("pretrain", 0.05, name="checkpoints/p/step-6000"),
        candidate("sft", 0.30, name="checkpoints/s/step-100"),
        candidate("dpo", 0.25, name="checkpoints/d/step-9999"),
    ]
    assert select_experimental(items, 0.05).stage == "sft"


def test_pretrained_models_and_repetitive_models_are_never_promoted():
    assert select_experimental([candidate("pretrain", 0.9)]) is None
    assert select_experimental([candidate("sft", 0.9, repetition=0.8)]) is None
    assert select_experimental([candidate("sft", 0.1)], reference_pass_rate=0.2) is None


def test_ties_prefer_lower_repetition_then_later_stage():
    items = [candidate("sft", 0.4, 0.2), candidate("dpo", 0.4, 0.1)]
    assert select_experimental(items).stage == "dpo"


def test_manifest_roundtrip_and_ready_flag(tmp_path):
    (tmp_path / "checkpoints" / "s" / "step-1").mkdir(parents=True)
    (tmp_path / "checkpoints" / "s" / "step-1" / "model.safetensors").touch()
    items = [candidate("sft", 0.3, name="checkpoints/s/step-1")]
    manifest = promote(tmp_path, items, 0.0)
    assert manifest["production"] is None
    assert manifest["experimental"]["checkpoint"] == "checkpoints/s/step-1"
    assert production_checkpoint(tmp_path) is None
    assert read_manifest(tmp_path)["models"]["checkpoints/s/step-1"]["stage"] == "sft"
    promote(tmp_path, [candidate("pretrain", 0.9)], 0.0)
    assert production_checkpoint(tmp_path) is None


def test_promotion_requires_held_out_language_metrics(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "sft" / "step-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").touch()
    path = str(checkpoint.relative_to(tmp_path))
    reference = {
        "conv_pass_rate": 0.1,
        "test_perplexity": 20.0,
        "perplexity_en": 20.0,
        "perplexity_ru": 20.0,
        "perplexity_kk": 20.0,
    }
    missing = candidate("sft", 0.6, name=path)
    assert promote(tmp_path, [missing], reference_metrics=reference)["production"] is None
    degraded = Candidate(path, "sft", {**reference, **missing.metrics, "perplexity_kk": 40.0})
    assert promote(tmp_path, [degraded], reference_metrics=reference)["production"] is None

    improved = Candidate(path, "sft", {**reference, **missing.metrics})
    assert promote(tmp_path, [improved], reference_metrics=reference)["production"]["ready"] is True
    assert production_checkpoint(tmp_path) == path
    assert (
        promote(tmp_path, [degraded], reference_metrics=reference)["production"]["checkpoint"]
        == path
    )

    another = checkpoint.with_name("step-2")
    another.mkdir()
    (another / "model.safetensors").touch()
    second = str(another.relative_to(tmp_path))
    promote(tmp_path, [Candidate(second, "sft", improved.metrics)], reference_metrics=reference)
    assert production_checkpoint(tmp_path) == second
    assert rollback(tmp_path) == path
    assert production_checkpoint(tmp_path) == path
