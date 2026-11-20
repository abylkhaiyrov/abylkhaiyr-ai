from llm.training.promotion import (
    Candidate,
    production_checkpoint,
    promote,
    read_manifest,
    select_production,
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
    assert select_production(items, 0.05).stage == "sft"


def test_pretrained_models_and_repetitive_models_are_never_promoted():
    assert select_production([candidate("pretrain", 0.9)]) is None
    assert select_production([candidate("sft", 0.9, repetition=0.8)]) is None
    assert select_production([candidate("sft", 0.1)], reference_pass_rate=0.2) is None


def test_ties_prefer_lower_repetition_then_later_stage():
    items = [candidate("sft", 0.4, 0.2), candidate("dpo", 0.4, 0.1)]
    assert select_production(items).stage == "dpo"


def test_manifest_roundtrip_and_ready_flag(tmp_path):
    (tmp_path / "checkpoints" / "s" / "step-1").mkdir(parents=True)
    items = [candidate("sft", 0.3, name="checkpoints/s/step-1")]
    manifest = promote(tmp_path, items, 0.0)
    assert manifest["production"]["ready"] is False
    assert production_checkpoint(tmp_path) == "checkpoints/s/step-1"
    assert read_manifest(tmp_path)["models"]["checkpoints/s/step-1"]["stage"] == "sft"
    promote(tmp_path, [candidate("pretrain", 0.9)], 0.0)
    assert production_checkpoint(tmp_path) is None
