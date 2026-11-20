import json

from llm.studio.evaluations import evaluation_reports
from llm.studio.model_registry import annotate_models
from llm.studio.pipeline_view import data_stages, stage_history


def test_data_stages_use_the_selected_manifest(tmp_path):
    for name, tokens in (("processed", 47_000_000), ("v2", 309_257_064)):
        directory = tmp_path / "data" / name
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps({"version": name, "tokenized": {"train": {"tokens": tokens}}})
        )

    prepared, tokenized = data_stages(tmp_path, "data/v2")

    assert prepared["results"]["dataset"] == "v2"
    assert tokenized["results"] == {"train_tokens": 309_257_064}


def test_evaluations_keep_distinct_stages_and_skip_reused_final_results(tmp_path):
    directory = tmp_path / "experiments" / "pipeline_v2"
    directory.mkdir(parents=True)
    pretrain = "checkpoints/pretrain/best"
    sft = "checkpoints/sft/best"
    state = {
        "config": {"data_dir": "data/v2"},
        "stages": {
            "eval_pretrain": {
                "status": "completed",
                "candidates": {pretrain: {"test_perplexity": 19.66}},
            },
            "eval_sft": {
                "status": "completed",
                "checkpoint": sft,
                "metrics": {"test_perplexity": 46.55},
            },
            "eval_final": {
                "status": "completed",
                "source": "sft",
                "selected": sft,
                "metrics": {"test_perplexity": 46.55},
            },
        },
    }
    (directory / "state.json").write_text(json.dumps(state))

    reports = evaluation_reports(tmp_path)

    assert [(item["stage"], item["checkpoint"], item["dataset"]) for item in reports] == [
        ("pretrain", pretrain, "data/v2"),
        ("sft", sft, "data/v2"),
    ]


def test_export_does_not_promote_a_checkpoint_that_failed_quality(tmp_path):
    checkpoint = "checkpoints/sft/best"
    pipeline = {
        "artifacts": {"sft_checkpoint": checkpoint, "final_checkpoint": checkpoint},
        "stages": [
            {"key": "eval_sft", "status": "completed", "checkpoint_dir": None},
            {
                "key": "dpo",
                "status": "skipped",
                "checkpoint_dir": None,
                "note_code": "sft_perplexity_regression",
            },
            {"key": "export_release", "status": "completed", "checkpoint_dir": None},
        ],
    }
    model = {"id": checkpoint, "name": "sft", "step": 1500, "created_at": "2026-11-20"}

    roles = annotate_models(tmp_path, pipeline, [model], None)

    assert roles["production"] is None
    assert model["evaluated"] is True
    assert model["quality_failed"] is True
    assert model["verified"] is False


def test_stage_history_preserves_failed_and_completed_attempts(tmp_path):
    log = tmp_path / "pipeline.log"
    log.write_text(
        "2026-11-20 10:00:00,000 INFO starting stage sft\n"
        "2026-11-20 10:10:00,000 ERROR stage sft failed\n"
        "2026-11-20 10:20:00,000 INFO starting stage sft\n"
        "2026-11-20 10:30:00,000 INFO finished stage sft (status=completed)\n"
    )

    assert [event["status"] for event in stage_history(log)["sft"]] == [
        "running",
        "failed",
        "running",
        "completed",
    ]
