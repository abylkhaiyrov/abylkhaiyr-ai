import json

from llm.evaluation.hooks import StageEvaluationHooks
from llm.progress import ProgressTracker, rate_and_eta, read_progress
from llm.studio.model_registry import annotate_models
from llm.studio.service import make_title
from llm.studio.streaming import DeltaBuffer


def test_tracker_persists_stage_progress_atomically(tmp_path):
    tracker = ProgressTracker(tmp_path, min_interval=0)
    handle = tracker.begin("eval_sft", "benchmarks", total=4)
    handle.advance(1, benchmark="arithmetic")
    stage = read_progress(tmp_path)["eval_sft"]
    assert stage["done"] == 1
    assert stage["total"] == 4
    assert stage["detail"]["benchmark"] == "arithmetic"
    handle.finish(ok=True)
    stage = read_progress(tmp_path)["eval_sft"]
    assert stage["status"] == "completed"
    assert stage["done"] == 4
    assert stage["eta_seconds"] is None


def test_tracker_sanitizes_non_finite_values(tmp_path):
    tracker = ProgressTracker(tmp_path, min_interval=0)
    handle = tracker.begin("train", "steps", total=10)
    handle.update(loss=float("nan"))
    assert (
        json.loads((tmp_path / "progress.json").read_text())["stages"]["train"]["detail"]["loss"]
        is None
    )


def test_rate_and_eta_never_invents_values():
    assert rate_and_eta(0, 10, 5) == (None, None)
    assert rate_and_eta(5, None, 5) == (1.0, None)
    assert rate_and_eta(5, 10, 5) == (1.0, 5.0)


def test_evaluation_hooks_track_checkpoints_benchmarks_and_items(tmp_path):
    tracker = ProgressTracker(tmp_path, min_interval=0)
    handle = tracker.begin("eval_pretrain", "benchmarks", total=4)
    hooks = StageEvaluationHooks(handle, checkpoint_total=2, suites_per_checkpoint=2)
    hooks.begin_checkpoint("baseline", 0)
    hooks.suite_started("arithmetic", 0, 2)
    hooks.items_planned(3)
    hooks.item_done(2)
    detail = read_progress(tmp_path)["eval_pretrain"]["detail"]
    assert detail["items_done"] == 2
    assert detail["items_remaining"] == 1
    assert detail["checkpoint_index"] == 1
    hooks.suite_finished("arithmetic", {"arithmetic_exact_match": 0.5}, 1.0)
    hooks.suite_started("factual", 1, 2)
    hooks.suite_finished("factual", {"factual_accuracy": 0.25}, 1.0)
    hooks.end_checkpoint()
    stage = read_progress(tmp_path)["eval_pretrain"]
    assert stage["done"] == 2
    assert stage["detail"]["interim"]["baseline"]["arithmetic_exact_match"] == 0.5
    assert stage["eta_seconds"] is not None


def test_delta_buffer_flushes_first_piece_and_coalesces_fast_pieces():
    buffer = DeltaBuffer(flush_interval=10, flush_characters=100)
    assert buffer.add("a") == "a"
    assert buffer.add("b") is None
    assert buffer.add("c") is None
    assert buffer.drain() == "bc"
    assert buffer.drain() is None


def test_make_title_trims_on_word_boundary():
    assert make_title("**Hello** world") == "Hello world"
    long = make_title("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda")
    assert long.endswith("…")
    assert len(long) <= 49


def test_models_get_roles_and_verified_flags(tmp_path):
    pipeline = {
        "artifacts": {"best_pretrain_checkpoint": str(tmp_path / "checkpoints/a/step-2")},
        "stages": [
            {"key": "select_best_pretrain", "status": "completed", "checkpoint_dir": None},
            {"key": "pretrain", "status": "completed", "checkpoint_dir": "checkpoints/a"},
        ],
    }
    models = [
        {"id": "checkpoints/a/step-1", "name": "a", "step": 1, "created_at": "2026-01-01"},
        {"id": "checkpoints/a/step-2", "name": "a", "step": 2, "created_at": "2026-01-02"},
    ]
    roles = annotate_models(tmp_path.resolve(), pipeline, models, "checkpoints/a/step-1")
    assert roles["best_evaluated"] == "checkpoints/a/step-2"
    assert roles["latest_trained"] == "checkpoints/a/step-2"
    assert models[0]["roles"] == ["loaded"]
    assert models[1]["verified"] is True
