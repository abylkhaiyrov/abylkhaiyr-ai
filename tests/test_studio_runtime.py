import json
from pathlib import Path
from unittest.mock import Mock

from llm.studio.runtime import RuntimeState


def write_project(root: Path, state: dict) -> None:
    configs = root / "configs"
    configs.mkdir()
    (configs / "tiny_v2.yaml").write_text(Path("configs/tiny_v2.yaml").read_text())
    state_dir = root / "experiments" / "pipeline_tiny_v2"
    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text(json.dumps(state))
    output = root / "experiments" / "tiny_v2"
    output.mkdir()
    (output / "metrics.jsonl").write_text(
        '{"step":100,"train/loss":4.2,"perf/eta_seconds":60}\n{"step":100,"validation/loss":4.3}\n'
    )


def fake_process(root: Path) -> Mock:
    process = Mock()
    process.info = {
        "cmdline": [
            "python",
            "scripts/run_pipeline.py",
            "--state-dir",
            "experiments/pipeline_tiny_v2",
        ],
        "create_time": 1000,
    }
    process.pid = 42
    process.cwd.return_value = str(root)
    process.memory_info.return_value.rss = 100
    process.cpu_percent.return_value = 5.0
    return process


def test_gpu_training_is_detected_outside_pipeline(tmp_path, monkeypatch):
    process = fake_process(tmp_path)
    process.info["cmdline"] = [
        "python",
        "scripts/train_gpu.py",
        "--config",
        "configs/tiny_gpu_sft.yaml",
    ]
    monkeypatch.setattr("llm.studio.runtime.psutil.process_iter", lambda *args: [process])
    assert RuntimeState(tmp_path).training_active()


def test_snapshot_links_state_configuration_and_metrics(tmp_path, monkeypatch):
    write_project(
        tmp_path,
        {"stages": {"pretrain": {"status": "running", "started_at": "2026-01-01T00:00:00+00:00"}}},
    )
    monkeypatch.setattr(
        "llm.studio.runtime.psutil.process_iter", lambda *args: [fake_process(tmp_path)]
    )
    pipeline = RuntimeState(tmp_path).snapshot()["pipeline"]
    assert pipeline["running"] is True
    assert pipeline["pid"] == 42
    assert pipeline["model_name"] == "tiny_v2"
    assert pipeline["current_stage"] == "pretrain"
    stage = next(item for item in pipeline["stages"] if item["key"] == "pretrain")
    assert stage["total"] == 6000
    assert stage["done"] == 100
    assert stage["percent"] == 100 / 6000 * 100
    assert stage["eta_seconds"] == 60
    assert stage["results"]["validation_loss"] == 4.3
    assert [item["key"] for item in pipeline["stages"]].count("eval_pretrain") == 1


def test_running_evaluation_without_telemetry_is_indeterminate(tmp_path, monkeypatch):
    write_project(
        tmp_path,
        {
            "stages": {
                "pretrain": {
                    "status": "completed",
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T01:00:00+00:00",
                    "metrics": {"final_step": 6000},
                },
                "eval_pretrain": {"status": "running", "started_at": "2026-01-01T01:00:00+00:00"},
            }
        },
    )
    monkeypatch.setattr(
        "llm.studio.runtime.psutil.process_iter", lambda *args: [fake_process(tmp_path)]
    )
    pipeline = RuntimeState(tmp_path).snapshot()["pipeline"]
    evaluation = next(item for item in pipeline["stages"] if item["key"] == "eval_pretrain")
    assert evaluation["indeterminate"] is True
    assert evaluation["percent"] is None
    pretrain = next(item for item in pipeline["stages"] if item["key"] == "pretrain")
    assert pretrain["percent"] == 100.0
    assert pretrain["duration_seconds"] == 3600
    assert pipeline["last_result_stage"] == "pretrain"
    assert pipeline["next_stage"] == "select_best_pretrain"


def test_dead_process_marks_running_stage_interrupted(tmp_path, monkeypatch):
    write_project(tmp_path, {"stages": {"sft": {"status": "running"}}})
    monkeypatch.setattr("llm.studio.runtime.psutil.process_iter", lambda *args: [])
    runtime = RuntimeState(tmp_path)
    pipeline = runtime.snapshot()["pipeline"]
    assert pipeline["running"] is False
    assert (
        next(item for item in pipeline["stages"] if item["key"] == "sft")["status"] == "interrupted"
    )
    assert runtime.training_active() is False
