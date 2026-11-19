import json
from pathlib import Path
from unittest.mock import Mock

from llm.studio.runtime import RuntimeState


def test_pipeline_links_state_configuration_and_metrics(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    source = Path("configs/tiny_v2.yaml").read_text()
    (configs / "tiny_v2.yaml").write_text(source)
    state_dir = tmp_path / "experiments" / "pipeline_tiny_v2"
    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text(
        json.dumps({"stages": {"pretrain": {"status": "running"}, "sft": {"status": "pending"}}})
    )
    output = tmp_path / "experiments" / "tiny_v2"
    output.mkdir()
    (output / "metrics.jsonl").write_text(
        '{"step":100,"train/loss":4.2}\n{"step":100,"validation/loss":4.3}\n'
    )
    process = Mock()
    process.info = {"cmdline": ["python", "scripts/run_pipeline.py"], "create_time": 1000}
    process.pid = 42
    process.cwd.return_value = str(tmp_path)
    process.memory_info.return_value.rss = 100
    monkeypatch.setattr("llm.studio.runtime.psutil.process_iter", lambda *args: [process])
    result = RuntimeState(tmp_path).pipeline()
    assert result is not None
    assert result["pid"] == 42
    assert result["model_name"] == "tiny_v2"
    assert result["max_steps"] == 6000
    assert result["metrics"]["step"] == 100
    assert result["metrics"]["train/loss"] == 4.2
    assert result["stages"]["sft"]["status"] == "pending"
