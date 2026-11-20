from llm.cli.main import _latest_checkpoint_root


def test_studio_prefers_best_instruction_weights_over_newer_pretraining(tmp_path):
    best = tmp_path / "checkpoints" / "tiny_gpu_sft" / "best"
    step = best.parent / "step-00000100"
    base = tmp_path / "checkpoints" / "tiny_gpu" / "step-00006000"
    for directory in (best, step, base):
        directory.mkdir(parents=True)
        (directory / "meta.json").write_text("{}")
    assert _latest_checkpoint_root(tmp_path) == best


def test_studio_ignores_unfinished_instruction_checkpoints(tmp_path):
    base = tmp_path / "checkpoints" / "tiny_gpu" / "step-00000100"
    pending = tmp_path / "checkpoints" / "tiny_gpu_sft" / "step-00000100.tmp"
    for directory in (base, pending):
        directory.mkdir(parents=True)
        (directory / "meta.json").write_text("{}")
    assert _latest_checkpoint_root(tmp_path) == base.parent
