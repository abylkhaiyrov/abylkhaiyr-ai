import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from llm.core.config import load_experiment_config
from llm.optimization.gpu_monitor import parse_sample
from llm.optimization.profiling import Variant, runtime

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "cuda_benchmark", ROOT / "scripts" / "cuda_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_covers_precisions_backends_and_packing_only_for_sft():
    module = load_script()
    pretrain = module.plan(load_experiment_config(str(ROOT / "configs/tiny_gpu.yaml")))
    names = {v.name for v in pretrain}
    assert {"fp32", "tf32", "bf16", "fp16", "bf16_causal_flash", "bf16_causal_compile"} <= names
    assert not any(v.pack for v in pretrain)
    sft = module.plan(load_experiment_config(str(ROOT / "configs/tiny_gpu_sft.yaml")))
    assert any(v.pack for v in sft)
    assert len({v.name for v in sft}) == len(sft)


def test_benchmark_refuses_to_run_without_cuda_and_never_reports_numbers():
    if torch.cuda.is_available():
        pytest.skip("this check is for machines without CUDA")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "cuda_benchmark.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 2 and "NOT RUN" in result.stderr and result.stdout == ""


def test_dry_run_prints_the_plan_and_runtime_is_a_noop_on_cpu():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "cuda_benchmark.py"), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    assert len(json.loads(result.stdout)) >= 14
    with runtime(Variant("x", tf32=True, sdpa="flash"), torch.device("cpu")):
        pass


def test_nvidia_smi_lines_are_parsed_strictly():
    assert parse_sample("87, 1534") == (87.0, 1534.0)
    assert parse_sample("garbage") is None and parse_sample("1,2,3") is None
