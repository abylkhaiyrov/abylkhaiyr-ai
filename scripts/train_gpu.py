"""Run a resumable GPU training session; create experiments/tiny_gpu/STOP to stop."""

from __future__ import annotations

import json
import signal
import threading
from pathlib import Path

import torch

from llm.core.config import load_experiment_config
from llm.training.session import run_training


def main() -> int:
    config = load_experiment_config("configs/tiny_gpu.yaml")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run this GPU session on CPU")
    torch.set_float32_matmul_precision("high")
    output = Path(config.train.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stop_file = output / "STOP"
    if stop_file.exists():
        print(f"Remove {stop_file} before resuming", flush=True)
        return 0
    done = threading.Event()

    def watch_stop() -> None:
        while not done.wait(1):
            if stop_file.exists():
                signal.raise_signal(signal.SIGTERM)
                return

    print(f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}", flush=True)
    watcher = threading.Thread(target=watch_stop, daemon=True)
    watcher.start()
    try:
        result = run_training(config, resume=True, max_restarts=0)
        (output / "result.json").write_text(json.dumps(result.__dict__, indent=2), encoding="utf-8")
        print(json.dumps(result.__dict__), flush=True)
    finally:
        done.set()
        watcher.join(timeout=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
