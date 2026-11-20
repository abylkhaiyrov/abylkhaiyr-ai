import signal
import subprocess
import sys
import threading

import pytest
import torch

from llm.core.errors import ResourceError
from llm.training.session import exclusive_session, graceful_shutdown
from tests.test_training import build_trainer


def test_lock_prevents_second_session(tmp_path):
    lock = tmp_path / "lock"
    with exclusive_session(lock), pytest.raises(ResourceError), exclusive_session(lock):
        pass
    with exclusive_session(lock):
        pass


def test_lock_prevents_another_process(tmp_path):
    lock = tmp_path / "lock"
    code = (
        "import sys; from pathlib import Path; "
        "from llm.training.session import exclusive_session; "
        "with_lock = exclusive_session(Path(sys.argv[1])); with_lock.__enter__()"
    )
    with exclusive_session(lock):
        size = lock.stat().st_size
        blocked = subprocess.run(
            [sys.executable, "-c", code, str(lock)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert blocked.returncode != 0
        assert "another training session" in blocked.stderr
        assert lock.stat().st_size == size
    released = subprocess.run(
        [sys.executable, "-c", code, str(lock)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert released.returncode == 0, released.stderr


def test_signals_set_stop_event_and_restore_handlers():
    event = threading.Event()
    before = signal.getsignal(signal.SIGTERM)
    with graceful_shutdown(event):
        signal.raise_signal(signal.SIGTERM)
        assert event.wait(timeout=2)
    assert signal.getsignal(signal.SIGTERM) == before


def test_interrupt_saves_checkpoint_and_resume_continues(tmp_path, model_config):
    trainer = build_trainer(tmp_path, model_config, steps=20, name="sig")
    original = trainer._train_step
    calls = {"n": 0}

    def stopping():
        calls["n"] += 1
        if calls["n"] == 4:
            trainer.stop_event.set()
        return original()

    trainer._train_step = stopping
    result = trainer.train()
    assert result.interrupted and result.final_step == 4
    resumed = build_trainer(tmp_path, model_config, steps=20, name="sig")
    assert resumed.resume() and resumed.state.step == 4
    assert resumed.train().final_step == 20
    assert not torch.isnan(torch.cat([p.flatten() for p in resumed.model.parameters()])).any()
