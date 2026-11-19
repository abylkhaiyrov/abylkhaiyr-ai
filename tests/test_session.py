import os
import signal
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


def test_signals_set_stop_event_and_restore_handlers():
    event = threading.Event()
    before = signal.getsignal(signal.SIGTERM)
    with graceful_shutdown(event):
        os.kill(os.getpid(), signal.SIGTERM)
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
