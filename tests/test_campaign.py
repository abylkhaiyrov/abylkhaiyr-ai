from __future__ import annotations

import sys
from pathlib import Path

import pytest

from llm.core.errors import ConfigError, ResourceError
from llm.training.budget import BudgetClock, BudgetLimits, TimeLedger
from llm.training.preflight import verify_lock
from llm.training.series import run_series
from llm.training.spec import ExperimentSpec, load_spec, spec_digest

LIMITS = BudgetLimits(active_seconds=100, absolute_seconds=1000, max_gap_seconds=30)


class FakeTime:
    def __init__(self) -> None:
        self.monotonic = 0.0
        self.wall = 1_000_000.0

    def now_monotonic(self) -> float:
        return self.monotonic

    def now_wall(self) -> float:
        return self.wall

    def advance(self, seconds: float, asleep: bool = False, counted: bool = False) -> None:
        self.wall += seconds
        if not asleep or counted:
            self.monotonic += seconds


def clock_with(fake: FakeTime, limits: BudgetLimits = LIMITS) -> BudgetClock:
    return BudgetClock(limits, TimeLedger(), fake.now_monotonic, fake.now_wall)


def test_active_time_counts_only_working_seconds():
    fake = FakeTime()
    clock = clock_with(fake)
    fake.advance(10)
    assert clock.tick() == 0
    assert clock.ledger.active_seconds == pytest.approx(10)
    assert clock.remaining_active == pytest.approx(90)


def test_sleep_that_the_monotonic_clock_skips_is_idle_time():
    fake = FakeTime()
    clock = clock_with(fake)
    fake.advance(5)
    fake.advance(3600, asleep=True)
    fake.advance(5)
    gap = clock.tick()
    assert gap == pytest.approx(3600)
    assert clock.ledger.active_seconds == pytest.approx(10)
    assert clock.ledger.idle_seconds == pytest.approx(3600)
    assert clock.ledger.wakeups == 1
    assert clock.exhausted() == "absolute wall-clock guard reached"


def test_sleep_that_the_monotonic_clock_counts_is_idle_time_too():
    fake = FakeTime()
    clock = clock_with(fake)
    fake.advance(2000, asleep=True, counted=True)
    assert clock.tick() == pytest.approx(2000)
    assert clock.ledger.active_seconds == 0
    assert clock.ledger.idle_seconds == pytest.approx(2000)


def test_resource_waiting_is_not_charged_to_the_active_budget():
    fake = FakeTime()
    clock = clock_with(fake)
    with clock.waiting():
        fake.advance(20)
    fake.advance(4)
    clock.tick()
    assert clock.ledger.wait_seconds == pytest.approx(20)
    assert clock.ledger.active_seconds == pytest.approx(4)


def test_budget_is_exhausted_by_active_time_and_by_the_absolute_guard():
    fake = FakeTime()
    clock = clock_with(fake)
    fake.advance(25)
    clock.tick()
    for _ in range(3):
        fake.advance(25)
        clock.tick()
    assert clock.exhausted() == "active budget exhausted"
    with pytest.raises(ResourceError):
        clock.check()
    assert clock.remaining_active == 0


def test_ledger_survives_a_restart(tmp_path):
    fake = FakeTime()
    clock = clock_with(fake)
    fake.advance(12)
    clock.tick()
    path = tmp_path / "ledger.json"
    clock.ledger.save(path)
    restored = TimeLedger.load(path)
    assert restored is not None and restored.active_seconds == pytest.approx(12)
    assert TimeLedger.load(tmp_path / "missing.json") is None


def child_command(marker: Path, delay: float = 0.0):
    def command(job: str, remaining: float) -> list[str]:
        code = f"import time; time.sleep({delay}); open(r'{marker}', 'a').write('{job}')"
        return [sys.executable, "-c", code]

    return command


def written(marker: Path):
    return lambda job: marker.exists() and job in marker.read_text()


def test_series_runs_jobs_in_order_and_skips_completed_jobs(tmp_path):
    marker = tmp_path / "done"
    marker.write_text("b")
    clock = BudgetClock(LIMITS, TimeLedger())
    reason = run_series(["a", "b", "c"], clock, child_command(marker), written(marker))
    assert reason is None
    assert marker.read_text() == "bac"


def test_series_does_not_start_a_job_after_the_budget_is_exhausted(tmp_path):
    marker = tmp_path / "done"
    clock = BudgetClock(LIMITS, TimeLedger(active_seconds=100))
    reason = run_series(["a"], clock, child_command(marker), written(marker))
    assert reason == "active budget exhausted"
    assert not marker.exists()


def test_series_rejects_a_job_that_exits_without_finishing(tmp_path):
    clock = BudgetClock(LIMITS, TimeLedger())
    with pytest.raises(ResourceError):
        run_series(
            ["a"], clock, lambda job, remaining: [sys.executable, "-c", "pass"], lambda job: False
        )
    with pytest.raises(Exception, match="exit status 3"):
        run_series(
            ["a"],
            clock,
            lambda job, remaining: [sys.executable, "-c", "raise SystemExit(3)"],
            lambda job: False,
        )


def test_series_stops_a_running_job_when_the_active_budget_runs_out(tmp_path):
    limits = BudgetLimits(active_seconds=1, absolute_seconds=100, max_gap_seconds=30)
    clock = BudgetClock(limits, TimeLedger())
    reason = run_series(
        ["a"],
        clock,
        lambda job, remaining: [sys.executable, "-c", "import time; time.sleep(30)"],
        lambda job: False,
        poll_seconds=0.1,
    )
    assert reason == "active budget exhausted"


def test_series_restarts_the_same_job_after_a_failed_wake_check(tmp_path):
    fake = FakeTime()
    marker = tmp_path / "done"
    checks = iter([False, True, True, True])
    starts: list[str] = []

    def command(job: str, remaining: float) -> list[str]:
        starts.append(job)
        if len(starts) == 1:
            fake.advance(500, asleep=True)
            return [sys.executable, "-c", "import time; time.sleep(30)"]
        return [sys.executable, "-c", f"open(r'{marker}', 'a').write('{job}')"]

    limits_clock = BudgetClock(
        BudgetLimits(active_seconds=100, absolute_seconds=10_000, max_gap_seconds=30),
        TimeLedger(),
        fake.now_monotonic,
        fake.now_wall,
    )
    reason = run_series(
        ["a"], limits_clock, command, written(marker), lambda job: next(checks), poll_seconds=0.1
    )
    assert reason is None
    assert starts == ["a", "a"]
    assert limits_clock.ledger.idle_seconds == pytest.approx(500)


def valid_spec(**changes) -> dict:
    payload = {
        "name": "unit",
        "base_config": "configs/tiny_v2.yaml",
        "seeds": [1, 2],
        "arms": {"A": {}, "B": {"optimizer": {"learning_rate": 1e-4}}},
        "token_budget": 32768,
        "tokens_per_step": 16384,
        "output_root": "experiments/unit",
        "checkpoint_root": "checkpoints/unit",
        "limits": {"active_seconds": 10, "absolute_seconds": 100},
    }
    return {**payload, **changes}


def test_jobs_are_expanded_seed_major_and_reuse_is_validated():
    spec = ExperimentSpec.model_validate(valid_spec(reuse={"A-1": "checkpoints/old"}))
    assert spec.job_names() == ["A-1", "B-1", "A-2", "B-2"]
    assert spec.runnable_jobs() == ["B-1", "A-2", "B-2"]
    assert spec.target_steps == 2
    assert spec.checkpoint_dir("A-1") == Path("checkpoints/old")
    assert spec.checkpoint_dir("B-1") == Path("checkpoints/unit/B-1")
    with pytest.raises(ValueError, match="unknown job"):
        ExperimentSpec.model_validate(valid_spec(reuse={"Z-9": "x"}))
    with pytest.raises(ValueError, match="every job"):
        ExperimentSpec.model_validate(valid_spec(order=["A-1"]))
    with pytest.raises(ValueError):
        ExperimentSpec.model_validate(valid_spec(unknown_field=1))


def test_job_configurations_apply_arm_seed_and_budget_overrides():
    spec = ExperimentSpec.model_validate(valid_spec(optimizer={"warmup_steps": 3}))
    first = spec.job_configuration("B-2")
    assert first.name == "unit-B-2"
    assert (first.train.seed, first.train.max_steps) == (2, 2)
    assert first.train.optimizer.learning_rate == 1e-4
    assert first.train.optimizer.warmup_steps == 3
    assert spec.job_configuration("A-2").train.optimizer.learning_rate != 1e-4
    assert first.train.checkpoint_dir == "checkpoints/unit/B-2"


def test_invalid_training_overrides_are_reported_as_configuration_errors():
    spec = ExperimentSpec.model_validate(valid_spec(train={"micro_batch_size": 0}))
    with pytest.raises(ConfigError):
        spec.job_configuration("A-1")


def test_specification_digest_changes_with_any_registered_field():
    first = ExperimentSpec.model_validate(valid_spec())
    second = ExperimentSpec.model_validate(valid_spec(token_budget=49152))
    assert spec_digest(first) == spec_digest(ExperimentSpec.model_validate(valid_spec()))
    assert spec_digest(first) != spec_digest(second)


def test_frozen_input_lock_names_the_changed_input(tmp_path):
    lock = tmp_path / "lock.json"
    lock.write_text('{"corpus": "aa", "tokenizer": "bb"}', encoding="utf-8")
    verify_lock({"corpus": "aa", "tokenizer": "bb"}, str(lock), ["corpus", "tokenizer"])
    with pytest.raises(ResourceError, match="tokenizer"):
        verify_lock({"corpus": "aa", "tokenizer": "zz"}, str(lock), ["corpus", "tokenizer"])


@pytest.mark.parametrize("name", ["stage19", "stage20"])
def test_registered_specifications_load(name):
    spec = load_spec(f"configs/experiments/{name}.yaml")
    assert spec.target_steps == 305
    assert set(spec.lock_labels) <= set(spec.frozen_inputs)
