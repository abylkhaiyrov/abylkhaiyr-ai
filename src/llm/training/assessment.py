from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llm.core.checksums import file_sha256
from llm.core.errors import ConfigError
from llm.evaluation.campaign_suites import SUITES, Suite
from llm.evaluation.capacity_metrics import paired_summary
from llm.evaluation.comparison import load_engine
from llm.evaluation.gates import Gate, GateResult, Measurement, decide, judge_gate
from llm.evaluation.independence import verify_independence
from llm.evaluation.outcomes import Outcome, SuiteOutput
from llm.evaluation.plan import BASELINE, EvaluationPlan
from llm.evaluation.registry import REGISTRY_FILE, record_experiment
from llm.training.checkpoint import CheckpointManager
from llm.training.progress import is_complete
from llm.training.spec import ExperimentSpec, spec_digest

METHODOLOGY = "2"


def digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def option_files(value: object) -> dict[str, str]:
    found: dict[str, str] = {}
    if isinstance(value, str):
        path = Path(value)
        if path.is_file():
            found[value] = file_sha256(path)
    elif isinstance(value, Mapping):
        for item in value.values():
            found.update(option_files(item))
    elif isinstance(value, list):
        for item in value:
            found.update(option_files(item))
    return found


def shared_key(stamp: str, plan: EvaluationPlan) -> str:
    tokenizer = Path(plan.tokenizer) / "tokenizer.json"
    return digest(
        {
            "stamp": stamp,
            "tokenizer": file_sha256(tokenizer) if tokenizer.is_file() else "",
            "device": plan.device,
            "methodology": METHODOLOGY,
        }
    )


def load_saved(path: Path, stamp: str) -> SuiteOutput | None:
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    return SuiteOutput.from_json(saved["output"]) if saved["stamp"] == stamp else None


def write_saved(path: Path, stamp: str, checkpoint: str, output: SuiteOutput) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"stamp": stamp, "checkpoint": checkpoint, "output": output.to_json()}) + "\n",
        encoding="utf-8",
    )


def fingerprint(checkpoint: str) -> str:
    resolved = CheckpointManager(checkpoint, keep=1).latest_valid() or Path(checkpoint)
    meta = CheckpointManager.verify(resolved, require_training_state=False)
    return digest({"files": meta["files"], "tokenizer": meta.get("tokenizer_version")})


def evaluate_checkpoint(
    label: str,
    checkpoint: str,
    plan: EvaluationPlan,
    directory: Path,
    suites: Mapping[str, Suite] = SUITES,
) -> dict[str, SuiteOutput]:
    directory.mkdir(parents=True, exist_ok=True)
    weights = fingerprint(checkpoint)
    results: dict[str, SuiteOutput] = {}
    engine = None
    for suite in plan.suites:
        stamp = digest(
            {
                "weights": weights,
                "suite": suite.model_dump(),
                "files": option_files(suite.options),
            }
        )
        path = directory / f"{label}.{suite.key}.json"
        shared = (
            Path(plan.cache) / f"{shared_key(stamp, plan)}.json" if plan.cache is not None else None
        )
        reused = load_saved(path, stamp) or (load_saved(shared, stamp) if shared else None)
        if reused is not None:
            results[suite.key] = reused
            write_saved(path, stamp, checkpoint, reused)
            continue
        if engine is None:
            engine = load_engine(checkpoint, plan.tokenizer, plan.device)
        output = suites[suite.name](engine, suite.options)
        write_saved(path, stamp, checkpoint, output)
        if shared is not None:
            write_saved(shared, stamp, checkpoint, output)
        results[suite.key] = output
    return results


def seeds_ready(spec: ExperimentSpec, done: set[str]) -> list[int]:
    pairs = spec.job_pairs()
    return [
        seed
        for seed in spec.seeds
        if all(name in done for name, (_, number) in pairs.items() if number == seed)
    ]


def check_inputs(reference: Mapping[str, SuiteOutput], other: Mapping[str, SuiteOutput]) -> None:
    for name, output in other.items():
        if output.inputs != reference[name].inputs:
            raise ConfigError(f"evaluation inputs differ between models for suite {name}")


def mean_metric(runs: list[Mapping[str, SuiteOutput]], suite: str, metric: str) -> float | None:
    values = [run[suite].metrics[metric] for run in runs if metric in run[suite].metrics]
    return statistics.fmean(values) if len(values) == len(runs) and values else None


def flat_runs(output: SuiteOutput, metric: str) -> list[list[Outcome]]:
    return output.items.get(metric, [])


def paired_measurement(
    reference: list[Mapping[str, SuiteOutput]],
    candidate: list[Mapping[str, SuiteOutput]],
    suite: str,
    metric: str,
    repetitions: int,
) -> Measurement | None:
    mean_candidate = mean_metric(candidate, suite, metric)
    mean_reference = mean_metric(reference, suite, metric)
    if mean_candidate is None or mean_reference is None:
        return None
    pairs = [
        ([o.as_choice() for o in ref_run], [o.as_choice() for o in cand_run])
        for ref, cand in zip(reference, candidate, strict=True)
        for ref_run, cand_run in zip(
            flat_runs(ref[suite], metric), flat_runs(cand[suite], metric), strict=True
        )
    ]
    if not pairs:
        return Measurement(mean_candidate, mean_reference)
    summary = paired_summary(pairs, method="choice", repetitions=repetitions)
    low, high = summary["cluster_bootstrap_95"]
    return Measurement(mean_candidate, mean_reference, (low, high), int(summary["n"]))


def absolute_measurement(
    candidate: list[Mapping[str, SuiteOutput]], suite: str, metric: str, repetitions: int
) -> Measurement | None:
    mean = mean_metric(candidate, suite, metric)
    if mean is None:
        return None
    pairs = [
        (
            [Outcome(o.id, o.cluster, o.group, 0).as_choice() for o in run],
            [o.as_choice() for o in run],
        )
        for run_set in candidate
        for run in flat_runs(run_set[suite], metric)
    ]
    if not pairs:
        return Measurement(mean, 0.0)
    summary = paired_summary(pairs, method="choice", repetitions=repetitions)
    low, high = summary["cluster_bootstrap_95"]
    return Measurement(mean, 0.0, (low, high), int(summary["n"]))


def measure_gate(
    gate: Gate,
    reference: list[Mapping[str, SuiteOutput]],
    candidate: list[Mapping[str, SuiteOutput]],
    suite: str,
    repetitions: int,
) -> Measurement | None:
    if gate.kind in ("min_value", "max_value") and gate.bound == "interval":
        return absolute_measurement(candidate, suite, gate.metric, repetitions)
    return paired_measurement(reference, candidate, suite, gate.metric, repetitions)


def arm_runs(
    spec: ExperimentSpec,
    arm: str,
    seeds: list[int],
    outputs: Mapping[str, dict[str, SuiteOutput]],
    baseline: dict[str, SuiteOutput],
) -> list[Mapping[str, SuiteOutput]]:
    if arm == BASELINE:
        return [baseline for _ in seeds]
    names = {seed: name for name, (owner, seed) in spec.job_pairs().items() if owner == arm}
    return [outputs[names[seed]] for seed in seeds]


def assess(
    spec: ExperimentSpec,
    suites: Mapping[str, Suite] = SUITES,
) -> dict[str, Any]:
    plan = spec.evaluation
    if plan is None:
        raise ConfigError("the experiment declares no evaluation")
    if plan.independence is not None:
        verify_independence(plan.independence)
    directory = Path(spec.output_root) / "evaluation"
    done = {job for job in spec.job_names() if is_complete(spec, job)}
    seeds = seeds_ready(spec, done)
    if not seeds:
        raise ConfigError("no seed has every arm complete")
    baseline = evaluate_checkpoint(BASELINE, plan.baseline, plan, directory, suites)
    outputs = {}
    for job in spec.job_names():
        if job in done and spec.job_pairs()[job][1] in seeds:
            outputs[job] = evaluate_checkpoint(
                job, str(spec.checkpoint_dir(job)), plan, directory, suites
            )
            check_inputs(baseline, outputs[job])
    results: list[GateResult] = []
    per_seed: dict[str, list[dict[str, float]]] = {}
    for gate in plan.gates:
        suite = suite_of(baseline, gate.metric)
        candidate = arm_runs(spec, gate.arm, seeds, outputs, baseline)
        reference = arm_runs(spec, gate.against, seeds, outputs, baseline)
        measurement = measure_gate(gate, reference, candidate, suite, plan.repetitions)
        results.append(judge_gate(gate, measurement))
        per_seed[gate.name] = seed_differences(seeds, reference, candidate, suite, gate.metric)
    accepted, reasons = (
        decide(results) if results else (None, ["no gates registered: diagnostic run"])
    )
    report = {
        "specification": spec_digest(spec),
        "seeds": seeds,
        "accepted": accepted,
        "reasons": reasons,
        "promoted": False,
        "gates": [
            {"name": r.gate.name, "passed": r.passed, "observed": r.observed, "detail": r.detail}
            for r in results
        ],
        "per_seed": per_seed,
        "arms": summarize_arms(spec, seeds, outputs, baseline),
    }
    (directory / "decision.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    register(spec, report)
    return report


def seed_differences(
    seeds: list[int],
    reference: list[Mapping[str, SuiteOutput]],
    candidate: list[Mapping[str, SuiteOutput]],
    suite: str,
    metric: str,
) -> list[dict[str, float]]:
    rows = []
    for seed, ref, cand in zip(seeds, reference, candidate, strict=True):
        if metric in ref[suite].metrics and metric in cand[suite].metrics:
            rows.append(
                {
                    "seed": seed,
                    "reference": ref[suite].metrics[metric],
                    "candidate": cand[suite].metrics[metric],
                    "delta": cand[suite].metrics[metric] - ref[suite].metrics[metric],
                }
            )
    return rows


def suite_of(baseline: Mapping[str, SuiteOutput], metric: str) -> str:
    owners = [key for key, output in baseline.items() if metric in output.metrics]
    if len(owners) != 1:
        raise ConfigError(f"gate metric {metric} comes from an unregistered suite")
    return owners[0]


def summarize_arms(
    spec: ExperimentSpec,
    seeds: list[int],
    outputs: Mapping[str, dict[str, SuiteOutput]],
    baseline: dict[str, SuiteOutput],
) -> dict[str, dict[str, float]]:
    labels = [BASELINE, *spec.arms]
    summary: dict[str, dict[str, float]] = {}
    for label in labels:
        runs = arm_runs(spec, label, seeds, outputs, baseline)
        metrics: dict[str, float] = {}
        for suite, output in baseline.items():
            for metric in output.metrics:
                value = mean_metric(runs, suite, metric)
                if value is not None:
                    metrics[metric] = value
        summary[label] = metrics
    return summary


def register(spec: ExperimentSpec, report: dict[str, Any]) -> None:
    path = Path(spec.output_root) / REGISTRY_FILE.name
    details = {
        "kind": "evaluation",
        "accepted": report["accepted"],
        "reasons": report["reasons"],
        "specification": report["specification"],
        "seeds": report["seeds"],
        "promoted": False,
    }
    results = {
        gate["name"]: float(gate["observed"])
        for gate in report["gates"]
        if gate["observed"] is not None
    }
    record_experiment(f"{spec.name}-evaluation", details, results, path)


def fix_baseline(spec: ExperimentSpec, suites: Mapping[str, Suite] = SUITES) -> dict[str, Any]:
    plan = spec.evaluation
    if plan is None:
        raise ConfigError("the experiment declares no evaluation")
    if plan.independence is not None:
        verify_independence(plan.independence)
    directory = Path(spec.output_root) / "evaluation"
    results = evaluate_checkpoint(BASELINE, plan.baseline, plan, directory, suites)
    manifest = {
        "specification": spec_digest(spec),
        "checkpoint": plan.baseline,
        "weights": fingerprint(plan.baseline),
        "metrics": {key: output.metrics for key, output in results.items()},
        "inputs": {key: output.inputs for key, output in results.items()},
    }
    (directory / "baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
