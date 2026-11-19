from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm.core.config import ExperimentConfig, load_experiment_config
from llm.core.errors import TrainingError
from llm.core.hardware import resolve_device
from llm.evaluation.engine import EvaluationConfig, Evaluator
from llm.evaluation.registry import record_experiment
from llm.inference.engine import InferenceEngine
from llm.tokenizer.bpe import BPETokenizer
from llm.training.checkpoint import CheckpointManager, load_model_for_inference
from llm.training.release import export_release
from llm.training.session import run_training

STAGES = (
    "pretrain",
    "eval_pretrain",
    "select_best_pretrain",
    "sft",
    "eval_sft",
    "dpo",
    "eval_final",
    "export_release",
)


@dataclass
class QualityGates:
    max_repetition_ratio: float = 0.85
    min_non_empty_rate: float = 0.5
    max_sft_perplexity_ratio: float = 1.5
    require_pretrain_improvement: bool = True


@dataclass
class PipelineConfig:
    pretrain_config: str
    sft_config: str
    dpo_config: str
    baseline_checkpoint: str
    tokenizer_path: str = "data/tokenizer"
    data_dir: str = "data/v2"
    state_dir: str = "experiments/pipeline_tiny_v2"
    release_dir: str = "releases"
    release_version: str = "v0.1.0"
    device: str = "mps"
    offline_eval: bool = True
    gates: QualityGates = field(default_factory=QualityGates)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _default_train(config: ExperimentConfig, resume: bool) -> dict[str, Any]:
    result = run_training(config, resume=resume, max_restarts=2)
    return dict(result.__dict__)


def _default_eval(
    checkpoint: str,
    tokenizer_path: str,
    data_dir: str,
    device: str,
    offline: bool,
    name: str,
) -> tuple[dict[str, float], list[dict[str, str]]]:
    ckpt_path = Path(checkpoint)
    resolved = CheckpointManager(ckpt_path, keep=1).latest_valid() or ckpt_path
    model, _, _ = load_model_for_inference(resolved)
    tokenizer = BPETokenizer.load(tokenizer_path)
    engine = InferenceEngine(model, tokenizer, resolve_device(device))
    evaluator = Evaluator(engine, EvaluationConfig(data_dir=data_dir))
    metrics = evaluator.run(include_dataset_suites=not offline)
    samples = evaluator.qualitative_samples()
    record_experiment(name, {"checkpoint": str(resolved)}, metrics)
    return metrics, samples


def _candidate_checkpoints(checkpoint_dir: Path, alias: str | None) -> list[Path]:
    candidates: list[Path] = []
    if alias:
        for suffix in ("best", "latest", "step_1000", "step_2000", "step_4000", "step_6000"):
            path = checkpoint_dir / f"{alias}_{suffix}"
            if path.exists():
                resolved = path.resolve()
                if resolved not in candidates:
                    candidates.append(resolved)
    latest = CheckpointManager(checkpoint_dir, keep=1).latest_valid()
    if latest is not None and latest.resolve() not in candidates:
        candidates.append(latest.resolve())
    return candidates


def _resolve_best_checkpoint(checkpoint_dir: Path, alias: str | None) -> Path:
    if alias:
        best = checkpoint_dir / f"{alias}_best"
        if best.exists():
            return best.resolve()
    latest = CheckpointManager(checkpoint_dir, keep=1).latest_valid()
    if latest is not None:
        return latest.resolve()
    return checkpoint_dir


class TrainingPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        train_fn: Callable[[ExperimentConfig, bool], dict[str, Any]] = _default_train,
        eval_fn: Callable[
            [str, str, str, str, bool, str], tuple[dict[str, float], list[dict[str, str]]]
        ] = _default_eval,
        export_fn: Callable[..., Path] = export_release,
    ) -> None:
        self.config = config
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.export_fn = export_fn
        self.state_dir = Path(config.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "state.json"
        self.log_path = self.state_dir / "pipeline.log"
        self.logger = self._setup_logger()
        self.state = self._load_state()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"llm.pipeline.{self.state_dir.name}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        return logger

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {
            "created_at": _now(),
            "updated_at": _now(),
            "stages": {name: {"status": "pending"} for name in STAGES},
            "artifacts": {},
        }

    def _save_state(self) -> None:
        self.state["updated_at"] = _now()
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_path)

    def _set_stage(self, stage: str, status: str, **payload: Any) -> None:
        entry = self.state["stages"].setdefault(stage, {})
        entry["status"] = status
        entry["updated_at"] = _now()
        entry.update(payload)
        self._save_state()

    def _run_stage(self, stage: str, handler: Callable[[], dict[str, Any]]) -> None:
        current = self.state["stages"].get(stage, {}).get("status")
        if current in {"completed", "skipped"}:
            self.logger.info("skipping %s (status=%s)", stage, current)
            return
        self.logger.info("starting stage %s", stage)
        self._set_stage(stage, "running", started_at=_now())
        try:
            result = handler()
        except Exception as error:
            self.logger.exception("stage %s failed", stage)
            self._set_stage(stage, "failed", error=str(error))
            raise
        status = result.pop("_status", "completed")
        self._set_stage(stage, status, **result)
        self.logger.info("finished stage %s (status=%s)", stage, status)

    def _stage_pretrain(self) -> dict[str, Any]:
        cfg = load_experiment_config(self.config.pretrain_config)
        result = self.train_fn(cfg, True)
        self.state["artifacts"]["pretrain_dir"] = cfg.train.checkpoint_dir
        self.state["artifacts"]["pretrain_alias"] = cfg.train.checkpoint_alias
        return {"metrics": result}

    def _stage_eval_pretrain(self) -> dict[str, Any]:
        baseline_metrics, baseline_samples = self.eval_fn(
            self.config.baseline_checkpoint,
            self.config.tokenizer_path,
            self.config.data_dir,
            self.config.device,
            self.config.offline_eval,
            "baseline_tiny_session",
        )
        cfg = load_experiment_config(self.config.pretrain_config)
        candidates = _candidate_checkpoints(
            Path(cfg.train.checkpoint_dir), cfg.train.checkpoint_alias
        )
        evaluated: dict[str, dict[str, float]] = {}
        samples_by_ckpt: dict[str, list[dict[str, str]]] = {}
        for path in candidates:
            metrics, samples = self.eval_fn(
                str(path),
                self.config.tokenizer_path,
                self.config.data_dir,
                self.config.device,
                self.config.offline_eval,
                f"pretrain_{path.name}",
            )
            evaluated[str(path)] = metrics
            samples_by_ckpt[str(path)] = samples
        self.state["artifacts"]["baseline_metrics"] = baseline_metrics
        self.state["artifacts"]["baseline_samples"] = baseline_samples
        self.state["artifacts"]["pretrain_eval"] = evaluated
        self.state["artifacts"]["pretrain_samples"] = samples_by_ckpt
        return {"baseline": baseline_metrics, "candidates": evaluated}

    def _stage_select_best_pretrain(self) -> dict[str, Any]:
        evaluated: dict[str, dict[str, float]] = self.state["artifacts"].get("pretrain_eval", {})
        baseline: dict[str, float] = self.state["artifacts"].get("baseline_metrics", {})
        if not evaluated:
            raise TrainingError("no pretrain checkpoints evaluated")
        best_path, best_metrics = min(
            evaluated.items(),
            key=lambda item: float(item[1].get("test_perplexity", float("inf"))),
        )
        gates = self.config.gates
        ppl = float(best_metrics.get("test_perplexity", float("inf")))
        base_ppl = float(baseline.get("test_perplexity", float("inf")))
        rep = float(best_metrics.get("repeated_trigram_ratio_greedy", 0.0))
        if not math.isfinite(ppl):
            raise TrainingError("best pretrain checkpoint has non-finite test_perplexity")
        if gates.require_pretrain_improvement and math.isfinite(base_ppl) and ppl >= base_ppl:
            raise TrainingError(
                f"pretrain quality gate failed: test_perplexity {ppl:.2f} >= baseline {base_ppl:.2f}"
            )
        if rep > gates.max_repetition_ratio:
            raise TrainingError(
                f"pretrain repetition gate failed: {rep:.3f} > {gates.max_repetition_ratio:.3f}"
            )
        self.state["artifacts"]["best_pretrain_checkpoint"] = best_path
        self.state["artifacts"]["best_pretrain_metrics"] = best_metrics
        return {"selected": best_path, "test_perplexity": ppl, "baseline_perplexity": base_ppl}

    def _stage_sft(self) -> dict[str, Any]:
        cfg = load_experiment_config(self.config.sft_config)
        cfg.train.init_from = self.state["artifacts"]["best_pretrain_checkpoint"]
        result = self.train_fn(cfg, True)
        best = _resolve_best_checkpoint(Path(cfg.train.checkpoint_dir), cfg.train.checkpoint_alias)
        self.state["artifacts"]["sft_checkpoint"] = str(best)
        return {"metrics": result, "checkpoint": str(best)}

    def _stage_eval_sft(self) -> dict[str, Any]:
        ckpt = self.state["artifacts"]["sft_checkpoint"]
        metrics, samples = self.eval_fn(
            ckpt,
            self.config.tokenizer_path,
            self.config.data_dir,
            self.config.device,
            self.config.offline_eval,
            "sft_tiny_v2",
        )
        self.state["artifacts"]["sft_metrics"] = metrics
        self.state["artifacts"]["sft_samples"] = samples
        return {"checkpoint": ckpt, "metrics": metrics}

    def _sft_quality_ok(self) -> tuple[bool, str]:
        sft_metrics: dict[str, float] = self.state["artifacts"].get("sft_metrics", {})
        pre_metrics: dict[str, float] = self.state["artifacts"].get("best_pretrain_metrics", {})
        gates = self.config.gates
        sft_ppl = float(sft_metrics.get("test_perplexity", float("inf")))
        pre_ppl = float(pre_metrics.get("test_perplexity", float("inf")))
        rep = float(sft_metrics.get("repeated_trigram_ratio_greedy", 0.0))
        non_empty = float(sft_metrics.get("non_empty_response_rate", 1.0))
        if not math.isfinite(sft_ppl):
            return False, "non-finite SFT test_perplexity"
        if math.isfinite(pre_ppl) and sft_ppl > pre_ppl * gates.max_sft_perplexity_ratio:
            return (
                False,
                f"SFT perplexity {sft_ppl:.2f} regressed > {gates.max_sft_perplexity_ratio}x pretrain {pre_ppl:.2f}",
            )
        if rep > gates.max_repetition_ratio:
            return False, f"SFT repetition {rep:.3f} > {gates.max_repetition_ratio:.3f}"
        if non_empty < gates.min_non_empty_rate:
            return (
                False,
                f"SFT non_empty_response_rate {non_empty:.2f} < {gates.min_non_empty_rate:.2f}",
            )
        return True, "ok"

    def _stage_dpo(self) -> dict[str, Any]:
        ok, reason = self._sft_quality_ok()
        if not ok:
            return {"_status": "skipped", "reason": reason}
        pref_file = Path(self.config.data_dir) / "preference" / "train.jsonl"
        if not pref_file.is_file():
            return {"_status": "skipped", "reason": f"missing preference data: {pref_file}"}
        cfg = load_experiment_config(self.config.dpo_config)
        cfg.train.init_from = self.state["artifacts"]["sft_checkpoint"]
        result = self.train_fn(cfg, True)
        best = _resolve_best_checkpoint(Path(cfg.train.checkpoint_dir), cfg.train.checkpoint_alias)
        self.state["artifacts"]["dpo_checkpoint"] = str(best)
        return {"metrics": result, "checkpoint": str(best)}

    def _stage_eval_final(self) -> dict[str, Any]:
        dpo_ckpt = self.state["artifacts"].get("dpo_checkpoint")
        sft_ckpt = self.state["artifacts"]["sft_checkpoint"]
        sft_metrics = self.state["artifacts"]["sft_metrics"]
        sft_samples = self.state["artifacts"].get("sft_samples", [])
        if dpo_ckpt:
            dpo_metrics, dpo_samples = self.eval_fn(
                dpo_ckpt,
                self.config.tokenizer_path,
                self.config.data_dir,
                self.config.device,
                self.config.offline_eval,
                "dpo_tiny_v2",
            )
            self.state["artifacts"]["dpo_metrics"] = dpo_metrics
            dpo_ok = (
                math.isfinite(float(dpo_metrics.get("test_perplexity", float("inf"))))
                and float(dpo_metrics.get("repeated_trigram_ratio_greedy", 1.0))
                <= self.config.gates.max_repetition_ratio
                and float(dpo_metrics.get("instruction_following_accuracy", 0.0))
                >= float(sft_metrics.get("instruction_following_accuracy", 0.0))
            )
            if dpo_ok:
                self.state["artifacts"]["final_checkpoint"] = dpo_ckpt
                self.state["artifacts"]["final_metrics"] = dpo_metrics
                self.state["artifacts"]["final_samples"] = dpo_samples
                return {"selected": dpo_ckpt, "source": "dpo", "metrics": dpo_metrics}
        self.state["artifacts"]["final_checkpoint"] = sft_ckpt
        self.state["artifacts"]["final_metrics"] = sft_metrics
        self.state["artifacts"]["final_samples"] = sft_samples
        return {"selected": sft_ckpt, "source": "sft", "metrics": sft_metrics}

    def _stage_export_release(self) -> dict[str, Any]:
        stage_metrics = {
            name: self.state["stages"].get(name, {}).get("metrics")
            for name in ("pretrain", "sft", "dpo")
        }
        release_path = self.export_fn(
            checkpoint_path=self.state["artifacts"]["final_checkpoint"],
            tokenizer_path=self.config.tokenizer_path,
            data_dir=self.config.data_dir,
            output_dir=self.config.release_dir,
            version=self.config.release_version,
            stage_metrics=stage_metrics,
            eval_metrics=self.state["artifacts"].get("final_metrics", {}),
            baseline_metrics=self.state["artifacts"].get("baseline_metrics", {}),
            samples=self.state["artifacts"].get("final_samples", []),
        )
        self.state["artifacts"]["release_path"] = str(release_path)
        return {"release_path": str(release_path)}

    def run(self) -> dict[str, Any]:
        handlers = {
            "pretrain": self._stage_pretrain,
            "eval_pretrain": self._stage_eval_pretrain,
            "select_best_pretrain": self._stage_select_best_pretrain,
            "sft": self._stage_sft,
            "eval_sft": self._stage_eval_sft,
            "dpo": self._stage_dpo,
            "eval_final": self._stage_eval_final,
            "export_release": self._stage_export_release,
        }
        for stage in STAGES:
            self._run_stage(stage, handlers[stage])
        return self.state
