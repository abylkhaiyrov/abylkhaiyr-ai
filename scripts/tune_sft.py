from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from llm.core.config import load_experiment_config
from llm.evaluation.comparison import load_engine
from llm.evaluation.engine import EvaluationConfig, Evaluator
from llm.inference.engine import InferenceEngine
from llm.optimization.memory import release_device_memory
from llm.tokenizer.bpe import BPETokenizer
from llm.training.experiments import (
    EfficiencyVariant,
    configure,
    fixed_validation,
    steps_for_budget,
)
from llm.training.runner import build_trainer


@dataclass(frozen=True)
class Trial:
    name: str
    learning_rate: float
    uniform: bool = False
    replay_fraction: float = 0.0


TRIALS = (
    Trial("lower_lr", 3e-5),
    Trial("higher_lr", 8e-5),
    Trial("uniform", 8e-5, uniform=True),
    Trial("replay", 8e-5, replay_fraction=0.03),
)


def language_validation(engine: InferenceEngine, data_dir: str) -> dict[str, float]:
    evaluator = Evaluator(engine, EvaluationConfig(data_dir, corpus_split="validation"))
    return evaluator.language_modeling()


def main() -> int:
    parser = argparse.ArgumentParser(prog="tune_sft")
    parser.add_argument("--config", default="configs/tiny_v31_sft.yaml")
    parser.add_argument("--corpus", default="data/v2")
    parser.add_argument("--budget", type=int, default=120_000)
    parser.add_argument("--output", default="experiments/stage11/pilots.json")
    args = parser.parse_args()
    base = load_experiment_config(args.config)
    tokenizer = BPETokenizer.load(base.tokenizer.path)
    corpus = Path(args.corpus)
    initial = base.train.init_from
    if not initial:
        parser.error("the config must specify a pretrained checkpoint")
    engine = load_engine(initial, base.tokenizer.path, base.train.device)
    baseline = language_validation(engine, args.corpus)
    device = engine.device
    del engine
    release_device_memory(device)
    payload: dict[str, object] = {"baseline": baseline, "target_tokens": args.budget}
    reports: list[dict[str, object]] = []
    payload["trials"] = reports
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for trial in TRIALS:
        train = base.train.model_copy(
            update={
                "sft_replay_dir": str(corpus / "tokens"),
                "sft_replay_fraction": trial.replay_fraction,
                "optimizer": base.train.optimizer.model_copy(update={"foreach": True}),
            }
        )
        config = base.model_copy(update={"train": train})
        mixture = dict.fromkeys(train.mixture, 1.0) if trial.uniform else train.mixture
        variant = EfficiencyVariant(
            trial.name, mixture, pack=True, learning_rate=trial.learning_rate
        )
        steps = steps_for_budget(config, variant, tokenizer, args.budget)
        with tempfile.TemporaryDirectory() as workdir:
            run = configure(config, variant, steps, Path(workdir))
            trainer = build_trainer(run, resume=False)
            result = trainer.train()
            assistant_loss = fixed_validation(trainer, base, tokenizer)
            engine = InferenceEngine(trainer.model, tokenizer, device)
            report: dict[str, object] = {
                **asdict(trial),
                "steps": result.final_step,
                "objective_tokens": result.tokens_seen,
                "assistant_validation_loss": assistant_loss,
                **language_validation(engine, args.corpus),
            }
            del engine, trainer
        release_device_memory(device)
        reports.append(report)
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report), flush=True)
        if result.interrupted:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
