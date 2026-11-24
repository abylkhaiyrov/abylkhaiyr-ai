from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from llm.core.errors import ConfigError, LLMError

DEFAULT_CONFIG = "configs/debug.yaml"


def _tokenizer_and_engine(checkpoint: str, tokenizer_path: str, device_name: str):
    from llm.core.hardware import resolve_device
    from llm.inference.engine import InferenceEngine
    from llm.tokenizer.bpe import BPETokenizer
    from llm.training.checkpoint import CheckpointManager, load_model_for_inference

    path = Path(checkpoint)
    latest = CheckpointManager(path, keep=1).latest_valid()
    model, _, metadata = load_model_for_inference(latest or path)
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer.validate_checkpoint(model.config.vocab_size, metadata.get("tokenizer_version"))
    return InferenceEngine(model, tokenizer, resolve_device(device_name))


def _sampling(args: argparse.Namespace):
    from llm.inference.sampling import SamplingParams

    return SamplingParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        beam_width=args.beams,
        greedy=args.greedy,
        stop_sequences=args.stop or [],
        seed=args.seed,
    )


def evaluate_and_print(campaign, spec) -> int:
    report = campaign.evaluate_campaign(spec)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 4 if report["accepted"] is False else 0


def cmd_experiment(args: argparse.Namespace) -> int:
    from llm.training import campaign

    spec, path = campaign.open_spec(args.spec)
    if args.action == "plan":
        print(json.dumps(campaign.build_plan(spec), indent=2))
    elif args.action == "status":
        print(json.dumps(campaign.status(spec), indent=2))
    elif args.action == "run":
        reason = campaign.run_campaign(spec, path)
        print(json.dumps({"stopped": reason}))
        if reason is not None:
            return 3
        return evaluate_and_print(campaign, spec) if spec.evaluation else 0
    elif args.action == "baseline":
        print(json.dumps(campaign.evaluate_baseline(spec), indent=2, ensure_ascii=False))
    elif args.action == "evaluate":
        return evaluate_and_print(campaign, spec)
    elif args.job is None:
        raise ConfigError("the job action needs a job name")
    else:
        print(json.dumps(campaign.run_job(spec, args.job, args.active_seconds)), flush=True)
    return 0


def cmd_kernels(args: argparse.Namespace) -> int:
    from llm.core.hardware import resolve_device
    from llm.optimization.kernels import AttentionShape, measure_attention, report

    shape = AttentionShape(args.batch, args.heads, args.kv_heads, args.length, args.head_dim)
    timings = measure_attention(shape, resolve_device(args.device), args.repeats)
    print(json.dumps({"shape": shape.__dict__, "variants": report(timings)}, indent=2))
    return 0


def cmd_diagnostics(args: argparse.Namespace) -> int:
    from llm.data.probes import write_diagnostics
    from llm.data.sft_profile import profile_sft
    from llm.tokenizer.bpe import BPETokenizer

    if args.sft_profile:
        print(json.dumps(profile_sft(Path(args.sft_profile)), indent=2, ensure_ascii=False))
        return 0
    if args.hidden:
        from llm.evaluation.comparison import load_engine
        from llm.evaluation.grounded import load_cases
        from llm.evaluation.hidden import diagnose

        engine = load_engine(args.hidden, args.tokenizer, args.device)
        print(json.dumps(diagnose(engine, load_cases(args.cases), args.limit), indent=2))
        return 0
    if args.context_probes:
        from llm.data.probes import write_context_probes

        print(json.dumps(write_context_probes(Path(args.context_probes)), indent=2))
        return 0
    if args.grounded:
        from llm.data.grounded_blocks import write_grounded

        summary = write_grounded(
            Path(args.grounded),
            Path(args.base),
            BPETokenizer.load(args.tokenizer),
            args.seed,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    summary = write_diagnostics(Path(args.output), BPETokenizer.load(args.tokenizer))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_hardware(args: argparse.Namespace) -> int:
    from llm.core.hardware import detect_hardware
    from llm.core.profiles import PROFILES, available_profiles

    hardware = detect_hardware(args.device)
    print(json.dumps(hardware.as_dict(), indent=2))
    print("runnable profiles:", ", ".join(available_profiles(args.vocab_size, hardware)))
    print("all profiles:", ", ".join(PROFILES))
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    from llm.core.config import DataConfig, TokenizerConfig, load_yaml
    from llm.data.pipeline import CorpusBuilder
    from llm.data.prepare import tokenize_dataset, train_tokenizer
    from llm.tokenizer.bpe import BPETokenizer

    raw = load_yaml(args.data_config)
    data_config = DataConfig.model_validate(raw.get("data", raw))
    tokenizer_config = TokenizerConfig.model_validate(raw.get("tokenizer", {}))
    if args.vocab_size:
        tokenizer_config.vocab_size = args.vocab_size
    if not args.tokenize_only:
        print(json.dumps(CorpusBuilder(data_config).build()["splits"]))
    if args.skip_tokenize:
        return 0
    token_file = Path(tokenizer_config.path) / "tokenizer.json"
    tokenizer = (
        BPETokenizer.load(token_file.parent)
        if token_file.is_file() and not args.retrain_tokenizer
        else train_tokenizer(data_config.output_dir, tokenizer_config)
    )
    print(json.dumps(tokenize_dataset(data_config.output_dir, tokenizer)))
    return 0


def cmd_train_tokenizer(args: argparse.Namespace) -> int:
    from llm.core.config import TokenizerConfig
    from llm.data.prepare import roundtrip_check, tokenizer_training_texts, train_tokenizer

    config = TokenizerConfig(vocab_size=args.vocab_size, path=args.output)
    tokenizer = train_tokenizer(args.data_dir, config)
    sample = [t for _, t in zip(range(args.check_samples), tokenizer_training_texts(args.data_dir))]
    print(
        f"vocab={tokenizer.vocab_size} version={tokenizer.version} roundtrip_ok={roundtrip_check(tokenizer, sample)}"
    )
    return 0


def _train(args: argparse.Namespace, resume: bool, finetuning: bool = False) -> int:
    from llm.core.config import load_experiment_config
    from llm.training.session import run_training

    config = load_experiment_config(args.config)
    if args.stage:
        config.train.stage = args.stage
    elif finetuning and config.train.stage == "pretrain":
        config.train.stage = "sft"
    if args.max_steps:
        config.train.max_steps = args.max_steps
    if args.init_from:
        config.train.init_from = args.init_from
    result = run_training(config, resume, args.stop_step, args.max_restarts)
    print(json.dumps(result.__dict__))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    return _train(args, resume=False)


def cmd_resume(args: argparse.Namespace) -> int:
    return _train(args, resume=True)


def cmd_finetune(args: argparse.Namespace) -> int:
    return _train(args, resume=args.resume, finetuning=True)


def cmd_evaluate(args: argparse.Namespace) -> int:
    from llm.evaluation.engine import EvaluationConfig, Evaluator

    engine = _tokenizer_and_engine(args.checkpoint, args.tokenizer, args.device)
    evaluator = Evaluator(
        engine, EvaluationConfig(data_dir=args.data_dir, gsm8k_cases=args.gsm8k_cases)
    )
    results = evaluator.run(include_dataset_suites=not args.offline)
    evaluator.save(results, args.output)
    if args.name:
        from llm.evaluation.registry import record_experiment

        record_experiment(args.name, {"checkpoint": args.checkpoint, "note": args.note}, results)
    print(json.dumps(results, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from llm.evaluation.registry import comparison_table, load_registry

    entries = [e for e in load_registry() if not args.names or e["name"] in args.names]
    print(comparison_table(entries))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    engine = _tokenizer_and_engine(args.checkpoint, args.tokenizer, args.device)
    params = _sampling(args)
    from llm.tokenizer.bpe import Message

    if args.chat:
        pieces = engine.stream_chat([Message("user", args.prompt)], params)
    else:
        pieces = engine.stream_text(engine.tokenizer.encode(args.prompt, add_bos=True), params)
    if not args.chat:
        print(args.prompt, end="")
    for piece in pieces:
        print(piece, end="", flush=True)
    print()
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from llm.tokenizer.bpe import Message

    engine = _tokenizer_and_engine(args.checkpoint, args.tokenizer, args.device)
    params = _sampling(args)
    history = [Message("system", args.system)] if args.system else []
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line in {"/exit", "/quit"}:
            return 0
        if not line:
            continue
        history.append(Message("user", line))
        reply = []
        print("bot> ", end="", flush=True)
        for piece in engine.stream_chat(history, params):
            reply.append(piece)
            print(piece, end="", flush=True)
        print()
        history.append(Message("assistant", "".join(reply)))


def _serving_app(args: argparse.Namespace):
    from llm.api.server import create_app
    from llm.training.checkpoint import CheckpointManager

    path = Path(args.checkpoint)
    latest = CheckpointManager(path, keep=1).latest_valid() if path.is_dir() else None
    resolved = (latest or path).resolve()
    root = Path.cwd().resolve()
    checkpoint_id = str(resolved.relative_to(root)) if root in resolved.parents else str(resolved)
    model_name = (
        resolved.parent.name
        if args.model_name == "llm-local" and resolved.parent.name
        else args.model_name
    )
    engine = _tokenizer_and_engine(str(resolved), args.tokenizer, args.device)
    return create_app(
        engine,
        model_name,
        studio_root=root,
        checkpoint_id=checkpoint_id,
        tokenizer_path=Path(args.tokenizer),
    )


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(_serving_app(args), host=args.host, port=args.port, log_level="info")
    return 0


def cmd_studio(args: argparse.Namespace) -> int:
    from llm.api.server import create_app
    from llm.core.errors import ConfigError
    from llm.studio.launcher import DEV_HOST, check_port, prepare_frontend, run_studio
    from llm.studio.runtime import RuntimeState

    root = Path.cwd().resolve()
    frontend = root / "frontend"
    check_port(args.host, args.port)
    if args.dev:
        if args.host != DEV_HOST or args.port == args.frontend_port:
            raise ConfigError(
                "Studio --dev requires backend host 127.0.0.1 and distinct frontend and backend ports"
            )
        check_port(DEV_HOST, args.frontend_port)
    if args.checkpoint is None:
        checkpoint = _latest_checkpoint_root(root)
        args.checkpoint = str(checkpoint) if checkpoint else None
    if args.device == "auto" and RuntimeState(root).training_active():
        args.device = "cpu"
    prepare_frontend(frontend, args.dev)
    if args.checkpoint is None:
        print(
            "Studio started without a model. Load a checkpoint from Models when available.",
            file=sys.stderr,
        )
        app = create_app(
            None, args.model_name, studio_root=root, tokenizer_path=Path(args.tokenizer)
        )
    else:
        app = _serving_app(args)
    return run_studio(app, frontend, args.host, args.port, args.frontend_port if args.dev else None)


def cmd_benchmark(args: argparse.Namespace) -> int:
    from llm.evaluation.engine import EvaluationConfig, Evaluator

    engine = _tokenizer_and_engine(args.checkpoint, args.tokenizer, args.device)
    results = Evaluator(engine, EvaluationConfig(data_dir=args.data_dir)).speed_and_memory()
    print(json.dumps(results, indent=2))
    return 0


def _latest_checkpoint_root(root: Path) -> Path | None:
    from llm.training.continuous import active_checkpoint
    from llm.training.promotion import production_checkpoint

    approved = production_checkpoint(root)
    if approved is not None:
        checkpoint = (root / approved).resolve()
        if checkpoint.is_relative_to((root / "checkpoints").resolve()):
            return checkpoint
    promoted = active_checkpoint(root / "experiments" / "continuous")
    if promoted is not None:
        checkpoint = (root / promoted.checkpoint).resolve()
        if checkpoint.is_relative_to(root / "checkpoints") and (checkpoint / "meta.json").is_file():
            return checkpoint
    metadata = [
        path
        for path in (root / "checkpoints").glob("*/step-*/meta.json")
        if not path.parent.name.endswith(".tmp")
    ]
    if not metadata:
        return None
    dialogue = [
        path
        for path in metadata
        if any(stage in path.parent.parent.name.split("_") for stage in ("sft", "dpo"))
    ]
    selected = max(dialogue or metadata, key=lambda item: item.stat().st_mtime).parent.parent
    if dialogue:
        for label in ("best", f"{selected.name}_best"):
            best = selected / label
            if (best / "meta.json").is_file():
                return best
    return selected


def add_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--stop", action="append")
    parser.add_argument("--seed", type=int)


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer")
    parser.add_argument("--device", default="auto")


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["pretrain", "sft", "dpo"])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--init-from")
    parser.add_argument("--max-restarts", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm")
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler: Callable[[argparse.Namespace], int]) -> argparse.ArgumentParser:
        sub = commands.add_parser(name)
        sub.set_defaults(handler=handler)
        return sub

    hardware = add("hardware", cmd_hardware)
    hardware.add_argument("--device", default="auto")
    hardware.add_argument("--vocab-size", type=int, default=8192)

    prepare = add("prepare-data", cmd_prepare_data)
    prepare.add_argument("--data-config", default="configs/data_small.yaml")
    prepare.add_argument("--vocab-size", type=int)
    prepare.add_argument("--skip-tokenize", action="store_true")
    prepare.add_argument("--tokenize-only", action="store_true")
    prepare.add_argument("--retrain-tokenizer", action="store_true")

    tokenizer = add("train-tokenizer", cmd_train_tokenizer)
    tokenizer.add_argument("--data-dir", default="data/processed")
    tokenizer.add_argument("--output", default="data/tokenizer")
    tokenizer.add_argument("--vocab-size", type=int, default=8192)
    tokenizer.add_argument("--check-samples", type=int, default=500)

    train = add("train", cmd_train)
    add_training_arguments(train)
    resume = add("resume", cmd_resume)
    add_training_arguments(resume)
    finetune = add("finetune", cmd_finetune)
    add_training_arguments(finetune)
    finetune.add_argument("--resume", action="store_true")

    evaluate = add("evaluate", cmd_evaluate)
    add_model_arguments(evaluate)
    evaluate.add_argument("--data-dir", default="data/processed")
    evaluate.add_argument("--output", default="experiments/evaluation.json")
    evaluate.add_argument("--gsm8k-cases", type=int, default=50)
    evaluate.add_argument("--offline", action="store_true")
    evaluate.add_argument("--name")
    evaluate.add_argument("--note", default="")

    compare = add("compare", cmd_compare)
    compare.add_argument("names", nargs="*")

    generate = add("generate", cmd_generate)
    add_model_arguments(generate)
    add_sampling_arguments(generate)
    generate.add_argument("prompt")
    generate.add_argument("--chat", action="store_true")

    chat = add("chat", cmd_chat)
    add_model_arguments(chat)
    add_sampling_arguments(chat)
    chat.add_argument("--system")

    serve = add("serve", cmd_serve)
    add_model_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model-name", default="llm-local")

    studio = add("studio", cmd_studio)
    studio.add_argument("--checkpoint")
    studio.add_argument("--tokenizer", default="data/tokenizer")
    studio.add_argument("--device", default="auto")
    studio.add_argument("--host", default="127.0.0.1")
    studio.add_argument("--port", type=int, default=8000)
    studio.add_argument("--model-name", default="llm-local")
    studio.add_argument("--dev", action="store_true")
    studio.add_argument("--frontend-port", type=int, default=5173)

    experiment = add("experiment", cmd_experiment)
    experiment.add_argument(
        "action", choices=("plan", "status", "run", "baseline", "evaluate", "job")
    )
    experiment.add_argument("spec")
    experiment.add_argument("job", nargs="?")
    experiment.add_argument("--active-seconds", type=float)

    kernels = add("kernels", cmd_kernels)
    kernels.add_argument("--device", default="auto")
    kernels.add_argument("--batch", type=int, default=8)
    kernels.add_argument("--heads", type=int, default=6)
    kernels.add_argument("--kv-heads", type=int, default=2)
    kernels.add_argument("--length", type=int, default=512)
    kernels.add_argument("--head-dim", type=int, default=64)
    kernels.add_argument("--repeats", type=int, default=7)

    diagnostics = add("diagnostics", cmd_diagnostics)
    diagnostics.add_argument("--output", default="data/diagnostics")
    diagnostics.add_argument("--tokenizer", default="data/tokenizer")
    diagnostics.add_argument("--sft-profile")
    diagnostics.add_argument("--grounded")
    diagnostics.add_argument("--hidden")
    diagnostics.add_argument("--cases", default="data/grounded/dev.jsonl")
    diagnostics.add_argument("--limit", type=int, default=60)
    diagnostics.add_argument("--device", default="cpu")
    diagnostics.add_argument("--context-probes")
    diagnostics.add_argument("--base", default="data/v3_1")
    diagnostics.add_argument("--seed", type=int, default=1)

    benchmark = add("benchmark", cmd_benchmark)
    add_model_arguments(benchmark)
    benchmark.add_argument("--data-dir", default="data/processed")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except LLMError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
