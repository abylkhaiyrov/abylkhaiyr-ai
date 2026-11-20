from __future__ import annotations

import copy
from pathlib import Path

from llm.core.config import ExperimentConfig, TrainConfig
from llm.core.errors import ConfigError, ResourceError
from llm.core.hardware import detect_hardware, resolve_device, resolve_dtype
from llm.core.profiles import (
    estimate_training_memory_gb,
    suggest_compact_profile,
    suggest_micro_batch,
)
from llm.data.binary import (
    PAD_MULTIPLE,
    ConversationLoader,
    MixtureLoader,
    PackedConversationLoader,
    PackedTokenLoader,
    token_dtype,
)
from llm.data.preference import PreferenceLoader
from llm.model.transformer import CausalLM
from llm.optimization.memory import configure_threads
from llm.tokenizer.bpe import BPETokenizer
from llm.training.checkpoint import CheckpointManager, load_model_weights
from llm.training.engine import Trainer, seed_everything
from llm.training.monitor import TrainingMonitor
from llm.training.objectives import DPOObjective, Objective

MEMORY_BUDGET_FRACTION = 0.7


def _load_init_weights(model: CausalLM, init_from: str) -> None:
    checkpoint = CheckpointManager(init_from, keep=1).latest_valid()
    path = checkpoint if checkpoint is not None else Path(init_from)
    CheckpointManager.verify(path)
    load_model_weights(model, path / "model.safetensors")


def _check_memory(config: ExperimentConfig, hardware_memory_gb: float) -> None:
    train = config.train
    required = estimate_training_memory_gb(
        config.model, train.min_micro_batch_size, train.sequence_length
    )
    if required > hardware_memory_gb * MEMORY_BUDGET_FRACTION:
        raise ResourceError(
            f"model needs about {required:.1f} GB at the smallest micro batch but only "
            f"{hardware_memory_gb:.1f} GB is available; use a more compact profile"
        )


CURRICULUM_GROUPS = ("math",)


def _conversation_mixture(
    root: Path,
    split: str,
    weights: dict[str, float],
    length: int,
    batch: int,
    tokenizer: BPETokenizer,
    seed: int,
    train: TrainConfig | None = None,
) -> MixtureLoader:
    dtype = token_dtype(tokenizer.vocab_size)
    loaders = {}
    for name in weights:
        prefix = root / "sft_groups" / f"{split}.{name}"
        if prefix.parent.joinpath(f"{prefix.name}.ids.bin").is_file():
            shuffle = split == "train" and name not in CURRICULUM_GROUPS
            pack = train is not None and train.pack_sequences
            loader_class = PackedConversationLoader if pack else ConversationLoader
            window = train.bucket_window if train is not None and not pack else 0
            loaders[name] = loader_class(
                prefix,
                dtype,
                length + 1,
                batch,
                tokenizer.pad_id,
                seed,
                shuffle,
                pad_multiple=train.pad_multiple if train is not None else PAD_MULTIPLE,
                bucket_window=window,
            )
    start = None
    if train is not None and train.curriculum_start and split == "train":
        start = {name: train.curriculum_start.get(name, weights[name]) for name in loaders}
    ramp = train.curriculum_batches if train is not None else 0
    return MixtureLoader(
        loaders, {name: weights[name] for name in loaders}, seed, start, ramp if start else 0
    )


def _with_replay(
    source: MixtureLoader, train: TrainConfig, tokenizer: BPETokenizer
) -> MixtureLoader:
    if not train.sft_replay_dir or train.curriculum_start:
        raise ConfigError("SFT replay needs a token directory and a fixed SFT mixture")
    root = Path(train.sft_replay_dir)
    languages = ("en", "ru", "kk")
    loaders = dict(source.loaders)
    weights = {
        name: float(weight) * (1 - train.sft_replay_fraction)
        for name, weight in zip(source.names, source.final, strict=True)
    }
    for language in languages:
        name = f"replay_{language}"
        if name in loaders:
            raise ConfigError(f"reserved SFT replay group: {name}")
        loaders[name] = PackedTokenLoader(
            root / f"train.{language}.bin",
            token_dtype(tokenizer.vocab_size),
            train.sequence_length,
            train.micro_batch_size,
            train.seed,
        )
        weights[name] = train.sft_replay_fraction / len(languages)
    return MixtureLoader(loaders, weights, train.seed)


def _build_loaders(config: ExperimentConfig, tokenizer: BPETokenizer):
    train, root = config.train, Path(config.train.data_dir)
    dtype = token_dtype(tokenizer.vocab_size)
    batch, length, seed = train.micro_batch_size, train.sequence_length, train.seed
    if train.sft_replay_fraction and (train.stage != "sft" or not train.mixture):
        raise ConfigError("SFT replay requires a grouped SFT dataset")
    if train.stage == "pretrain":
        if train.mixture:
            groups = {
                name: PackedTokenLoader(
                    root / "tokens" / f"train.{name}.bin", dtype, length, batch, seed
                )
                for name in train.mixture
            }
            training_source: MixtureLoader | PackedTokenLoader = MixtureLoader(
                groups, train.mixture, seed
            )
        else:
            training_source = PackedTokenLoader(
                root / "tokens" / "train.bin", dtype, length, batch, seed
            )
        return (
            training_source,
            PackedTokenLoader(
                root / "tokens" / "validation.bin", dtype, length, batch, seed, False
            ),
        )
    if train.stage == "sft" and train.mixture:
        training = _conversation_mixture(
            root, "train", train.mixture, length, batch, tokenizer, seed, train
        )
        validation = _conversation_mixture(
            root, "validation", train.mixture, length, batch, tokenizer, seed
        )
        if train.sft_replay_fraction:
            training = _with_replay(training, train, tokenizer)
        return training, validation
    if train.stage == "sft":
        make = lambda split, shuffle: ConversationLoader(
            root / "sft" / split, dtype, length + 1, batch, tokenizer.pad_id, seed, shuffle
        )
        return make("train", True), make("validation", False)
    return (
        PreferenceLoader(root, "train", tokenizer, length + 1, batch, seed),
        PreferenceLoader(root, "validation", tokenizer, length + 1, batch, seed, False),
    )


def build_trainer(config: ExperimentConfig, resume: bool) -> Trainer:
    train = config.train
    if train.sequence_length > config.model.max_position_embeddings:
        raise ConfigError("sequence_length exceeds max_position_embeddings")
    device = resolve_device(train.device)
    dtype = resolve_dtype(train.precision, device)
    configure_threads(train.num_threads)
    hardware = detect_hardware(train.device)
    _check_memory(config, hardware.device_memory_gb)
    tokenizer = BPETokenizer.load(config.tokenizer.path)
    if tokenizer.vocab_size != config.model.vocab_size:
        raise ConfigError(
            f"model vocab_size {config.model.vocab_size} != tokenizer vocab {tokenizer.vocab_size}"
        )
    seed_everything(train.seed)
    model = CausalLM(config.model)
    checkpoints = CheckpointManager(
        train.checkpoint_dir, train.keep_checkpoints, train.protected_steps, train.checkpoint_alias
    )
    if train.init_from:
        _load_init_weights(model, train.init_from)
    objective: Objective | None = None
    if train.stage == "dpo":
        reference = copy.deepcopy(model).to(device)
        objective = DPOObjective(reference, train.dpo_beta, device, dtype)
    train_source, validation_source = _build_loaders(config, tokenizer)
    monitor = TrainingMonitor(train.output_dir, device, train.max_steps)
    trainer = Trainer(
        model,
        config.model,
        train,
        train_source,
        validation_source,
        device,
        dtype,
        checkpoints,
        monitor,
        tokenizer.version,
        objective,
    )
    if train.init_from and train.restore_optimizer and not resume:
        source = CheckpointManager(train.init_from, keep=1).latest_valid() or Path(train.init_from)
        CheckpointManager.load_optimizer_state(source, trainer.optimizer)
        for group, decay in zip(
            trainer.optimizer.param_groups, (train.optimizer.weight_decay, 0.0), strict=True
        ):
            group["weight_decay"] = decay
    if resume:
        trainer.resume()
    return trainer


def recommend_micro_batch(config: ExperimentConfig) -> int:
    hardware = detect_hardware(config.train.device)
    return suggest_micro_batch(config.model, config.train.sequence_length, hardware)


def recommend_profile(profile: str, vocab_size: int) -> str:
    return suggest_compact_profile(profile, vocab_size, detect_hardware())
