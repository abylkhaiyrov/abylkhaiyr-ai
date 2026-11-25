from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from llm.core.config import ExperimentConfig, ModelConfig, TokenizerConfig, TrainConfig
from llm.data.binary import tokenize_sft_split
from llm.data.grounded import (
    ANSWER,
    REFUSAL,
    row_of,
    training_examples,
)
from llm.training.exposure_audit import audit, classify, group_catalog


def write_group(directory: Path, name: str, rows: list[dict], tokenizer) -> None:
    path = directory / f"train.{name}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    tokenize_sft_split(path, directory / f"train.{name}", tokenizer)


def chat_rows(count: int) -> list[dict]:
    return [
        {
            "category": "qa",
            "messages": [
                {"role": "user", "content": f"Question number {i}?"},
                {"role": "assistant", "content": f"Answer number {i} is here."},
            ],
        }
        for i in range(count)
    ]


def config_of(root: Path, tokenizer, model_config: ModelConfig, steps: int) -> ExperimentConfig:
    train = TrainConfig(
        stage="sft",
        data_dir=str(root),
        max_steps=steps,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        sequence_length=63,
        pack_sequences=True,
        mixture={"reading_en": 0.5, "chat_en": 0.5},
        device="cpu",
        precision="fp32",
        seed=3,
    )
    return ExperimentConfig(
        name="audit",
        model=model_config,
        tokenizer=TokenizerConfig(vocab_size=tokenizer.vocab_size, path="unused"),
        train=train,
    )


def test_classification_separates_surface_answerable_and_unanswerable_categories() -> None:
    assert classify("grounded_missing") == "unanswerable"
    assert classify("grounded_extract") == "answerable"
    assert classify("reading_first_sentence") == "surface"


def test_missing_share_controls_the_unanswerable_fraction_and_keeps_pairs_together() -> None:
    for share in (0.25, 0.5):
        examples = training_examples("en", 600, 1, missing_share=share)
        refused = sum(e.plan.expects == REFUSAL for e in examples)
        assert abs(refused / len(examples) - share) < 0.01
    twins = Counter(e.family for e in training_examples("en", 200, 1))
    assert set(twins.values()) == {2}
    assert row_of(training_examples("en", 4, 1)[0], 0)["family"] == "train/en/0"
    assert training_examples("en", 200, 1)[0].plan.expects == ANSWER


def test_audit_counts_batches_tokens_and_unanswerable_examples(
    tmp_path, tokenizer, model_config
) -> None:
    groups = tmp_path / "sft_groups"
    groups.mkdir()
    examples = training_examples("en", 80, 1)
    write_group(groups, "reading_en", [row_of(e, 0) for e in examples], tokenizer)
    write_group(groups, "chat_en", chat_rows(60), tokenizer)
    config = config_of(tmp_path, tokenizer, model_config, 6)
    catalog = {"reading_en": group_catalog(groups / "train.reading_en", "en", 1)}
    report = audit(config, tokenizer, catalog)
    steps = report["steps"]
    assert len(steps) == 6
    assert all(sum(item["batches"].values()) == 2 for item in steps)
    summary = report["summary"]
    assert (
        summary["groups"]["reading_en"]["batches"] + summary["groups"]["chat_en"]["batches"] == 12
    )
    taken = sum(v["examples"] for v in summary["reading_examples"].values())
    assert (
        taken
        == summary["reading_examples_taken"]
        == sum(
            item.get("examples", {}).get("answerable", 0)
            + item.get("examples", {}).get("unanswerable", 0)
            for item in steps
        )
    )
    assert summary["repeat_exposure"] >= 1.0
    assert summary["sft_supervised_tokens"] > 0
