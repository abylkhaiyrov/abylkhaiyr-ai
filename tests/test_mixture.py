from llm.training.mixture import adjust_for_languages, group_language, language_scores


def test_languages_with_lower_scores_receive_more_weight_without_touching_other_groups():
    weights = {"chat_en": 0.1, "chat_ru": 0.1, "reading_kk": 0.1, "math": 0.1}
    adjusted = adjust_for_languages(weights, {"en": 0.8, "ru": 0.4, "kk": 0.0})
    assert adjusted["math"] == weights["math"]
    assert adjusted["chat_en"] < adjusted["chat_ru"] < adjusted["reading_kk"]
    assert adjusted["reading_kk"] == 0.2


def test_group_language_detection_and_score_extraction():
    assert group_language("chat_kk") == "kk" and group_language("chat_other") is None
    assert group_language("math") is None
    assert language_scores({"conv_lang_en": 0.5, "conv_pass_rate": 1.0}) == {"en": 0.5}


def test_sft_replay_is_training_only_and_counts_full_language_targets(
    tmp_path, model_config, tokenizer
):
    import json

    import numpy as np

    from llm.core.config import ExperimentConfig, TrainConfig
    from llm.data.binary import token_dtype, tokenize_sft_split
    from llm.training.runner import _build_loaders

    groups = tmp_path / "sft_groups"
    groups.mkdir()
    messages = {
        "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    }
    for split in ("train", "validation"):
        source = groups / f"{split}.chat.jsonl"
        source.write_text(json.dumps(messages) + "\n")
        tokenize_sft_split(source, groups / f"{split}.chat", tokenizer)
    replay = tmp_path / "tokens"
    replay.mkdir()
    for language in ("en", "ru", "kk"):
        np.arange(100, dtype=token_dtype(tokenizer.vocab_size)).tofile(
            replay / f"train.{language}.bin"
        )
    config = ExperimentConfig(
        name="sft-replay",
        model=model_config,
        train=TrainConfig(
            stage="sft",
            data_dir=str(tmp_path),
            mixture={"chat": 1.0},
            sequence_length=31,
            micro_batch_size=2,
            pack_sequences=True,
            sft_replay_dir=str(replay),
            sft_replay_fraction=0.03,
        ),
    )
    train, validation = _build_loaders(config, tokenizer)

    assert validation.names == ["chat"]
    assert dict(zip(train.names, train.probabilities, strict=True)) == {
        "chat": 0.97,
        "replay_en": 0.01,
        "replay_kk": 0.01,
        "replay_ru": 0.01,
    }
    _, labels = train.loaders["replay_kk"].next_batch()
    assert labels.numel() == 62
    assert (labels >= 0).all()
    assert len(validation.next_batch()) == 2
