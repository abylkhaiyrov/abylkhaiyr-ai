from llm.training.continuous import Example, decide, split_rows


def test_prompt_variants_stay_in_one_split():
    examples = [
        Example(prompt="What Result?", answer="One."),
        Example(prompt="  what  result?", answer="Two."),
    ]
    assert sorted(map(len, split_rows(examples))) == [0, 2]


def test_improvement_does_not_override_retention_gate():
    reference = {
        "conv_pass_rate": 0.5,
        "conv_lang_ru": 0.5,
        "conv_repetition_ratio": 0.0,
        "test_loss": 2.0,
        "arithmetic_exact_match": 0.3,
    }
    candidate = {**reference, "conv_pass_rate": 0.6, "test_loss": 2.4}
    assert decide(reference, candidate) == ["held-out corpus perplexity increased by over 10%"]
    assert decide(reference, {**candidate, "test_loss": 2.0}) == []


def test_gate_rejects_missing_language_and_nonfinite_metrics():
    reference = {
        "conv_pass_rate": 0.5,
        "conv_lang_ru": 0.5,
        "conv_repetition_ratio": 0.0,
        "test_loss": 2.0,
        "arithmetic_exact_match": 0.3,
    }
    candidate = {**reference, "conv_pass_rate": 0.6}
    del candidate["conv_lang_ru"]
    assert decide(reference, candidate) == ["missing or non-finite quality metrics"]
    assert decide(reference, {**reference, "test_loss": float("nan")}) == [
        "missing or non-finite quality metrics"
    ]
