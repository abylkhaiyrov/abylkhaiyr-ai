import re

import pytest

from llm.core.config import ModelConfig
from llm.data import lexicon as lx
from llm.data.identity import (
    ANSWERS,
    QUESTIONS,
    IdentityFacts,
    build_identity_dataset,
    load_identity_facts,
)
from llm.data.instructgen import CATEGORIES, build_instruction_dataset
from llm.data.sft_quality import (
    PromptBlocklist,
    QualityRules,
    SFTQualityFilter,
    equations_hold,
    is_off_topic,
    looks_kyrgyz,
    repair_gsm8k,
    script_mix,
)
from llm.evaluation.conversation import (
    CASES,
    Check,
    WordList,
    benchmark_prompts,
    evaluate_check,
    run_conversation_benchmark,
)
from llm.evaluation.prompts import evaluation_prompts
from llm.reasoning.tools import evaluate_expression


@pytest.fixture(scope="module")
def facts() -> IdentityFacts:
    model = ModelConfig(
        vocab_size=8192,
        hidden_size=384,
        num_hidden_layers=8,
        num_attention_heads=6,
        num_key_value_heads=2,
        intermediate_size=1024,
        max_position_embeddings=512,
    )
    return load_identity_facts(model)


def row(user: str, assistant: str, source: str = "dolly_sft", group: str = "chat") -> dict:
    return {
        "source": source,
        "group": group,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def test_identity_facts_come_from_model_config(facts):
    assert facts.parameters_millions == 16
    assert facts.context_tokens == 512
    assert "16" in facts.values("ru")["params"]


def test_identity_answers_match_project_facts(facts):
    rows = build_identity_dataset(facts, 400, 0)
    assert len(rows) > 100
    text = " ".join(m["content"] for r in rows for m in r["messages"] if m["role"] == "assistant")
    assert "Abylkhaiyr AI" in text
    assert "{" not in text
    assert "GPT" not in text and "OpenAI" not in text


def test_every_identity_intent_has_questions_and_answers_in_all_languages():
    for intent, by_language in QUESTIONS.items():
        for language in lx.LANGUAGES:
            assert by_language[language]
            assert ANSWERS[intent][language]


def test_instruction_generator_is_deterministic_and_unique(facts):
    first = build_instruction_dataset(facts, 600, 3)
    second = build_instruction_dataset(facts, 600, 3)
    assert first == second
    keys = [str(r["messages"]) for r in first]
    assert len(keys) == len(set(keys))
    assert {r["category"] for r in first} == set(CATEGORIES)


def test_context_questions_are_answerable_from_the_text(facts):
    rows = [r for r in build_instruction_dataset(facts, 800, 5) if r["category"] == "context_qa"]
    assert rows
    for item in rows:
        prompt, answer = item["messages"][0]["content"], item["messages"][1]["content"]
        if answer in {
            "The text does not say that.",
            "В тексте об этом не сказано.",
            "Мәтінде бұл туралы айтылмаған.",
        }:
            continue
        values = re.findall(r"\d+|[^\W\d_]{3,}", answer)
        assert any(value in prompt for value in values)


def test_multi_turn_arithmetic_is_correct(facts):
    rows = [
        r for r in build_instruction_dataset(facts, 800, 6) if r["category"] == "multi_turn_math"
    ]
    assert rows
    for item in rows:
        first, second = item["messages"][1]["content"], item["messages"][3]["content"]
        a, b, total = map(int, re.findall(r"\d+", first)[:3])
        assert a + b == total
        s, k, product = map(int, re.findall(r"\d+", second)[:3])
        assert s == total and s * k == product


def test_translation_rows_match_lexicon(facts):
    rows = [r for r in build_instruction_dataset(facts, 800, 7) if r["category"] == "translation"]
    terms = [*lx.NOUNS, *lx.NUMBERS, *lx.COLORS, *lx.PHRASES]
    flat = {value for t in terms for value in (t.en, t.ru, t.kk)}
    sentences = {
        lx.SENTENCE_FRAMES[code].format(word=noun.in_language(code))
        for noun in lx.NOUNS
        for code in lx.LANGUAGES
    }
    assert rows
    for item in rows:
        answer = item["messages"][1]["content"]
        quoted = re.findall(r"[“«]([^”»]+)[”»]", answer)
        assert answer in sentences or (quoted and all(value in flat for value in quoted))


def test_synthetic_prompts_never_match_independent_benchmark_after_blocklist(facts):
    blocklist = PromptBlocklist(evaluation_prompts())
    rows = build_instruction_dataset(facts, 3000, 1) + build_identity_dataset(facts, 400, 1)
    kept = [r for r in rows if not blocklist.blocks(r)]
    prompts = {p.strip().lower() for p in benchmark_prompts()}
    assert kept
    assert all(r["messages"][0]["content"].strip().lower() not in prompts for r in kept)


def test_blocklist_catches_benchmark_prompt_and_paraphrase_ngrams():
    blocklist = PromptBlocklist(
        ["Anna is taller than Bota. Bota is taller than Dana. Who is shortest?"]
    )
    assert blocklist.blocks(
        row("anna is taller than bota. bota is taller than dana. who is shortest?", "Dana.")
    )
    assert not blocklist.blocks(row("What is the capital of France?", "Paris is the capital."))


def test_benchmark_entities_are_disjoint_from_generator_pools():
    pools = {name.lower() for names in lx.PEOPLE.values() for name in names}
    pools |= {city.in_language(code).lower() for city in lx.CITIES for code in lx.LANGUAGES}
    text = " ".join(benchmark_prompts()).lower()
    assert not any(re.search(rf"\b{re.escape(name)}\b", text) for name in pools)


def test_quality_filter_rejects_known_defects():
    rules = QualityRules(source_caps={"capped": 1})
    filt = SFTQualityFilter(rules)
    assert filt.reason(row("Q?", "")) == "empty_message"
    kyrgyz = (
        "Манас эпосу жөнүндө кандай билесиз? Бул эпос кыргыз элинин улуу чыгармасы жана тарыхы."
    )
    assert filt.reason(row("Расскажи", kyrgyz)) == "kyrgyz_text"
    assert (
        filt.reason(row("Who are you?", "I am Open Assistant, made by LAION."))
        == "identity_conflict"
    )
    long_cut = "This sentence goes on and on without a proper ending " * 6
    assert filt.reason(row("Describe it", long_cut.strip())) == "incomplete_answer"
    mixed = "Это очень важный вопрос и мой ответ на него такой: The answer is forty two for sure"
    assert filt.reason(row("Почему?", mixed)) == "mixed_scripts"
    assert (
        filt.reason(
            row(
                "Tell me about the history of Rome and the empire",
                "Bananas are yellow and sweet fruit that monkeys like to eat daily, especially in warm tropical places.",
            )
        )
        == "off_topic"
    )
    assert filt.reason(row("What is 2 + 2?", "It is 4.")) is None
    assert filt.admit(row("a b c one", "Fine answer one.", source="capped"))
    assert not filt.admit(row("a b c two", "Fine answer two.", source="capped"))


def test_kazakh_text_is_not_flagged_as_kyrgyz():
    kazakh = "Қазақстан Республикасының астанасы — Астана қаласы, ол еліміздің саяси орталығы."
    assert not looks_kyrgyz(kazakh)
    assert not looks_kyrgyz("Москва является столицей России и крупнейшим городом страны.")


def test_gsm8k_artifact_repair_and_arithmetic_verification():
    broken = "Mara has 2 x 12 = >24 marbles.\n#### 24"
    assert repair_gsm8k(broken) == "Mara has 2 x 12 = 24 marbles.\n#### 24"
    assert equations_hold(repair_gsm8k(broken))
    assert not equations_hold("He has 3 + 4 = 8 apples.")
    assert script_mix("Привет") == 0.0
    assert not is_off_topic("short q", "x" * 200)


def test_evaluate_expression_survives_malformed_input():
    assert evaluate_expression("2 + 3)") is None
    assert evaluate_expression("((2 + 3") is None
    assert not evaluate_expression("1/0").is_finite
    assert evaluate_expression("2 + 3") == 5


def test_conversation_checks():
    assert evaluate_check(Check("number", ("42",)), "The answer is 42.")
    assert not evaluate_check(Check("number", ("42",)), "The answer is 142.")
    assert evaluate_check(Check("max_words", number=3), "Green.")
    assert not evaluate_check(Check("max_words", number=3), "")
    assert evaluate_check(Check("sentences", number=2), "It is big. It is blue.")
    assert evaluate_check(Check("min_items", number=3), "red, blue, green")
    assert evaluate_check(Check("exact", ("HELLO",)), "HELLO")
    assert not evaluate_check(Check("exact", ("HELLO",)), "hello")
    assert evaluate_check(Check("none", ("aktau",)), "Madina is from Oral.")
    assert evaluate_check(Check("script", ("ru",)), "Привет, как дела?")
    assert not evaluate_check(Check("script", ("en",)), "Привет")


def test_benchmark_scores_reference_answers_and_penalizes_garbage():
    references = {
        item.id: next(
            (
                value
                for check in item.checks
                if check.kind in {"any", "all", "exact"}
                for value in check.value
            ),
            "",
        )
        for item in CASES
    }
    words = WordList.from_texts(["hello world this is a test sentence"] * 3)
    good = run_conversation_benchmark(lambda item: (references[item.id], True), words).metrics
    bad = run_conversation_benchmark(lambda item: ("zzz qqq xxx", False), words).metrics
    assert good["conv_completion_rate"] == 1.0
    assert bad["conv_completion_rate"] == 0.0
    assert good["conv_pass_rate"] > bad["conv_pass_rate"]
    assert bad["conv_pass_rate"] < 0.1


def test_benchmark_case_ids_are_unique_and_cover_required_categories():
    ids = [item.id for item in CASES]
    assert len(ids) == len(set(ids))
    categories = {item.category for item in CASES}
    required = {
        "greeting",
        "identity",
        "explain",
        "context",
        "math",
        "translation",
        "instruction",
        "clarify",
        "unknown",
        "multi_turn",
        "logic",
        "code",
    }
    assert required <= categories
    assert {item.language for item in CASES} == {"en", "ru", "kk"}


def test_preference_pairs_share_prompt_and_differ_in_answer(facts):
    import random

    from llm.data.preference_builder import synthetic_pairs

    rows = build_instruction_dataset(facts, 400, 2) + build_identity_dataset(facts, 100, 2)
    pairs = synthetic_pairs(rows, 200, random.Random(0))
    assert len(pairs) > 150
    for pair in pairs:
        assert pair["chosen"][:-1] == pair["rejected"][:-1]
        assert pair["chosen"][-1]["content"].strip() != pair["rejected"][-1]["content"].strip()
        assert pair["kind"] == "corruption"
    on_policy = synthetic_pairs(
        rows, 50, random.Random(0), sampler=lambda messages: "I do not know"
    )
    assert {pair["kind"] for pair in on_policy} <= {"on_policy", "corruption"}
    assert sum(pair["kind"] == "on_policy" for pair in on_policy) > 40


def test_sampling_sweep_picks_best_variant_per_group(model, tokenizer):
    import torch

    from llm.evaluation.comparison import (
        SamplingVariant,
        category_table,
        markdown_table,
        sweep_sampling,
    )
    from llm.inference.engine import InferenceEngine
    from llm.inference.sampling import SamplingParams
    from llm.model.transformer import CausalLM

    engine = InferenceEngine(CausalLM(model.config), tokenizer, torch.device("cpu"))
    variants = [
        SamplingVariant("greedy", SamplingParams(greedy=True, max_new_tokens=4)),
        SamplingVariant("sampled", SamplingParams(temperature=0.8, max_new_tokens=4)),
    ]
    result = sweep_sampling(engine, variants, seeds=(0,))
    assert len(result.rows) == 2 * 5
    assert set(result.best_by_group()) == {"dialogue", "explanation", "math", "code", "instruction"}
    reports = {"m": run_conversation_benchmark(lambda item: ("x y z", True), None)}
    assert "conv_pass_rate" in markdown_table(reports)
    assert "greeting" in category_table(reports)


def test_suite_count_matches_the_suites_actually_run(model, tokenizer):
    import torch

    from llm.evaluation.engine import EvaluationConfig, Evaluator, suite_count
    from llm.inference.engine import InferenceEngine
    from llm.model.transformer import CausalLM

    engine = InferenceEngine(CausalLM(model.config), tokenizer, torch.device("cpu"))
    evaluator = Evaluator(engine, EvaluationConfig(data_dir="unused"))
    for online in (False, True):
        assert len(evaluator.suites_for_run(online)) == suite_count(online)


def test_equation_check_handles_fraction_results():
    assert equations_hold("3/2 * 1 = <<(3/2)*(1/1)=3/2>>3/2.")
    assert equations_hold("4/3 - 8/5 = -4/15")
    assert not equations_hold("1/2 + 1/3 = 2/5")


def test_validation_sweep_scores_overlap_completion_and_repetition(model, tokenizer, tmp_path):
    import json

    import torch

    from llm.evaluation.comparison import (
        SamplingVariant,
        sweep_validation,
        token_f1,
        validation_cases,
    )
    from llm.inference.engine import InferenceEngine
    from llm.inference.sampling import SamplingParams
    from llm.model.transformer import CausalLM

    assert token_f1("the cat sat", "the cat sat") == 1.0
    assert token_f1("a b", "c d") == 0.0
    assert 0 < token_f1("the cat", "the cat sat") < 1
    groups = tmp_path / "sft_groups"
    groups.mkdir()
    row = {
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    }
    (groups / "validation.dialog.jsonl").write_text(json.dumps(row) + "\n")
    cases = validation_cases(str(tmp_path), per_group=5)
    assert list(cases) == ["dialogue"]
    engine = InferenceEngine(CausalLM(model.config), tokenizer, torch.device("cpu"))
    variants = [SamplingVariant("g", SamplingParams(greedy=True, max_new_tokens=3))]
    result = sweep_validation(engine, variants, cases)
    assert len(result.rows) == 1 and 0 <= result.rows[0].pass_rate <= 1
