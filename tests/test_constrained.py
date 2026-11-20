import random

import torch

from llm.agent.tools import TOOL_ARGUMENTS, ToolCall, parse_tool_call
from llm.inference.constrained import ToolGrammar
from llm.inference.engine import InferenceEngine
from llm.inference.sampling import SamplingParams
from llm.model.transformer import CausalLM
from llm.tokenizer.bpe import Message


def accepts(grammar: ToolGrammar, text: str) -> bool:
    state = grammar.state
    for char in text:
        state = grammar._advance_char(state, char)
        if state is None:
            return False
    return True


def test_automaton_accepts_every_canonical_call_and_rejects_deviations(tokenizer):
    grammar = ToolGrammar(tokenizer, TOOL_ARGUMENTS)
    samples = [
        ToolCall(tool="run_tests", args={}),
        ToolCall(tool="read_file", args={"path": "solution.py"}),
        ToolCall(
            tool="write_file", args={"path": "solution.py", "content": 'x = "q"\nprint(1)\n\t#é'}
        ),
        ToolCall(tool="edit_file", args={"path": "a.py", "old": "a\\b", "new": "c"}),
        ToolCall(tool="finish", args={"summary": "done"}),
    ]
    for call in samples:
        assert accepts(grammar, call.text())
    for broken in (
        '{"tool": "rm", ',
        '{"tool": "read_file", "args": {"file": ',
        '{"tool": "run_tests", "args": {"x"',
        '{"tool": "finish", "args": {"summary": "a\nb',
        '{"tool": "finish", "args": {"summary": "a\\q',
        '[{"tool"',
    ):
        assert not accepts(grammar, broken)


def test_random_walk_over_masked_tokens_always_yields_a_valid_tool_call(tokenizer):
    grammar = ToolGrammar(tokenizer, TOOL_ARGUMENTS)
    quote = next(i for i, t in enumerate(grammar.texts) if t == '"')
    produced = 0
    for seed in range(60):
        rng = random.Random(seed)
        grammar.reset()
        pieces: list[str] = []
        for _ in range(900):
            allowed = grammar.mask()
            choices = allowed.nonzero().flatten().tolist()
            assert choices
            token = quote if allowed[quote] and rng.random() < 0.25 else rng.choice(choices)
            if token == grammar.eos_id:
                break
            grammar.advance(token)
            pieces.append(grammar.texts[token])
        else:
            continue
        parse_tool_call("".join(pieces))
        produced += 1
    assert produced > 40


def test_engine_constrains_a_random_model_to_the_tool_schema(model, tokenizer):
    torch.manual_seed(0)
    engine = InferenceEngine(CausalLM(model.config), tokenizer, torch.device("cpu"))
    grammar = ToolGrammar(tokenizer, TOOL_ARGUMENTS)
    params = SamplingParams(temperature=1.0, max_new_tokens=40, seed=3)
    result = engine.chat([Message("user", "do it")], params, constraint=grammar)
    state = ToolGrammar(tokenizer, TOOL_ARGUMENTS)
    assert accepts(state, result.text)
    plain = engine.chat([Message("user", "do it")], params)
    assert not accepts(ToolGrammar(tokenizer, TOOL_ARGUMENTS), plain.text)
