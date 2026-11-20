import torch

from llm.evaluation import suites
from llm.evaluation.engine import EvaluationConfig, Evaluator, suite_count
from llm.evaluation.hooks import NullHooks
from llm.inference.engine import InferenceEngine
from llm.model.transformer import CausalLM


class Recorder(NullHooks):
    def __init__(self):
        self.planned = []
        self.done = 0
        self.started = []
        self.finished = []

    def suite_started(self, name, index, total):
        self.started.append((name, index, total))

    def items_planned(self, total):
        self.planned.append(total)

    def item_done(self, count=1):
        self.done += count

    def suite_finished(self, name, metrics, seconds):
        self.finished.append((name, sorted(metrics)))


def test_evaluator_reports_planned_and_completed_items_per_suite(model, tokenizer):
    engine = InferenceEngine(CausalLM(model.config), tokenizer, torch.device("cpu"))
    hooks = Recorder()
    evaluator = Evaluator(engine, EvaluationConfig(data_dir="unused", arithmetic_cases=3), hooks)
    evaluator.arithmetic()
    evaluator.factual()
    assert hooks.planned == [3, len(suites.FACTUAL_CASES)]
    assert hooks.done == sum(hooks.planned)


def test_run_announces_every_suite_in_order(model, tokenizer, monkeypatch):
    engine = InferenceEngine(CausalLM(model.config), tokenizer, torch.device("cpu"))
    hooks = Recorder()
    evaluator = Evaluator(engine, EvaluationConfig(data_dir="unused"), hooks)
    names = [suite.__name__ for suite in evaluator.suites_for_run(False)]
    for name in names:
        replacement = lambda name=name: {name: 1.0}
        replacement.__name__ = name
        monkeypatch.setattr(evaluator, name, replacement)
    evaluator.run(include_dataset_suites=False)
    assert [item[0] for item in hooks.started] == names
    assert [item[1] for item in hooks.started] == list(range(suite_count(False)))
    assert {item[2] for item in hooks.started} == {suite_count(False)}
    assert len(hooks.finished) == suite_count(False)
