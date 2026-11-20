from __future__ import annotations

from pathlib import Path

from llm.data.mathgen import MATH_BENCH_FILE, read_dataset
from llm.evaluation import suites
from llm.evaluation.conversation import benchmark_prompts


def evaluation_prompts(benchmark_dir: str | Path | None = None) -> list[str]:
    texts = [*suites.all_prompts(), *benchmark_prompts()]
    if benchmark_dir is not None:
        path = Path(benchmark_dir) / MATH_BENCH_FILE
        if path.is_file():
            texts.extend(problem.question for problem in read_dataset(path))
    return texts
