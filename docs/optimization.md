# Optimization and sample efficiency (stages 9–10)

Everything measured lives in `docs/stage10_report.md`, which `scripts/build_report.py` generates from the result
files in `experiments/`. This page describes the mechanisms and how to reproduce each measurement.

## Training speed

| Mechanism | Where | Default |
|---|---|---|
| One device sync per optimizer step instead of two per micro-batch (token counts computed on host batches, losses accumulated on device) | `training/engine.py` | always on |
| Multi-tensor AdamW (`torch._foreach_*`, same math, verified against `torch.optim.AdamW`) | `optimization/adamw.py`, `optimizer.foreach` | decided by benchmark |
| Native GQA in SDPA instead of materialised `repeat_kv` | `model/attention.py`, `train.native_gqa` | decided by benchmark |
| `is_causal` instead of a boolean mask for unpadded full sequences | `model/transformer.py`, `train.fast_causal` | decided by benchmark |
| `torch.nn.functional.silu` | `model/mlp.py` | always on (bit-identical result) |
| Batches padded to a multiple of 64 tokens (few distinct shapes on MPS) | `data/binary.py` | always on |
| Sequence packing with segment-isolated attention and per-segment positions | `PackedConversationLoader`, `train.pack_sequences` | decided by experiment |
| Length bucketing | `ConversationLoader(bucket_window=…)`, `train.bucket_window` | decided by experiment |
| Curriculum ramp of the mixture | `MixtureLoader(start_weights, ramp_batches)` | decided by experiment |
| Token-based gradient accumulation (loss divided by the supervised tokens of the whole step) | `training/engine.py` | always on, covered by `test_sft_loss_is_invariant_to_how_a_step_is_split_into_micro_batches` |

Data loading is not a bottleneck (0.3–1.5 ms per micro-batch, about 0.5% of a step), so no prefetch thread was added.

```bash
python scripts/profile_training.py --variants baseline foreach native_gqa fast_causal gqa_causal packed packed_gqa_causal bf16 mb4 mb16 ckpt_mb16 --repeats 3 --stability-steps 8 --phases
python scripts/efficiency_experiments.py --variants padded_plain padded_weighted bucketed_weighted packed_weighted curriculum_weighted --budget-tokens 400000 --quality
python scripts/inference_benchmark.py checkpoints/tiny_v2/tiny_best
python scripts/build_report.py
```

`profile_training.py` reports useful (supervised) tokens per second, slot tokens per second, the useful fraction, peak device memory,
swap, the median of repeated runs and, for non-packed variants, the largest loss difference against the first variant over the same steps.

## Sample efficiency protocol

All variants start from the same pretrained checkpoint and see the same number of supervised tokens (`steps_for_budget` measures
the supervised tokens per step of each variant and sets the step count accordingly). They are compared on one fixed padded validation
mixture with equal weight per group, never on their own training mixture, and on the conversational benchmark.
Throughput alone is never a reason to adopt a method.

## Data (v3.1)

Groups: `chat_{en,ru,kk,other}`, `reading_{en,ru,kk}`, `skills`, `dialog`, `identity`, `math`, `word_math`, `code`.

* `reading_*`: tasks built from real Russian, Kazakh and English corpus passages (sentence count, copy first/last sentence,
  language identification, list numbers, longest word, word present yes/no, which sentence contains a word). Every answer is computed from the passage.
  They add natural-language ru/kk instruction data without machine translation. They teach reading and copying, not open-ended chat.
* `code`: 26 new verified specs (algorithms, data structures, optimisation pairs with complexity, bit tricks, dynamic programming),
  JavaScript and TypeScript variants, bug-fix, test-writing, optimisation, complexity and concept-explanation tasks (SOLID, Clean Code, DRY, KISS, refactoring,
  recursion, Big O, stack, queue, hash table, linked list, OOP). SQL is capped at 400 examples. Every solution runs against tests in the sandbox and
  every mutant is checked to fail. Function names are disjoint from the independent benchmarks (tested).
* `chat_kk` contains only 149 real examples: the open instruction datasets contain almost no clean Kazakh (6.8k Kyrgyz examples were removed). It is sampled at a low
  weight to avoid memorisation; Kazakh coverage comes from `reading_kk`, `skills` and `dialog`.
* Language mixture weights can be re-weighted from evaluation scores: `scripts/make_language_weighted_config.py`.

Language identification on 2,026 held-out corpus sentences: Kazakh 99.0%, Russian 96.8%, English 95.1% (the rest is labelled "other", never another language).

## Reasoning engine

`reasoning/triage.py` decides, before any sampling, whether a request needs clarification, can be solved or refuted by a tool, or needs the model.

* Clarification: a missing expression, text to translate or summarise, target language, or task description produces a request for exactly what is missing
  (not for follow-up turns that already have context).
* Tool solvers (`reasoning/solvers.py`): arithmetic, percentages, equations and systems, inequalities, derivatives, integrals, determinants, mean/median, logarithms, probability,
  ordering. They return `solved`, `contradiction` (cyclic orderings, inconsistent equations) or `underdetermined` (identities), or abstain.
* Model candidates are still verified (annotations, calculator repair, logic check, code syntax and execution in the sandbox) and voted.
* Three separately reported accuracies: model only, model with verifier, model with tools; plus `tool_dev`, `tool_holdout` and `triage_holdout` sets.
  The template benchmark is generated from the same templates the solvers were tuned for, so it is not evidence of generality; the holdouts are.
