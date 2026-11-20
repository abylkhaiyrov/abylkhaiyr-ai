# Instruction tuning, evaluation and promotion (stage 8)

Root cause of the incoherent answers: Studio served a **pretrained-only** checkpoint
(`tiny_v2/step-00001000`, loaded once at start-up by modification time). A base language model
continues text, it does not follow instructions. The fix is instruction tuning plus a selection
mechanism based on measured conversational quality, not on step numbers.

## Inference audit (verified by tests in `tests/test_instruction_audit.py`)

| Item | Result |
|---|---|
| Chat template | training sequence and generation prompt share the exact prefix `<bos> [<role> text <eos>]* <assistant>`; verified for system + multi-turn |
| BOS/EOS, roles | assistant supervision starts after the `<assistant>` token and includes `<eos>`; role tokens and `<bos>` are masked |
| Attention mask / positions | right padding is causal-safe; padded loss of the short sequence equals its unpadded loss |
| KV cache | cached and uncached generation are identical (existing test) |
| Special tokens | user text cannot inject `<eos>`/role tokens (existing test) |
| Context management | whole oldest turns are dropped (`fit_messages`), system prompt and the latest user message are kept |
| **Fixed: repetition penalty** | it was applied to the prompt tokens too, which penalises copying names and numbers from the user text; now only generated tokens are penalised (`SamplingParams.penalize_prompt` restores the old behaviour) |
| **Fixed: `evaluate_expression`** | unbalanced parentheses raised `IndexError` inside SymPy and could crash the reasoning verifier on model output |
| **Fixed: pipeline** | a training run stopped by SIGINT/SIGTERM was recorded as a completed stage; it now fails the stage and is resumed |

## SFT audit

* Shift: `targets[t] == ids[t+1]`; only assistant tokens and the closing `<eos>` are supervised.
* Loss is a sum divided by supervised tokens of the whole accumulated step.
* Batches are right-padded to a multiple of 64 tokens (few distinct shapes, faster on MPS); padding is masked.
* There is no sequence packing in SFT: one conversation per row, so no cross-example attention leakage.
* Over-long conversations are removed at build time instead of being truncated (truncation would drop `<eos>`).
* `test_controlled_overfit_reproduces_training_answers_only` shows the model can memorise instruction pairs. It proves
  the training loop works, **not** quality on new prompts; quality is measured only on the independent benchmark.

## Data v3 (`scripts/build_instruction_corpus.py`)

Groups: `chat` (filtered dolly/aya/oasst1), `skills`, `dialog`, `identity`, `math`, `word_math`, `code`.

* `skills`: context QA with distractors and unanswerable variants, text analysis (word/sentence counts, longest word,
  reversal), lists, ru/en/kk translation from a lexicon, multi-turn context (name memory, chained arithmetic, translation follow-up).
* `dialog`: greetings, thanks, farewells, clarification of unclear requests, admitting missing knowledge, unknown entities, facts, concepts.
* `identity`: generated from `configs/identity.json` and the real model config (parameter count, context, hardware); no backend hard-coding.
* Quality filters (`data/sft_quality.py`): Kyrgyz text mislabelled as Kazakh, mixed scripts, incomplete answers, repetition,
  off-topic answers, base-model identity conflicts ("Open Assistant", "as an AI language model"), template/source caps,
  token limit, GSM8K artefact repair (`= >24`) and arithmetic verification.
* Decontamination: exact + 6-gram match against every benchmark prompt (suites, conversational benchmark, math benchmark).
* No third-party LLM outputs: all synthetic data comes from generators and verifiers in this repository.

## Conversational benchmark (`evaluation/conversation.py`)

55 independent cases in ru/en/kk (greeting, identity, explanation, context use, math, translation, instruction following,
clarification, admitting ignorance, multi-turn, logic, code). Entities and wordings are disjoint from the generators
(tested). Scoring is programmatic, no LLM judge: content checks, language/script stability, completion (ended with `<eos>`),
trigram repetition, well-formed-word rate against held-out corpus vocabulary. Fixed generation parameters: greedy, 96 tokens, repetition penalty 1.1.

## Preference optimisation (`data/preference_builder.py`)

Pairs share the prompt. Chosen = verified reference answer. Rejected = the SFT model's own sampled answer when it
differs (on-policy), otherwise a corruption (irrelevant answer, repetition loop, truncation, wrong number, wrong language,
foreign identity). A slice of human-ranked oasst1 pairs is mixed in. DPO is only run if SFT improves the conversational
pass rate, and its checkpoint is only selected if it does not regress it.

## Intent detection (`reasoning/intent.py`)

dialogue, information, math, logic, code, analysis, translation. Only math/logic/code use SymPy, the calculator or the
sandbox; everything else goes through the model. The model-only and with-tools scores are reported separately.

## Selection of the production model (`training/promotion.py`)

`scripts/promote_model.py` writes `checkpoints/production.json`. A checkpoint is eligible only if it is SFT/DPO, beats
the pretrained reference on `conv_pass_rate` and is not repetitive; ties prefer lower repetition. `ready` stays false
below a pass rate of 0.5 and Studio must then label the model as an experimental preview. Known limitations are stored in the manifest.

## Commands

```bash
python scripts/build_instruction_corpus.py
python scripts/run_pipeline.py --sft configs/tiny_v3_sft.yaml --dpo configs/tiny_v3_dpo.yaml --build-preference
python scripts/compare_conversations.py base=checkpoints/tiny_session sft=checkpoints/tiny_v3_sft/tiny_v3_sft_best
python scripts/sampling_sweep.py checkpoints/tiny_v3_sft/tiny_v3_sft_best
python scripts/promote_model.py experiments/conversation_compare.json sft=checkpoints/tiny_v3_sft/tiny_v3_sft_best=sft --reference base
```
