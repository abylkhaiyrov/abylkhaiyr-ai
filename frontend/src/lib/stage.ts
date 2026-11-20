import type { PipelineStage, StageKey } from "@/lib/api";

export type MetricKind = "int" | "compact" | "loss" | "ppl" | "pct" | "sci" | "num" | "rate" | "text";

export const STAGE_ORDER: StageKey[] = [
  "data_preparation",
  "tokenization",
  "pretrain",
  "eval_pretrain",
  "select_best_pretrain",
  "sft",
  "eval_sft",
  "dpo",
  "eval_final",
  "export_release",
];

export const METRIC_KINDS: Record<string, MetricKind> = {
  final_step: "int",
  step: "int",
  tokens_seen: "compact",
  train_loss: "loss",
  validation_loss: "loss",
  last_validation_loss: "loss",
  best_validation_loss: "loss",
  test_loss: "loss",
  validation_perplexity: "ppl",
  test_perplexity: "ppl",
  baseline_test_perplexity: "ppl",
  baseline_perplexity: "ppl",
  next_token_accuracy: "pct",
  arithmetic_exact_match: "pct",
  math_bench_model_accuracy: "pct",
  math_bench_tool_accuracy: "pct",
  instruction_following_accuracy: "pct",
  code_pass_at_1: "pct",
  repeated_trigram_ratio_greedy: "pct",
  reward_accuracy: "pct",
  reward_margin: "num",
  grad_norm: "num",
  learning_rate: "sci",
  tokens_per_second: "rate",
  generation_tokens_per_second: "rate",
  sources: "int",
  documents: "int",
  train_tokens: "compact",
  checkpoints_evaluated: "int",
  best_checkpoint: "text",
  checkpoint: "text",
  source: "text",
  release_path: "text",
};

const TRAIN_ORDER = ["final_step", "validation_loss", "validation_perplexity", "best_validation_loss", "tokens_seen", "train_loss", "reward_accuracy", "reward_margin", "learning_rate", "grad_norm", "tokens_per_second"];
const EVAL_ORDER = ["checkpoints_evaluated", "best_checkpoint", "test_perplexity", "baseline_test_perplexity", "test_loss", "next_token_accuracy", "instruction_following_accuracy", "math_bench_model_accuracy", "arithmetic_exact_match", "code_pass_at_1", "repeated_trigram_ratio_greedy", "generation_tokens_per_second"];

export const INTERIM_METRICS = ["test_perplexity", "next_token_accuracy", "arithmetic_exact_match", "math_bench_model_accuracy", "instruction_following_accuracy", "code_pass_at_1"];

export function orderedResults(stage: PipelineStage): [string, number | string][] {
  const order = stage.kind === "train" ? TRAIN_ORDER : stage.kind === "eval" ? EVAL_ORDER : Object.keys(stage.results);
  const entries: [string, number | string][] = [];
  for (const key of order) {
    const value = stage.results[key];
    if (value !== null && value !== undefined) {
      entries.push([key, value]);
    }
  }
  if (stage.kind === "train") {
    const final = entries.find(([key]) => key === "final_step");
    if (!final && stage.done !== null) entries.unshift(["step", stage.done]);
    const hasValidation = entries.some(([key]) => key === "validation_loss");
    return entries.filter(([key]) => !(hasValidation && key === "best_validation_loss" && stage.status !== "completed"));
  }
  return entries;
}

export function headlineResults(stage: PipelineStage, count = 3): [string, number | string][] {
  return orderedResults(stage).slice(0, count);
}

export function isTrainStage(key: StageKey) {
  return key === "pretrain" || key === "sft" || key === "dpo";
}

export const ACTION_KEYS: Record<StageKey, string> = {
  data_preparation: "preparing",
  tokenization: "tokenizing",
  pretrain: "training",
  eval_pretrain: "evaluating",
  select_best_pretrain: "selecting",
  sft: "training",
  eval_sft: "evaluating",
  dpo: "training",
  eval_final: "evaluating",
  export_release: "exporting",
};
