export type ChartDefinition = {
  id: string;
  metric: string;
  titleKey: string;
  goal: "min" | "max" | "none";
  unitKey?: string;
  kind: "loss" | "plain" | "scientific" | "percent" | "rate";
  allowLog?: boolean;
  stages?: string[];
};

export const RUN_CHARTS: ChartDefinition[] = [
  { id: "train_loss", metric: "train/loss", titleKey: "chart.trainLoss", goal: "min", kind: "loss", allowLog: true },
  { id: "validation_loss", metric: "validation/loss", titleKey: "chart.validationLoss", goal: "min", kind: "loss", allowLog: true },
  { id: "perplexity", metric: "validation/perplexity", titleKey: "chart.perplexity", goal: "min", kind: "plain", allowLog: true },
  { id: "learning_rate", metric: "train/learning_rate", titleKey: "chart.learningRate", goal: "none", kind: "scientific" },
  { id: "grad_norm", metric: "train/grad_norm", titleKey: "chart.gradNorm", goal: "min", kind: "plain" },
  { id: "tokens_per_second", metric: "perf/tokens_per_second", titleKey: "chart.tokensPerSecond", goal: "max", unitKey: "units.tokensPerSecond", kind: "rate" },
  { id: "reward_accuracy", metric: "dpo/reward_accuracy", titleKey: "chart.rewardAccuracy", goal: "max", kind: "percent", stages: ["dpo"] },
  { id: "reward_margin", metric: "dpo/reward_margin", titleKey: "chart.rewardMargin", goal: "max", kind: "plain", stages: ["dpo"] },
];

export const SYSTEM_CHARTS = [
  { id: "ram", titleKey: "chart.ram" },
  { id: "cpu", titleKey: "chart.cpu" },
  { id: "mps", titleKey: "chart.mps" },
  { id: "swap", titleKey: "chart.swap" },
] as const;
