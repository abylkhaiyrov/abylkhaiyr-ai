import type { RunEntry, RunStatus } from "@/lib/api";

export const PAGE_SIZE = 40;
export const MAX_COMPARE_RUNS = 4;
export const STATUSES: RunStatus[] = ["running", "completed", "stopped", "failed"];
export const STAGES = ["pipeline", "pretrain", "sft", "dpo"] as const;

export type HistoryFilters = { q: string; status: RunStatus | ""; stage: string; version: string; sort: "newest" | "oldest" };
export const EMPTY_FILTERS: HistoryFilters = { q: "", status: "", stage: "", version: "", sort: "newest" };

export function toQuery(filters: HistoryFilters, offset: number) {
  return {
    q: filters.q.trim() || undefined,
    status: filters.status || undefined,
    stage: filters.stage || undefined,
    version: filters.version.trim() || undefined,
    sort: filters.sort,
    limit: PAGE_SIZE,
    offset,
  };
}

export function toggleRun(selected: string[], id: string): string[] {
  if (selected.includes(id)) return selected.filter((item) => item !== id);
  return selected.length >= MAX_COMPARE_RUNS ? selected : [...selected, id];
}

export type CompareField = { key: string; read: (run: RunEntry) => string | number | null };

export const COMPARE_FIELDS: CompareField[] = [
  { key: "status", read: (run) => run.status },
  { key: "stage", read: (run) => run.stage },
  { key: "dataset", read: (run) => run.dataset },
  { key: "step", read: (run) => run.step },
  { key: "tokens", read: (run) => run.tokens_seen },
  { key: "duration", read: (run) => run.duration_seconds },
  { key: "trainLoss", read: (run) => run.train_loss },
  { key: "validationLoss", read: (run) => run.validation_loss },
  { key: "perplexity", read: (run) => run.validation_perplexity },
  { key: "speed", read: (run) => run.tokens_per_second },
];

export function datasetsDiffer(runs: RunEntry[]): boolean {
  const known = new Set(runs.map((run) => run.dataset).filter((value): value is string => value !== null));
  return known.size > 1 || (known.size === 1 && runs.some((run) => run.dataset === null));
}

export function nextOffset(pages: { items: unknown[]; total: number }[]): number | undefined {
  const loaded = pages.reduce((sum, page) => sum + page.items.length, 0);
  const total = pages.at(-1)?.total ?? 0;
  return loaded < total && (pages.at(-1)?.items.length ?? 0) > 0 ? loaded : undefined;
}
