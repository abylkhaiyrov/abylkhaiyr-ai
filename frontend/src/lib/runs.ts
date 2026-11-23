import type { Process, RunEntry, RunsSummary, RuntimeState } from "@/lib/api";

export function defaultRun(summary: RunsSummary): RunEntry | null {
  return summary.active ?? summary.latest;
}

export function pickRun(summary: RunsSummary, catalog: RunEntry[] | undefined, selectedId: string): RunEntry | null {
  const explicit = selectedId ? catalog?.find((item) => item.id === selectedId) : undefined;
  return explicit ?? defaultRun(summary);
}

export type ProcessGroup = "active" | "queued" | "completed" | "failed";

const ACTIVE = new Set(["running", "pausing", "paused"]);
const FAILED = new Set(["failed", "interrupted"]);

export function processGroup(item: Pick<Process, "status">): ProcessGroup {
  if (ACTIVE.has(item.status)) return "active";
  if (item.status === "pending") return "queued";
  if (FAILED.has(item.status)) return "failed";
  return "completed";
}

export function groupProcesses(items: Process[]): Record<ProcessGroup, Process[]> {
  const groups: Record<ProcessGroup, Process[]> = { active: [], queued: [], completed: [], failed: [] };
  for (const item of items) groups[processGroup(item)].push(item);
  return groups;
}

export function computeChanged(previous: RuntimeState, next: RuntimeState): boolean {
  const key = (state: RuntimeState) => `${state.compute.busy}|${state.compute.holder?.pid ?? ""}|${state.compute.holder?.stage ?? ""}|${state.runs.active?.id ?? ""}`;
  return key(previous) !== key(next);
}
