import { describe, expect, it } from "vitest";
import type { RunEntry, RuntimeState } from "@/lib/api";
import { computeChanged, groupProcesses, pickRun, processGroup } from "@/lib/runs";

const run = (id: string, status: RunEntry["status"]): RunEntry => ({
  id, kind: "training", name: id, model_name: id, version: null, stage: "sft", status, origin: "metrics",
  created_at: null, updated_at: null, step: null, max_steps: null, percent: null, current_stage: null, output_dir: null,
  state_dir: null, config_path: null, checkpoint: null, failure_reason: null, validation_loss: null, best_validation_loss: null,
  validation_perplexity: null, train_loss: null, tokens_seen: null, tokens_per_second: null, dataset: null, duration_seconds: null,
});

describe("pickRun", () => {
  const latest = run("run:new", "completed");
  const old = run("pipeline:old", "completed");
  it("prefers the active run over the latest", () => {
    const active = run("run:live", "running");
    expect(pickRun({ active, latest, completed_count: 1, total: 2 }, [old], "")?.id).toBe("run:live");
  });
  it("falls back to the latest finished run, never to a stale pipeline", () => {
    expect(pickRun({ active: null, latest, completed_count: 1, total: 2 }, [old], "")?.id).toBe("run:new");
  });
  it("uses an explicitly selected historical run", () => {
    expect(pickRun({ active: null, latest, completed_count: 1, total: 2 }, [old, latest], "pipeline:old")?.id).toBe("pipeline:old");
  });
  it("ignores a selection that no longer exists", () => {
    expect(pickRun({ active: null, latest, completed_count: 1, total: 2 }, [old], "gone")?.id).toBe("run:new");
  });
  it("returns null for an empty catalog", () => {
    expect(pickRun({ active: null, latest: null, completed_count: 0, total: 0 }, [], "")).toBeNull();
  });
});

describe("process groups", () => {
  it.each([["running", "active"], ["pausing", "active"], ["pending", "queued"], ["failed", "failed"], ["interrupted", "failed"], ["completed", "completed"], ["stopped", "completed"]] as const)("%s -> %s", (status, group) => {
    expect(processGroup({ status })).toBe(group);
  });
  it("splits a list", () => {
    const base = { kind: "sft", command: "", config_path: null, output_dir: null, started_at: "", completed_at: null, exit_code: null, error_reason: null, cpu_percent: null, memory_bytes: null, pid: null };
    const groups = groupProcesses([{ ...base, id: "a", status: "running" }, { ...base, id: "b", status: "failed" }, { ...base, id: "c", status: "completed" }]);
    expect([groups.active.length, groups.failed.length, groups.completed.length, groups.queued.length]).toEqual([1, 1, 1, 0]);
  });
});

describe("computeChanged", () => {
  const state = (busy: boolean, pid: number | null, active: string | null) => ({ compute: { busy, holder: pid === null ? null : { kind: "process", pid, stage: "sft" } }, runs: { active: active ? ({ id: active } as RunEntry) : null } }) as RuntimeState;
  it("detects a process starting, stopping or switching run", () => {
    expect(computeChanged(state(false, null, null), state(true, 5, "run:a"))).toBe(true);
    expect(computeChanged(state(true, 5, "run:a"), state(false, null, null))).toBe(true);
    expect(computeChanged(state(true, 5, "run:a"), state(true, 5, "run:b"))).toBe(true);
  });
  it("ignores unrelated runtime updates", () => {
    expect(computeChanged(state(true, 5, "run:a"), state(true, 5, "run:a"))).toBe(false);
  });
});
