import { describe, expect, it } from "vitest";
import type { Model, ModelRegistry } from "@/lib/api";
import { defaultModel } from "@/lib/models";

const model = (id: string, stage: string, extra: Partial<Model> = {}) => ({
  id, stage, compatible: true, available: true, quality_failed: false,
  created_at: "2026-11-20T10:00:00Z", ...extra,
} as Model);
const registry = (models: Model[]): ModelRegistry => ({
  models, roles: { loaded: "base", production: null, best_evaluated: null, latest_trained: "base" },
});

describe("chat default model", () => {
  it("prefers instruction weights over a loaded pretraining model", () => {
    const chat = model("chat", "sft");
    expect(defaultModel(registry([model("base", "pretrain"), chat]), undefined)).toBe(chat);
  });

  it("does not default to failed or unavailable instruction weights", () => {
    const base = model("base", "pretrain");
    expect(defaultModel(registry([base, model("failed", "sft", { quality_failed: true }), model("missing", "sft", { available: false })]), undefined)).toBe(base);
  });

  it("keeps the explicitly loaded instruction model", () => {
    const loaded = model("base", "sft");
    expect(defaultModel(registry([loaded, model("newer", "sft", { created_at: "2026-11-21T10:00:00Z" })]), undefined)).toBe(loaded);
  });
});
