import { describe, expect, it } from "vitest";
import type { GenerationEvent } from "@/lib/events";
import { initialGeneration, reduceGeneration } from "@/lib/generation";

const base = { seq: 1, ts: 0, request_id: "r1", chat_id: "c1" };
const event = (type: GenerationEvent["type"], sequence_id: number, extra: Partial<GenerationEvent> = {}) =>
  ({ ...base, type, sequence_id, ...extra }) as GenerationEvent;

describe("reduceGeneration", () => {
  it("appends deltas in order and ignores duplicates", () => {
    let state = initialGeneration("r1", "c1", "send", null, "hi");
    state = reduceGeneration(state, event("generation.started", 1, { message_id: "m1" }))!;
    state = reduceGeneration(state, event("generation.delta", 2, { text: "He", first_token_ms: 120 }))!;
    state = reduceGeneration(state, event("generation.delta", 2, { text: "He" }))!;
    state = reduceGeneration(state, event("generation.delta", 3, { text: "llo", tokens_per_second: 9 }))!;
    expect(state.text).toBe("Hello");
    expect(state.firstTokenMs).toBe(120);
    expect(state.tokensPerSecond).toBe(9);
    expect(state.status).toBe("running");
  });

  it("ignores events from other requests", () => {
    const state = initialGeneration("r1", "c1", "send", null, "hi");
    expect(reduceGeneration(state, { ...event("generation.delta", 1, { text: "x" }), request_id: "other" })).toBe(state);
  });

  it("restores state from a snapshot after reconnect", () => {
    const restored = reduceGeneration(null, event("generation.snapshot", 7, { status: "running", text: "partial", message_id: "m1" }))!;
    expect(restored.text).toBe("partial");
    expect(restored.sequence).toBe(7);
  });

  it("records failures", () => {
    const state = reduceGeneration(initialGeneration("r1", "c1", "send", null, "hi"), event("generation.failed", 1, { code: "queue_full" }))!;
    expect(state.status).toBe("failed");
    expect(state.errorCode).toBe("queue_full");
  });
});
