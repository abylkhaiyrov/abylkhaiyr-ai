import { describe, expect, it } from "vitest";
import { doneLabel, stageLabel, type Translate } from "@/lib/trace";
import type { ReasoningStageEvent } from "@/lib/api";

const t: Translate = (key, options) => {
  const values = options ?? {};
  return `${key}${Object.entries(values).filter(([name]) => name !== "defaultValue").map(([name, value]) => `|${name}=${String(value)}`).join("")}`;
};
const event = (stage: string, phase: string | undefined, data: Record<string, unknown> = {}): ReasoningStageEvent => ({ stage, status: "completed", phase, data });

describe("generation trace", () => {
  it("lists each executed phase once and never claims a semantic check", () => {
    const events = [event("task_classification", "preparation"), event("context_preparation", "preparation"), event("answer_formation", "generation")];
    expect(doneLabel(events, t)).toBe("trace.done|summary=trace.phases.preparation · trace.phases.generation");
  });
  it("names the kind of check that actually ran", () => {
    expect(stageLabel(event("verification", "format_check", { check: "format" }), t)).toBe("trace.stages.verification_format");
    expect(stageLabel(event("verification", "factual_check", { check: "factual" }), t)).toBe("trace.stages.verification_factual");
    expect(stageLabel(event("answer_formation", "generation"), t)).toBe("trace.stages.answer_formation");
  });
  it("falls back to a step count for traces stored without phases", () => {
    expect(doneLabel([event("answer_formation", undefined)], t)).toBe("trace.steps|count=1");
  });
});
