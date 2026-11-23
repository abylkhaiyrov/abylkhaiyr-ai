import { describe, expect, it } from "vitest";
import { checkpointLabel, formatDelta, formatMetricValue, MAX_COMPARE, rowId, sectionCoverage, toggle } from "@/lib/evaluation";
import type { EvaluationRow, MetricDefinition } from "@/lib/api";

const def = (key: string, unit: MetricDefinition["unit"], section: MetricDefinition["section"] = "language"): MetricDefinition => ({ key, section, direction: "higher", unit, language: null, tools: null });
const row = (results: Record<string, number>): EvaluationRow => ({ checkpoint: "a/b/c", model: null, stage: null, evaluated_at: null, sources: ["s"], results, gates: [], gate_status: "not_verified" });

describe("formatMetricValue", () => {
  it("never turns a missing metric into zero", () => {
    expect(formatMetricValue(def("x", "ratio"), undefined, "en")).toBeNull();
    expect(formatMetricValue(def("x", "ratio"), null, "en")).toBeNull();
    expect(formatMetricValue(def("x", "ratio"), Number.NaN, "en")).toBeNull();
  });
  it("shows a genuine zero as zero", () => {
    expect(formatMetricValue(def("x", "ratio"), 0, "en")).toBe("0%");
  });
  it("formats by unit", () => {
    expect(formatMetricValue(def("x", "ratio"), 0.4123, "en")).toBe("41.2%");
    expect(formatMetricValue(def("x", "number"), 19.6631, "en")).toBe("19.663");
    expect(formatMetricValue(def("x", "gigabytes"), 1.5, "en")).toBe("1.5 GB");
  });
});

describe("formatDelta", () => {
  it("signs changes and uses percentage points for ratios", () => {
    expect(formatDelta(def("x", "ratio"), 0.05, "en")).toBe("+5 pp");
    expect(formatDelta(def("x", "number"), -2.5, "en")).toBe("-2.5");
    expect(formatDelta(def("x", "number"), null, "en")).toBeNull();
  });
});

describe("selection", () => {
  it("toggles and caps the comparison size", () => {
    expect(toggle(["a"], "b")).toEqual(["a", "b"]);
    expect(toggle(["a", "b"], "a")).toEqual(["b"]);
    const full = Array.from({ length: MAX_COMPARE }, (_, index) => String(index));
    expect(toggle(full, "extra")).toEqual(full);
  });
  it("labels checkpoints and rows", () => {
    expect(checkpointLabel("checkpoints/model/step-1")).toBe("model/step-1");
    expect(checkpointLabel(null)).toBe("—");
    expect(rowId({ checkpoint: null, sources: ["file.json"] })).toBe("file.json");
  });
  it("counts measured metrics per section", () => {
    const catalog = [def("a", "ratio"), def("b", "ratio"), def("c", "ratio", "code")];
    expect(sectionCoverage(catalog, [row({ a: 1 })], "language")).toEqual({ measured: 1, total: 2 });
    expect(sectionCoverage(catalog, [row({})], "code")).toEqual({ measured: 0, total: 1 });
  });
});
