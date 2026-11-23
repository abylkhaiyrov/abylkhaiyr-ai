import { describe, expect, it } from "vitest";
import type { RunEntry } from "@/lib/api";
import { datasetsDiffer, EMPTY_FILTERS, MAX_COMPARE_RUNS, nextOffset, toggleRun, toQuery } from "@/lib/history";

const run = (dataset: string | null): RunEntry => ({ dataset } as RunEntry);

describe("toQuery", () => {
  it("drops empty filters and trims text", () => {
    expect(toQuery({ ...EMPTY_FILTERS, q: "  v4 ", status: "failed" }, 40)).toMatchObject({ q: "v4", status: "failed", stage: undefined, version: undefined, offset: 40, sort: "newest" });
  });
});

describe("toggleRun", () => {
  it("caps the comparison", () => {
    const full = Array.from({ length: MAX_COMPARE_RUNS }, (_, index) => String(index));
    expect(toggleRun(full, "x")).toEqual(full);
    expect(toggleRun(["a"], "a")).toEqual([]);
  });
});

describe("datasetsDiffer", () => {
  it("flags different or partially recorded datasets", () => {
    expect(datasetsDiffer([run("a"), run("b")])).toBe(true);
    expect(datasetsDiffer([run("a"), run(null)])).toBe(true);
    expect(datasetsDiffer([run("a"), run("a")])).toBe(false);
    expect(datasetsDiffer([run(null), run(null)])).toBe(false);
  });
});

describe("nextOffset", () => {
  it("continues until everything is loaded and stops on empty pages", () => {
    expect(nextOffset([{ items: [1, 2], total: 5 }])).toBe(2);
    expect(nextOffset([{ items: [1, 2], total: 5 }, { items: [3, 4, 5], total: 5 }])).toBeUndefined();
    expect(nextOffset([{ items: [], total: 5 }])).toBeUndefined();
  });
});
