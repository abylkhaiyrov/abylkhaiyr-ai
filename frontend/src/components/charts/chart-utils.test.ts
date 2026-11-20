import { describe, expect, it } from "vitest";
import { downsample, niceDomain, pivot, smooth, summarize } from "@/components/charts/chart-utils";

const line = (count: number) => Array.from({ length: count }, (_, index) => ({ x: index, y: Math.sin(index / 10) }));

describe("chart utils", () => {
  it("limits points while keeping the first and last", () => {
    const points = line(5000);
    const sampled = downsample(points, 200);
    expect(sampled).toHaveLength(200);
    expect(sampled[0]).toEqual(points[0]);
    expect(sampled[199]).toEqual(points[4999]);
  });

  it("does not mix series when pivoting", () => {
    const rows = pivot([
      { key: "a", label: "a", color: "", points: [{ x: 1, y: 1 }, { x: 3, y: 3 }] },
      { key: "b", label: "b", color: "", points: [{ x: 2, y: 20 }] },
    ]);
    expect(rows).toEqual([{ x: 1, a: 1 }, { x: 2, b: 20 }, { x: 3, a: 3 }]);
  });

  it("summarizes start, current and best", () => {
    expect(summarize([{ x: 0, y: 5 }, { x: 1, y: 2 }, { x: 2, y: 3 }], "min")).toEqual({ start: 5, current: 3, best: 2 });
    expect(summarize([], "min")).toBeNull();
  });

  it("keeps smoothing a convex combination", () => {
    const smoothed = smooth([{ x: 0, y: 0 }, { x: 1, y: 10 }], "light");
    expect(smoothed[1].y).toBeGreaterThan(0);
    expect(smoothed[1].y).toBeLessThan(10);
  });

  it("pads domains and respects log scale", () => {
    const [low, high] = niceDomain([{ x: 0, y: 1 }, { x: 1, y: 3 }], false);
    expect(low as number).toBeLessThan(1);
    expect(high as number).toBeGreaterThan(3);
    expect(niceDomain([{ x: 0, y: -1 }], true)).toEqual(["auto", "auto"]);
  });
});
