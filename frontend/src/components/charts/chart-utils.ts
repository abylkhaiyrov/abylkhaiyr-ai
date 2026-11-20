export type Point = { x: number; y: number };
export type Series = { key: string; label: string; color: string; points: Point[] };
export type Smoothing = "off" | "light" | "strong";

export const SERIES_COLORS = [1, 2, 3, 4, 5].map((index) => `hsl(var(--chart-${index}))`);
export const MAX_POINTS = 400;
const ALPHA: Record<Smoothing, number> = { off: 1, light: 0.35, strong: 0.12 };

export function downsample(points: Point[], limit = MAX_POINTS): Point[] {
  if (points.length <= limit) return points;
  const sampled: Point[] = [points[0]];
  const bucket = (points.length - 2) / (limit - 2);
  let anchor = 0;
  for (let index = 0; index < limit - 2; index += 1) {
    const start = Math.floor(index * bucket) + 1;
    const end = Math.min(Math.floor((index + 1) * bucket) + 1, points.length - 1);
    const nextStart = end;
    const nextEnd = Math.min(Math.floor((index + 2) * bucket) + 1, points.length);
    let averageX = 0;
    let averageY = 0;
    for (let cursor = nextStart; cursor < nextEnd; cursor += 1) {
      averageX += points[cursor].x;
      averageY += points[cursor].y;
    }
    const span = Math.max(nextEnd - nextStart, 1);
    averageX /= span;
    averageY /= span;
    let best = start;
    let bestArea = -1;
    for (let cursor = start; cursor < end; cursor += 1) {
      const area = Math.abs(
        (points[anchor].x - averageX) * (points[cursor].y - points[anchor].y) -
          (points[anchor].x - points[cursor].x) * (averageY - points[anchor].y),
      );
      if (area > bestArea) {
        bestArea = area;
        best = cursor;
      }
    }
    sampled.push(points[best]);
    anchor = best;
  }
  sampled.push(points[points.length - 1]);
  return sampled;
}

export function smooth(points: Point[], mode: Smoothing): Point[] {
  const alpha = ALPHA[mode];
  if (alpha === 1 || points.length === 0) return points;
  let state = points[0].y;
  return points.map((point) => {
    state = alpha * point.y + (1 - alpha) * state;
    return { x: point.x, y: state };
  });
}

export function pivot(series: Series[]): Record<string, number | null>[] {
  const rows = new Map<number, Record<string, number | null>>();
  for (const item of series) {
    for (const point of item.points) {
      const row = rows.get(point.x) ?? { x: point.x };
      row[item.key] = point.y;
      rows.set(point.x, row);
    }
  }
  return [...rows.values()].sort((a, b) => (a.x as number) - (b.x as number));
}

export function seriesFromRows(rows: Record<string, number>[], metric: string): Point[] {
  const points: Point[] = [];
  for (const row of rows) {
    const value = row[metric];
    if (typeof value === "number" && Number.isFinite(value)) points.push({ x: row.step, y: value });
  }
  return points;
}

export function summarize(points: Point[], goal: "min" | "max") {
  if (points.length === 0) return null;
  const values = points.map((point) => point.y);
  return {
    start: points[0].y,
    current: points[points.length - 1].y,
    best: goal === "min" ? Math.min(...values) : Math.max(...values),
  };
}

export function niceDomain(points: Point[], logScale: boolean): [number | string, number | string] {
  const values = points.map((point) => point.y).filter((value) => (logScale ? value > 0 : true));
  if (values.length === 0) return ["auto", "auto"];
  const low = Math.min(...values);
  const high = Math.max(...values);
  if (logScale) return [low * 0.9, high * 1.1];
  const pad = (high - low || Math.abs(high) || 1) * 0.08;
  return [low - pad, high + pad];
}

export function clamp(value: number, low: number, high: number) {
  return Math.min(Math.max(value, low), high);
}
