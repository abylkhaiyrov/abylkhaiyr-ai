import type { EvaluationRow, MetricDefinition } from "@/lib/api";

export const SECTIONS = ["language", "conversation", "reasoning", "code", "multilingual", "efficiency"] as const;
export type SectionKey = (typeof SECTIONS)[number];

export const DEFAULT_SELECTION = 3;
export const MAX_COMPARE = 8;

export const rowId = (row: Pick<EvaluationRow, "checkpoint" | "sources">) => row.checkpoint ?? row.sources[0];

export const checkpointLabel = (value: string | null) => (value ? value.split("/").slice(-2).join("/") : "—");

export function metricsOf(catalog: MetricDefinition[], section: SectionKey) {
  return catalog.filter((item) => item.section === section);
}

export function toggle(selected: string[], id: string): string[] {
  if (selected.includes(id)) return selected.filter((item) => item !== id);
  return selected.length >= MAX_COMPARE ? selected : [...selected, id];
}

export function sectionCoverage(catalog: MetricDefinition[], rows: EvaluationRow[], section: SectionKey) {
  const defined = metricsOf(catalog, section);
  const measured = defined.filter((item) => rows.some((row) => row.results[item.key] !== undefined));
  return { measured: measured.length, total: defined.length };
}

export function formatMetricValue(definition: Pick<MetricDefinition, "unit">, value: number | null | undefined, locale: string) {
  if (value === null || value === undefined || !Number.isFinite(value)) return null;
  const digits = (count: number) => new Intl.NumberFormat(locale, { maximumFractionDigits: count });
  switch (definition.unit) {
    case "ratio":
      return `${digits(1).format(value * 100)}%`;
    case "gigabytes":
      return `${digits(2).format(value)} GB`;
    case "rate":
      return digits(1).format(value);
    default:
      return digits(3).format(value);
  }
}

export function formatDelta(definition: Pick<MetricDefinition, "unit">, delta: number | null, locale: string) {
  if (delta === null) return null;
  const sign = delta > 0 ? "+" : "";
  const shown = definition.unit === "ratio" ? `${new Intl.NumberFormat(locale, { maximumFractionDigits: 1 }).format(delta * 100)} pp` : new Intl.NumberFormat(locale, { maximumFractionDigits: 3 }).format(delta);
  return `${sign}${shown}`;
}

export function formatInterval(definition: Pick<MetricDefinition, "unit">, interval: [number, number] | null | undefined, locale: string) {
  if (!interval) return null;
  const [low, high] = interval;
  const low_text = formatMetricValue(definition, low, locale);
  const high_text = formatMetricValue(definition, high, locale);
  return low_text && high_text ? `${low_text} – ${high_text}` : null;
}
