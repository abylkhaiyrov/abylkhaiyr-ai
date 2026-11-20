import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { SERIES_COLORS, seriesFromRows, type Series } from "@/components/charts/chart-utils";
import { TimeSeriesChart } from "@/components/charts/time-series-chart";
import { useFormat } from "@/hooks/use-format";
import type { Metric } from "@/lib/api";
import { RUN_CHARTS, type ChartDefinition } from "@/lib/run-charts";

export type RunSeries = { name: string; rows: Metric[] };

export function RunCharts({ runs, stage, markers }: { runs: RunSeries[]; stage: string; markers?: number[] }) {
  const { t, i18n } = useTranslation();
  const format = useFormat();
  const formatter = useCallback(
    (kind: ChartDefinition["kind"]) => {
      if (kind === "loss") return (value: number) => format.number(value, 3);
      if (kind === "scientific") return (value: number) => new Intl.NumberFormat(i18n.language, { notation: "scientific", maximumFractionDigits: 1 }).format(value);
      if (kind === "percent") return (value: number) => format.percent(value * 100, 0);
      if (kind === "rate") return (value: number) => format.number(value, 0);
      return (value: number) => format.number(value, Math.abs(value) >= 100 ? 0 : 2);
    },
    [format, i18n.language],
  );
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      {RUN_CHARTS.filter((chart) => !chart.stages || chart.stages.includes(stage)).map((chart) => {
        const series: Series[] = runs.map((run, index) => ({
          key: run.name,
          label: run.name,
          color: SERIES_COLORS[index % SERIES_COLORS.length],
          points: seriesFromRows(run.rows, chart.metric),
        }));
        if (series.every((item) => item.points.length === 0)) return null;
        return (
          <TimeSeriesChart
            key={chart.id}
            title={t(chart.titleKey)}
            unit={chart.unitKey ? t(chart.unitKey) : undefined}
            series={series}
            goal={chart.goal}
            allowLog={chart.allowLog}
            formatY={formatter(chart.kind)}
            xLabel={t("chart.step")}
            markers={runs.length === 1 ? (markers ?? []).map((x) => ({ x, label: String(x) })) : []}
          />
        );
      })}
    </div>
  );
}
