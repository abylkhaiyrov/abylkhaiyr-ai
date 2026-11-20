import { useTranslation } from "react-i18next";
import { Panel, Stat } from "@/components/data-state";
import { StatusBadge } from "@/components/status-badge";
import { ProgressBar } from "@/components/ui/progress-bar";
import { StageProgress } from "@/components/pipeline/stage-progress";
import { useFormat } from "@/hooks/use-format";
import type { Metric, PipelineStage, RunDetail } from "@/lib/api";

function pick(rows: Metric[], metric: string) {
  const values = rows.map((row) => row[metric]).filter((value): value is number => typeof value === "number");
  return values.length === 0 ? null : { first: values[0], last: values[values.length - 1], best: Math.min(...values) };
}

export function RunOverview({ run, rows, stage }: { run: RunDetail; rows: Metric[]; stage: PipelineStage | null }) {
  const { t } = useTranslation();
  const format = useFormat();
  const train = pick(rows, "train/loss");
  const validation = pick(rows, "validation/loss");
  const perplexity = pick(rows, "validation/perplexity");
  const rows3 = [
    [t("chart.trainLoss"), train, 4],
    [t("chart.validationLoss"), validation, 4],
    [t("chart.perplexity"), perplexity, 2],
  ] as const;
  return (
    <div className="grid gap-4">
      <Panel title={t("training.process")}>
        <div className="grid gap-4">
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge value={run.status} />
            <span className="rounded bg-muted px-1.5 py-0.5 text-xs uppercase">{t(`modelStage.${run.stage}`)}</span>
          </div>
          {stage && run.status === "running" ? <StageProgress stage={stage} /> : <ProgressBar percent={run.percent} label={run.name} tone={run.status === "completed" ? "success" : "warning"} />}
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
            <Stat label={t("metric.step")} value={format.ratio(run.step, run.max_steps)} />
            <Stat label={t("metric.tokens_seen")} value={format.compact(run.tokens_seen)} />
            <Stat label={t("training.started")} value={format.dateTime(run.started_at)} />
            <Stat label={t("training.updated")} value={format.dateTime(run.updated_at)} />
          </dl>
        </div>
      </Panel>
      <Panel title={t("training.progressMetrics")}>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead><tr><th>{t("training.metric")}</th><th>{t("chart.start")}</th><th>{t("chart.current")}</th><th>{t("chart.bestLow")}</th></tr></thead>
            <tbody>
              {rows3.map(([label, values, digits]) => (
                <tr key={label}>
                  <td className="font-medium">{label}</td>
                  <td className="tabular-nums">{format.number(values?.first, digits)}</td>
                  <td className="tabular-nums">{format.number(values?.last, digits)}</td>
                  <td className="tabular-nums">{format.number(values?.best, digits)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
