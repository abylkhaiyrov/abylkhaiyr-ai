import { useTranslation } from "react-i18next";
import { ProgressBar } from "@/components/ui/progress-bar";
import { Stat } from "@/components/data-state";
import { useFormat } from "@/hooks/use-format";
import { useMetricFormat } from "@/hooks/use-metric-format";
import type { PipelineStage } from "@/lib/api";
import { INTERIM_METRICS } from "@/lib/stage";

export function EvalDetail({ stage }: { stage: PipelineStage }) {
  const { t } = useTranslation();
  const format = useFormat();
  const metric = useMetricFormat();
  const detail = stage.detail;
  if (!detail.checkpoint) {
    return stage.status === "running" ? <p className="text-sm text-muted-foreground">{t("progress.evalNoTelemetry")}</p> : null;
  }
  const itemPercent = detail.items_total ? ((detail.items_done ?? 0) / detail.items_total) * 100 : null;
  const interim = detail.interim ?? {};
  const rows = Object.entries(interim);
  return (
    <div className="grid gap-4">
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
        <Stat label={t("eval.checkpoint")} value={detail.checkpoint} hint={t("eval.checkpointOf", { index: detail.checkpoint_index, total: detail.checkpoint_total })} />
        <Stat label={t("eval.benchmark")} value={t(`benchmark.${detail.benchmark}`, { defaultValue: detail.benchmark ?? "—" })} hint={t("eval.benchmarkOf", { index: detail.benchmark_index, total: detail.benchmark_total })} />
        <Stat label={t("eval.tasksChecked")} value={format.number(detail.items_done, 0)} hint={t("eval.tasksRemaining", { count: detail.items_remaining ?? 0 })} />
        <Stat label={t("eval.speed")} value={`${format.number(detail.tasks_per_second, 2)} ${t("unit.tasksPerSecond")}`} />
      </dl>
      <div className="grid gap-1.5">
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{t("eval.batchProgress")}</span>
          <span className="tabular-nums">{format.ratio(detail.items_done, detail.items_total)}</span>
        </div>
        <ProgressBar percent={itemPercent} indeterminate={detail.items_total === null} label={t("eval.batchProgress")} />
      </div>
      {rows.length > 0 ? (
        <div className="overflow-x-auto">
          <p className="label-caps mb-1.5">{t("eval.interim")}</p>
          <table className="data-table">
            <thead>
              <tr>
                <th>{t("eval.checkpoint")}</th>
                {INTERIM_METRICS.map((key) => <th key={key}>{metric.label(key)}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.map(([name, values]) => (
                <tr key={name}>
                  <td className="font-medium">{name}</td>
                  {INTERIM_METRICS.map((key) => <td key={key} className="tabular-nums">{metric.value(key, values[key])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}
