import { useTranslation } from "react-i18next";
import { Stat } from "@/components/data-state";
import { StatusBadge } from "@/components/status-badge";
import { ProgressBar } from "@/components/ui/progress-bar";
import { useFormat } from "@/hooks/use-format";
import type { RunEntry } from "@/lib/api";

export function RunSummaryCard({ run }: { run: RunEntry }) {
  const { t } = useTranslation();
  const format = useFormat();
  return (
    <section className="grid gap-4 border-b hairline pb-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="label-caps">{run.status === "running" ? t("hero.now") : t("dashboard.selectedRun")}</p>
          <h2 className="mt-1.5 truncate text-2xl font-semibold tracking-tight" title={run.name}>{run.name}</h2>
          <p className="mt-1 text-[13px] text-muted-foreground">{t(`modelStage.${run.stage}`, { defaultValue: run.stage })}{run.version ? ` · ${run.version}` : ""}</p>
        </div>
        <StatusBadge value={run.status} />
      </div>
      {run.percent === null ? null : <ProgressBar percent={run.percent} label={t("hero.stageProgress")} />}
      <dl className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4">
        <Stat label={t("dashboard.steps")} value={run.step === null ? "—" : `${format.number(run.step, 0)}${run.max_steps ? ` / ${format.number(run.max_steps, 0)}` : ""}`} />
        <Stat label={t("metric.validation_loss")} value={format.number(run.validation_loss ?? undefined, 4)} />
        <Stat label={t("dashboard.updated")} value={format.dateTime(run.updated_at)} />
        <Stat label={t("dashboard.origin")} value={t(`dashboard.origins.${run.origin}`)} />
      </dl>
      {run.failure_reason ? <p role="alert" className="text-[13px] text-destructive">{t("dashboard.failure")}: {run.failure_reason}</p> : null}
    </section>
  );
}
