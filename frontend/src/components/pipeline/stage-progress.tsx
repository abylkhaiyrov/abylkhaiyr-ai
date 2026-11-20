import { useTranslation } from "react-i18next";
import { ProgressBar } from "@/components/ui/progress-bar";
import { useFormat } from "@/hooks/use-format";
import type { PipelineStage } from "@/lib/api";

export function StageProgress({ stage, compact = false }: { stage: PipelineStage; compact?: boolean }) {
  const { t } = useTranslation();
  const format = useFormat();
  const running = stage.status === "running";
  const unknown = stage.indeterminate || stage.percent === null;
  const tone = stage.status === "failed" ? "danger" : stage.status === "interrupted" ? "warning" : stage.status === "completed" ? "success" : "primary";
  return (
    <div className="grid gap-1.5">
      <ProgressBar percent={stage.percent} indeterminate={running && unknown} label={t(`stage.${stage.key}`)} tone={tone} />
      {compact ? null : (
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-0.5 text-xs text-muted-foreground">
          <span className="tabular-nums">
            {unknown
              ? t("progress.unknownTotal")
              : `${format.ratio(stage.done, stage.total)}${stage.unit ? ` ${t(`unit.${stage.unit}`, { defaultValue: stage.unit })}` : ""} · ${format.percent(stage.percent)}`}
          </span>
          {running ? (
            <span className="tabular-nums">
              {t("progress.eta")}: {stage.eta_seconds === null ? t("progress.etaUnknown") : format.duration(stage.eta_seconds)}
            </span>
          ) : null}
        </div>
      )}
    </div>
  );
}
