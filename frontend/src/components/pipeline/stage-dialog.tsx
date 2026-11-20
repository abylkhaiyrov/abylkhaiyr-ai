import { useTranslation } from "react-i18next";
import { Dialog } from "@/components/ui/dialog";
import { Stat } from "@/components/data-state";
import { StatusBadge } from "@/components/status-badge";
import { EvalDetail } from "@/components/pipeline/eval-detail";
import { ResultsGrid } from "@/components/pipeline/results-grid";
import { StageProgress } from "@/components/pipeline/stage-progress";
import { useFormat } from "@/hooks/use-format";
import type { PipelineStage } from "@/lib/api";
import { orderedResults } from "@/lib/stage";

export function StageDialog({ stage, onClose }: { stage: PipelineStage | null; onClose: () => void }) {
  const { t } = useTranslation();
  const format = useFormat();
  if (stage === null) return null;
  const results = orderedResults(stage);
  const hasProgress = stage.status === "running" || stage.percent !== null;
  return (
    <Dialog open title={t(`stage.${stage.key}`)} description={t(`stageHelp.${stage.key}`)} onClose={onClose} className="max-w-2xl">
      <div className="grid gap-5">
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge value={stage.status} />
          {stage.phase ? <span className="text-xs text-muted-foreground">{t(`phase.${stage.phase}`, { defaultValue: stage.phase })}</span> : null}
        </div>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
          <Stat label={t("stageDialog.started")} value={format.dateTime(stage.started_at)} />
          <Stat label={t("stageDialog.finished")} value={format.dateTime(stage.finished_at)} />
          <Stat label={t("stageDialog.duration")} value={format.duration(stage.duration_seconds)} />
        </dl>
        {hasProgress ? <StageProgress stage={stage} /> : null}
        {stage.kind === "eval" ? <EvalDetail stage={stage} /> : null}
        {stage.kind === "train" && stage.detail.validation_total && stage.phase === "validation" ? (
          <p className="text-sm text-muted-foreground">{t("progress.validationBatches", { done: stage.detail.validation_done, total: stage.detail.validation_total })}</p>
        ) : null}
        {results.length > 0 ? (
          <div>
            <p className="label-caps mb-2">{t("stageDialog.results")}</p>
            <ResultsGrid entries={results} />
          </div>
        ) : null}
        {stage.note ? <p className="rounded-lg bg-muted p-3 text-sm">{stage.note_code && stage.note_values ? t(`stageNote.${stage.note_code}`, stage.note_values) : stage.note}</p> : null}
        {stage.error ? <p role="alert" className="break-words rounded-lg bg-destructive/10 p-3 text-sm text-destructive">{stage.error}</p> : null}
      </div>
    </Dialog>
  );
}
