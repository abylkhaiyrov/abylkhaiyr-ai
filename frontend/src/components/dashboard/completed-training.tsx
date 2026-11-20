import { useTranslation } from "react-i18next";
import { Panel, EmptyState } from "@/components/data-state";
import { ResultsGrid } from "@/components/pipeline/results-grid";
import { useFormat } from "@/hooks/use-format";
import type { Pipeline } from "@/lib/api";
import { headlineResults, isTrainStage } from "@/lib/stage";

export function CompletedTraining({ pipeline }: { pipeline: Pipeline | null }) {
  const { t } = useTranslation();
  const format = useFormat();
  const finished = pipeline?.stages.filter((stage) => (isTrainStage(stage.key) || stage.key === "eval_pretrain" || stage.key === "eval_sft") && stage.status === "completed") ?? [];
  return (
    <Panel title={t("completed.title")}>
      {finished.length === 0 ? (
        <EmptyState title={t("completed.empty")} />
      ) : (
        <div className="grid gap-4">
          {finished.map((stage) => (
            <div key={stage.key} className="grid gap-2">
              <div className="flex items-baseline justify-between gap-2">
                <h3 className="font-medium">{t(`stage.${stage.key}`)}</h3>
                <span className="text-xs text-muted-foreground">{format.dateTime(stage.finished_at)} · {format.duration(stage.duration_seconds)}</span>
              </div>
              <ResultsGrid entries={headlineResults(stage, 6)} columns="grid-cols-2 sm:grid-cols-4" />
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
