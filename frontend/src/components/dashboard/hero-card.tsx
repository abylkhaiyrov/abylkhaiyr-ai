import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ArrowRight } from "lucide-react";
import { Card } from "@/components/ui/card";
import { ProgressBar } from "@/components/ui/progress-bar";
import { Stat } from "@/components/data-state";
import { StatusBadge } from "@/components/status-badge";
import { EvalDetail } from "@/components/pipeline/eval-detail";
import { ResultsGrid } from "@/components/pipeline/results-grid";
import { StageProgress } from "@/components/pipeline/stage-progress";
import { useFormat } from "@/hooks/use-format";
import type { Pipeline, PipelineStage } from "@/lib/api";
import { ACTION_KEYS, orderedResults } from "@/lib/stage";

function StageBody({ stage }: { stage: PipelineStage }) {
  if (stage.kind === "eval") return <EvalDetail stage={stage} />;
  const entries = orderedResults(stage);
  return entries.length > 0 ? <ResultsGrid entries={entries} columns="grid-cols-2 sm:grid-cols-4" /> : null;
}

export function HeroCard({ pipeline }: { pipeline: Pipeline | null }) {
  const { t } = useTranslation();
  const format = useFormat();
  const current = pipeline?.stages.find((stage) => stage.key === pipeline.current_stage) ?? null;
  if (pipeline === null || current === null || !pipeline.running) {
    return (
      <Card className="flex flex-wrap items-center justify-between gap-4 p-5">
        <div>
          <StatusBadge value={pipeline?.stages.some((stage) => stage.status === "interrupted") ? "interrupted" : "idle"} />
          <h2 className="mt-2 text-lg font-semibold">{t("hero.idleTitle")}</h2>
          <p className="mt-1 text-sm text-muted-foreground">{t("hero.idleText")}</p>
        </div>
        <Link className="inline-flex h-9 items-center gap-2 rounded-md border bg-card px-3 text-sm font-medium hover:bg-muted" to="/training">{t("nav.training")}<ArrowRight size={14} /></Link>
      </Card>
    );
  }
  const next = pipeline.stages.find((stage) => stage.key === pipeline.next_stage);
  const finishAt = current.eta_seconds === null ? null : Date.now() + current.eta_seconds * 1000;
  return (
    <Card className="grid gap-5 p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="label-caps">{t("hero.now")}</p>
          <h2 className="mt-1 text-xl font-semibold">{t(`action.${ACTION_KEYS[current.key]}`, { stage: t(`stage.${current.key}`) })}</h2>
          <p className="mt-0.5 text-sm text-muted-foreground">{t("hero.model")}: <span className="font-medium text-foreground">{current.output_dir ? current.output_dir.split("/").pop() : pipeline.model_name}</span></p>
        </div>
        <StatusBadge value="running" />
      </div>
      <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div className="grid gap-3">
          <p className="label-caps">{t("hero.stageProgress")}</p>
          <StageProgress stage={current} />
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
            <Stat label={t("hero.stageTime")} value={format.duration(current.duration_seconds)} />
            <Stat label={t("hero.finishAt")} value={finishAt === null ? t("progress.etaUnknown") : format.clock(finishAt)} />
            <Stat label={t("hero.next")} value={next ? t(`stage.${next.key}`) : "—"} />
            <Stat label={t("hero.pid")} value={pipeline.pid ?? "—"} />
          </dl>
        </div>
        <div className="grid content-start gap-3 rounded-lg bg-muted/50 p-3">
          <p className="label-caps">{t("hero.overall")}</p>
          <ProgressBar percent={pipeline.overall.percent} label={t("hero.overall")} />
          <p className="text-xs text-muted-foreground tabular-nums">
            {t("hero.stagesDone", { done: pipeline.overall.finished, total: pipeline.overall.total })} · {t("hero.elapsed")}: {format.duration(pipeline.overall.elapsed_seconds)}
          </p>
        </div>
      </div>
      <div className="border-t pt-4">
        <StageBody stage={current} />
      </div>
    </Card>
  );
}
