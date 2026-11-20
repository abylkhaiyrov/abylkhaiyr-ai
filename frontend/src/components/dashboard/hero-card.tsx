import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ArrowRight } from "lucide-react";
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
    const interrupted = pipeline?.stages.some((stage) => stage.status === "interrupted");
    return (
      <section className="flex flex-wrap items-center justify-between gap-4 border-b hairline pb-6">
        <div>
          <StatusBadge value={interrupted ? "interrupted" : "idle"} />
          <h2 className="mt-2 text-2xl font-semibold tracking-tight">{t("hero.idleTitle")}</h2>
          <p className="mt-1 max-w-xl text-[13px] text-muted-foreground">{t("hero.idleText")}</p>
        </div>
        <Link className="inline-flex h-8 items-center gap-1.5 rounded-md bg-primary px-3 text-[13px] font-medium text-primary-foreground hover:bg-primary/85" to="/training">{t("nav.training")}<ArrowRight size={14} /></Link>
      </section>
    );
  }
  const next = pipeline.stages.find((stage) => stage.key === pipeline.next_stage);
  const finishAt = current.eta_seconds === null ? null : Date.now() + current.eta_seconds * 1000;
  return (
    <section className="grid gap-6 border-b hairline pb-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="label-caps">{t("hero.now")}</p>
          <h2 className="mt-1.5 text-2xl font-semibold tracking-tight">{t(`action.${ACTION_KEYS[current.key]}`, { stage: t(`stage.${current.key}`) })}</h2>
          <p className="mt-1 text-[13px] text-muted-foreground">{t("hero.model")} · <span className="font-medium text-foreground">{current.output_dir ? current.output_dir.split("/").pop() : pipeline.model_name}</span></p>
        </div>
        <StatusBadge value="running" />
      </div>
      <div className="grid gap-8 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <div className="grid gap-4">
          <StageProgress stage={current} />
          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4">
            <Stat label={t("hero.stageTime")} value={format.duration(current.duration_seconds)} />
            <Stat label={t("hero.finishAt")} value={finishAt === null ? t("progress.etaUnknown") : format.clock(finishAt)} />
            <Stat label={t("hero.next")} value={next ? t(`stage.${next.key}`) : "—"} />
            <Stat label={t("hero.pid")} value={pipeline.pid ?? "—"} />
          </dl>
        </div>
        <div className="grid content-start gap-2 lg:border-l lg:pl-8 hairline">
          <p className="label-caps">{t("hero.overall")}</p>
          <p className="text-2xl font-semibold tabular-nums">{pipeline.overall.finished}<span className="text-muted-foreground"> / {pipeline.overall.total}</span></p>
          <ProgressBar percent={pipeline.overall.percent} label={t("hero.overall")} />
          <p className="text-xs text-muted-foreground tabular-nums">{t("hero.elapsed")}: {format.duration(pipeline.overall.elapsed_seconds)}</p>
        </div>
      </div>
      <StageBody stage={current} />
    </section>
  );
}
