import { useState } from "react";
import { AlertTriangle, Check, ChevronRight, CircleDashed, Loader2, MinusCircle } from "lucide-react";
import { useTranslation } from "react-i18next";
import { StageDialog } from "@/components/pipeline/stage-dialog";
import { StageProgress } from "@/components/pipeline/stage-progress";
import { useFormat } from "@/hooks/use-format";
import { useMetricFormat } from "@/hooks/use-metric-format";
import type { Pipeline, PipelineStage, StageStatus } from "@/lib/api";
import { headlineResults } from "@/lib/stage";
import { cn } from "@/lib/utils";

const ICONS: Record<StageStatus, { icon: typeof Check; tone: string }> = {
  completed: { icon: Check, tone: "bg-success/15 text-success" },
  running: { icon: Loader2, tone: "bg-info/15 text-info" },
  pending: { icon: CircleDashed, tone: "bg-muted text-muted-foreground" },
  skipped: { icon: MinusCircle, tone: "bg-muted text-muted-foreground" },
  failed: { icon: AlertTriangle, tone: "bg-destructive/15 text-destructive" },
  interrupted: { icon: AlertTriangle, tone: "bg-warning/15 text-warning" },
};

function StageRow({ stage, onOpen }: { stage: PipelineStage; onOpen: () => void }) {
  const { t } = useTranslation();
  const format = useFormat();
  const metric = useMetricFormat();
  const { icon: Icon, tone } = ICONS[stage.status];
  const headline = headlineResults(stage, 3);
  return (
    <li>
      <button type="button" onClick={onOpen} className="group flex w-full items-start gap-3 rounded-lg px-2 py-2 text-left transition-colors hover:bg-muted/60">
        <span className={cn("mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full", tone)}>
          <Icon size={13} className={stage.status === "running" ? "animate-spin" : undefined} />
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-baseline justify-between gap-x-3">
            <span className={cn("font-medium", stage.status === "pending" && "text-muted-foreground")}>{t(`stage.${stage.key}`)}</span>
            <span className="text-xs text-muted-foreground tabular-nums">
              {stage.status === "pending" ? t("status.pending") : stage.duration_seconds !== null ? format.duration(stage.duration_seconds) : t(`status.${stage.status}`)}
            </span>
          </span>
          {stage.status === "running" ? <div className="mt-1.5"><StageProgress stage={stage} compact /></div> : null}
          {headline.length > 0 && stage.status !== "running" ? (
            <span className="mt-0.5 block truncate text-xs text-muted-foreground">
              {headline.map(([key, value]) => `${metric.label(key)} ${metric.value(key, value)}`).join(" · ")}
            </span>
          ) : null}
          {stage.note && stage.status === "skipped" ? <span className="mt-0.5 block truncate text-xs text-muted-foreground">{stage.note}</span> : null}
        </span>
        <ChevronRight size={15} className="mt-1 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100" />
      </button>
    </li>
  );
}

export function PipelineTimeline({ pipeline }: { pipeline: Pipeline }) {
  const [open, setOpen] = useState<PipelineStage["key"] | null>(null);
  const selected = pipeline.stages.find((stage) => stage.key === open) ?? null;
  return (
    <>
      <ol className="grid gap-0.5">
        {pipeline.stages.map((stage) => (
          <StageRow key={stage.key} stage={stage} onOpen={() => setOpen(stage.key)} />
        ))}
      </ol>
      <StageDialog stage={selected} onClose={() => setOpen(null)} />
    </>
  );
}
