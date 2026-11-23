import { useQuery } from "@tanstack/react-query";
import { X } from "lucide-react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ErrorState, LoadingState, Panel, Stat } from "@/components/data-state";
import { PipelineTimeline } from "@/components/pipeline/pipeline-timeline";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { useFormat } from "@/hooks/use-format";
import { api } from "@/lib/api";

export function RunDetailPanel({ id, onClose }: { id: string; onClose: () => void }) {
  const { t } = useTranslation();
  const format = useFormat();
  const query = useQuery({ queryKey: ["catalog-run", id], queryFn: () => api.catalogRun(id), staleTime: 30_000 });
  if (query.isPending) return <LoadingState />;
  if (query.isError) return <ErrorState error={query.error} retry={() => void query.refetch()} />;
  const { run, pipeline, detail } = query.data;
  return (
    <aside aria-label={run.name} className="grid content-start gap-6">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h2 className="truncate text-lg font-semibold" title={run.name}>{run.name}</h2>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 text-xs text-muted-foreground"><StatusBadge value={run.status} /><span>{format.dateTime(run.created_at)}</span></div>
        </div>
        <Button variant="ghost" size="icon" aria-label={t("common.close")} onClick={onClose}><X size={15} /></Button>
      </div>
      {run.failure_reason ? <p role="alert" className="text-[13px] text-destructive">{t("dashboard.failure")}: {run.failure_reason}</p> : null}
      <dl className="grid grid-cols-2 gap-x-6 gap-y-3">
        <Stat label={t("history.fields.dataset")} value={run.dataset ?? "—"} />
        <Stat label={t("history.fields.duration")} value={format.duration(run.duration_seconds)} />
        <Stat label={t("history.fields.step")} value={run.step === null ? "—" : `${format.number(run.step, 0)}${run.max_steps ? ` / ${format.number(run.max_steps, 0)}` : ""}`} />
        <Stat label={t("history.fields.tokens")} value={format.compact(run.tokens_seen)} />
        <Stat label={t("history.fields.trainLoss")} value={format.number(run.train_loss, 3)} />
        <Stat label={t("history.fields.validationLoss")} value={format.number(run.validation_loss, 3)} />
        <Stat label={t("history.fields.perplexity")} value={format.number(run.validation_perplexity, 2)} />
        <Stat label={t("history.fields.speed")} value={run.tokens_per_second === null ? "—" : `${format.number(run.tokens_per_second, 0)} ${t("units.tokensPerSecond")}`} />
      </dl>
      {pipeline ? <Panel title={t("dashboard.pipeline")}><PipelineTimeline pipeline={pipeline} /></Panel> : null}
      {detail && detail.checkpoints.length > 0 ? (
        <Panel title={t("history.checkpoints")}>
          <ul className="grid max-h-40 gap-1 overflow-auto text-xs">{detail.checkpoints.map((item) => <li key={item.id} className="flex justify-between gap-3"><span className="truncate font-mono" title={item.id}>{item.label}</span><span className="tabular-nums text-muted-foreground">{item.step === null ? "—" : format.number(item.step, 0)}</span></li>)}</ul>
        </Panel>
      ) : null}
      {detail && detail.evaluations.length > 0 ? (
        <Panel title={t("history.evaluations")} actions={<Link className="text-xs text-accent hover:underline" to="/evaluation">{t("nav.evaluation")}</Link>}>
          <ul className="grid gap-1 text-xs">{detail.evaluations.map((item) => <li key={`${item.name}-${item.checkpoint}`} className="flex justify-between gap-3"><span className="truncate" title={item.checkpoint}>{item.checkpoint}</span><span className="tabular-nums text-muted-foreground">{format.number(item.results.test_perplexity, 2)} {t("metric.test_perplexity")}</span></li>)}</ul>
        </Panel>
      ) : null}
      {detail?.config ? (
        <Panel title={t("history.config")} actions={<span className="truncate font-mono text-xs text-muted-foreground">{detail.config.path}</span>}>
          <details><summary className="cursor-pointer text-xs text-muted-foreground">{t("history.showConfig")}</summary><pre className="mt-2 max-h-64 overflow-auto rounded-md bg-muted p-3 font-mono text-xs">{detail.config.text}</pre></details>
        </Panel>
      ) : <p className="text-xs text-muted-foreground">{t("history.noConfig")}</p>}
    </aside>
  );
}
