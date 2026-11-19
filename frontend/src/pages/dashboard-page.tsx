import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { ErrorState, LoadingState } from "@/components/data-state";
import { StatusBadge } from "@/components/status-badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";
import { formatBytes, formatDuration, formatNumber } from "@/lib/utils";

export function DashboardPage() {
  const { t, i18n } = useTranslation();
  const query = useQuery({ queryKey: ["dashboard"], queryFn: api.dashboard, refetchInterval: 5000 });
  if (query.isPending) return <LoadingState />;
  if (query.isError) return <ErrorState error={query.error} retry={() => query.refetch()} />;
  const { training, inference_model: inferenceModel, pipeline, system, counts, events } = query.data;
  const progress = training.step !== null && training.max_steps !== null && training.max_steps > 0
    ? Math.min(100, training.step / training.max_steps * 100)
    : null;
  const elapsed = training.started_at ? (Date.now() - new Date(training.started_at).getTime()) / 1000 : null;
  const values = [
    [t("dashboard.step"), `${formatNumber(training.step, 0)} / ${formatNumber(training.max_steps, 0)}`],
    [t("dashboard.tokens"), formatNumber(training.tokens_seen, 0)],
    [t("dashboard.elapsed"), formatDuration(elapsed)],
    [t("dashboard.eta"), formatDuration(training.eta_seconds)],
    [t("dashboard.throughput"), training.throughput === null ? t("common.unavailable") : t("dashboard.tokensPerSecond", { value: formatNumber(training.throughput, 0) })],
  ];
  return (
    <div className="grid gap-6">
      <div className="page-header"><div><h1 className="page-title">{t("dashboard.title")}</h1><p className="page-subtitle">{t("dashboard.subtitle")}</p></div><StatusBadge value={query.data.system_status} /></div>
      <Card><CardContent className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <div className="min-w-0"><p className="text-xs font-medium text-primary">{t("dashboard.whatNow")}</p><h2 className="mt-2 text-xl font-semibold">{training.stage ? t(`pipeline.${training.stage}`, { defaultValue: t("common.unavailable") }) : t("dashboard.noActiveTraining")}</h2><p className="mt-1 break-words text-sm text-muted-foreground">{training.model_name ?? t("common.unavailable")}</p>
          <div className="mt-6"><div className="mb-2 flex justify-between gap-3 text-sm"><span>{t("dashboard.progress")}</span><span>{progress === null ? t("common.unavailable") : `${new Intl.NumberFormat(i18n.language, { maximumFractionDigits: 2 }).format(progress)}%`}</span></div><div className="h-2 overflow-hidden rounded-full bg-muted">{progress !== null && <div className="h-full rounded-full bg-primary" style={{ width: `${progress}%` }} />}</div></div>
          <dl className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-5">{values.map(([label, value]) => <div key={label}><dt className="text-xs text-muted-foreground">{label}</dt><dd className="mt-1 break-words text-sm font-medium tabular-nums">{value}</dd></div>)}</dl>
        </div>
        <div className="border-t pt-5 lg:border-l lg:border-t-0 lg:pl-6 lg:pt-0"><p className="text-xs text-muted-foreground">{t("dashboard.chatModel")}</p><p className="mt-2 break-words font-semibold">{inferenceModel ?? t("common.unavailable")}</p><p className="mt-5 text-xs text-muted-foreground">{t("dashboard.activeProcesses")}</p><p className="mt-1 text-xl font-semibold">{counts.active_processes}</p><p className="mt-4 break-all text-xs text-muted-foreground">{training.checkpoint ?? t("dashboard.noCheckpoint")}</p></div>
      </CardContent></Card>
      <Card><CardHeader><CardTitle>{t("dashboard.pipeline")}</CardTitle></CardHeader><CardContent><ol className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">{pipeline.map((stage) => <li className="min-w-0 rounded-lg border p-3" key={stage.id}><p className="break-words text-sm font-medium">{t(`pipeline.${stage.id}`)}</p><p className="mt-1 text-xs text-muted-foreground">{t(`common.${stage.status}`, { defaultValue: t("common.unavailable") })}</p></li>)}</ol></CardContent></Card>
      <div className="grid gap-6 lg:grid-cols-2"><Card><CardHeader><CardTitle>{t("dashboard.system")}</CardTitle></CardHeader><CardContent><dl className="grid grid-cols-2 gap-4 text-sm"><div><dt className="text-muted-foreground">CPU</dt><dd>{formatNumber(system.cpu.percent)}%</dd></div><div><dt className="text-muted-foreground">RAM</dt><dd>{formatBytes(system.memory.used_bytes)} / {formatBytes(system.memory.total_bytes)}</dd></div><div><dt className="text-muted-foreground">{t("monitoring.swap")}</dt><dd>{formatBytes(system.memory.swap_used_bytes)}</dd></div><div><dt className="text-muted-foreground">{t("monitoring.device")}</dt><dd>{system.gpu.device}</dd></div></dl></CardContent></Card><Card><CardHeader><CardTitle>{t("dashboard.events")}</CardTitle></CardHeader><CardContent>{events.length === 0 ? <p className="text-sm text-muted-foreground">{t("dashboard.noEvents")}</p> : <ul className="grid gap-3">{events.map((event, index) => <li key={`${event.created_at}-${index}`} className="break-words text-sm">{event.message}<p className="text-xs text-muted-foreground">{new Date(event.created_at).toLocaleString(i18n.language)}</p></li>)}</ul>}</CardContent></Card></div>
    </div>
  );
}
