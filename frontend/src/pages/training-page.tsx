import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";
import { api } from "@/lib/api";
import { EmptyState, ErrorState, LoadingState } from "@/components/data-state";
import { MetricChart } from "@/components/metric-chart";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/status-badge";
import { formatDuration, formatNumber } from "@/lib/utils";

export function TrainingPage() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const [configPath, setConfigPath] = useState("configs/tiny_v2.yaml");
  const processes = useQuery({ queryKey: ["processes"], queryFn: api.processes, refetchInterval: 5000 });
  const metrics = useQuery({ queryKey: ["metrics"], queryFn: api.metrics, refetchInterval: 5000 });
  const start = useMutation({ mutationFn: () => api.startTraining(configPath), onSuccess: () => client.invalidateQueries({ queryKey: ["processes"] }) });
  const pause = useMutation({ mutationFn: api.pauseTraining, onSuccess: () => client.invalidateQueries({ queryKey: ["processes"] }) });
  const stop = useMutation({ mutationFn: api.stopTraining, onSuccess: () => client.invalidateQueries({ queryKey: ["processes"] }) });
  function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); start.mutate(); }
  if (processes.isPending || metrics.isPending) return <LoadingState />;
  if (processes.isError) return <ErrorState error={processes.error} retry={() => processes.refetch()} />;
  const metricData = metrics.data ?? [];
  return <div className="grid gap-5"><div><h1 className="text-xl font-semibold">{t("training.title")}</h1><p className="text-sm text-muted-foreground">{t("training.subtitle")}</p></div><Card><CardHeader><CardTitle>{t("training.start")}</CardTitle></CardHeader><CardContent><form className="flex flex-wrap gap-2" onSubmit={submit}><label className="sr-only" htmlFor="training-config">{t("training.config")}</label><input id="training-config" className="h-9 min-w-64 rounded-md border bg-transparent px-3 text-sm" value={configPath} onChange={(event) => setConfigPath(event.target.value)} /><Button type="submit" disabled={start.isPending}>{t("training.launch")}</Button>{start.error ? <p role="alert" className="self-center text-sm text-destructive">{start.error.message}</p> : null}</form></CardContent></Card>{processes.data.length === 0 ? <EmptyState title={t("training.noProcesses")} description={t("training.subtitle")} /> : <div className="overflow-x-auto rounded-lg border"><table className="w-full text-left text-sm"><thead className="bg-muted text-xs text-muted-foreground"><tr><th className="p-3">{t("training.process")}</th><th className="p-3">PID</th><th className="p-3">{t("dashboard.stage")}</th><th className="p-3">{t("dashboard.elapsed")}</th><th className="p-3">{t("monitoring.cpu")}</th><th className="p-3">{t("common.details")}</th></tr></thead><tbody>{processes.data.map((process) => <tr className="border-t" key={process.id}><td className="p-3">{t(`pipeline.${process.kind}`, { defaultValue: process.kind })}</td><td className="p-3 font-mono">{process.pid ?? "—"}</td><td className="p-3"><StatusBadge value={process.status} /></td><td className="p-3">{formatDuration((Date.now() - new Date(process.started_at).getTime()) / 1000)}</td><td className="p-3">{formatNumber(process.cpu_percent)}%</td><td className="p-3"><div className="flex gap-2"><Button variant="outline" disabled={process.managed === false || process.status !== "running" || pause.isPending} onClick={() => pause.mutate(process.id)}>{t("training.pause")}</Button><Button variant="danger" disabled={process.managed === false || !['running', 'pausing'].includes(process.status) || stop.isPending} onClick={() => stop.mutate(process.id)}>{t("training.stop")}</Button></div></td></tr>)}</tbody></table></div>}{metricData.length > 0 ? <div className="grid gap-5 xl:grid-cols-2"><MetricChart title={t("training.loss")} data={metricData} metric="train/loss" /><MetricChart title={t("training.validationLoss")} data={metricData} metric="validation/loss" color="#f59e0b" /><MetricChart title={t("training.learningRate")} data={metricData} metric="train/learning_rate" color="#60a5fa" /><MetricChart title={t("training.gradNorm")} data={metricData} metric="train/grad_norm" color="#f472b6" /><MetricChart title={t("training.tokensPerSecond")} data={metricData} metric="perf/tokens_per_second" color="#34d399" /></div> : null}</div>;
}
