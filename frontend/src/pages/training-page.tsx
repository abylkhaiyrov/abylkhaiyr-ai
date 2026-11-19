import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { api } from "@/lib/api";
import { ErrorState, LoadingState } from "@/components/data-state";
import { MetricChart } from "@/components/metric-chart";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/status-badge";
import { formatDuration, formatNumber } from "@/lib/utils";

export function TrainingPage() {
  const client = useQueryClient();
  const [configPath, setConfigPath] = useState("configs/debug.yaml");
  const processes = useQuery({ queryKey: ["processes"], queryFn: api.processes, refetchInterval: 5000 });
  const metrics = useQuery({ queryKey: ["metrics"], queryFn: api.metrics, refetchInterval: 5000 });
  const start = useMutation({ mutationFn: () => api.startTraining(configPath), onSuccess: () => client.invalidateQueries({ queryKey: ["processes"] }) });
  const pause = useMutation({ mutationFn: api.pauseTraining, onSuccess: () => client.invalidateQueries({ queryKey: ["processes"] }) });
  const stop = useMutation({ mutationFn: api.stopTraining, onSuccess: () => client.invalidateQueries({ queryKey: ["processes"] }) });
  function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); start.mutate(); }
  if (processes.isPending || metrics.isPending) return <LoadingState />;
  if (processes.isError) return <ErrorState error={processes.error} retry={() => processes.refetch()} />;
  const metricData = metrics.data ?? [];
  return <div className="grid gap-5"><div><h1 className="text-xl font-semibold">Training Center</h1><p className="text-sm text-muted-foreground">Managed jobs use the existing training engine and checkpoint lifecycle.</p></div><Card><CardHeader><CardTitle>Start training</CardTitle></CardHeader><CardContent><form className="flex flex-wrap gap-2" onSubmit={submit}><label className="sr-only" htmlFor="training-config">Configuration path</label><input id="training-config" className="h-9 min-w-64 rounded-md border bg-transparent px-3 text-sm" value={configPath} onChange={(event) => setConfigPath(event.target.value)} /><Button type="submit" disabled={start.isPending}>Start</Button>{start.error ? <p role="alert" className="self-center text-sm text-destructive">{start.error.message}</p> : null}</form></CardContent></Card><div className="overflow-x-auto rounded-lg border"><table className="w-full text-left text-sm"><thead className="bg-muted text-xs text-muted-foreground"><tr><th className="p-3">Type</th><th className="p-3">PID</th><th className="p-3">Status</th><th className="p-3">Started</th><th className="p-3">Resources</th><th className="p-3">Actions</th></tr></thead><tbody>{processes.data.map((process) => <tr className="border-t" key={process.id}><td className="p-3">{process.kind}</td><td className="p-3 font-mono">{process.pid ?? "—"}</td><td className="p-3"><StatusBadge value={process.status} /></td><td className="p-3">{formatDuration((Date.now() - new Date(process.started_at).getTime()) / 1000)}</td><td className="p-3">{formatNumber(process.cpu_percent)}% CPU</td><td className="p-3"><div className="flex gap-2"><Button variant="outline" disabled={process.status !== "running" || pause.isPending} onClick={() => pause.mutate(process.id)}>Pause</Button><Button variant="danger" disabled={!['running', 'pausing'].includes(process.status) || stop.isPending} onClick={() => stop.mutate(process.id)}>Stop</Button></div></td></tr>)}</tbody></table></div>{metricData.length > 0 ? <div className="grid gap-5 xl:grid-cols-2"><MetricChart title="Training loss" data={metricData} metric="train/loss" /><MetricChart title="Validation loss" data={metricData} metric="validation/loss" color="#f59e0b" /><MetricChart title="Learning rate" data={metricData} metric="train/learning_rate" color="#60a5fa" /><MetricChart title="Tokens per second" data={metricData} metric="perf/tokens_per_second" color="#34d399" /></div> : null}</div>;
}
