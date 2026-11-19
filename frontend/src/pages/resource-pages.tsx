import type { ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { EmptyState, ErrorState, LoadingState } from "@/components/data-state";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/status-badge";
import { formatBytes, formatNumber } from "@/lib/utils";

function DataPage({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return <div className="grid gap-5"><div><h1 className="text-xl font-semibold">{title}</h1><p className="text-sm text-muted-foreground">{description}</p></div>{children}</div>;
}

export function ProcessesPage() {
  const query = useQuery({ queryKey: ["processes"], queryFn: api.processes });
  if (query.isPending) return <LoadingState />;
  if (query.isError) return <ErrorState error={query.error} retry={() => query.refetch()} />;
  return <DataPage title="Process Manager" description="Only processes recognised as Abylkhaiyr AI CLI jobs are listed and controllable."><div className="grid gap-3">{query.data.length === 0 ? <EmptyState title="No project processes" description="Start a training or data job to see it here." /> : query.data.map((process) => <Card key={process.id}><CardContent className="grid gap-2 p-4 sm:grid-cols-4"><div><p className="text-xs text-muted-foreground">Process</p><p>{process.kind} · {process.pid ?? "unknown PID"}</p></div><div><p className="text-xs text-muted-foreground">Status</p><StatusBadge value={process.status} /></div><div><p className="text-xs text-muted-foreground">Resources</p><p>{formatNumber(process.cpu_percent)}% CPU · {formatBytes(process.memory_bytes)}</p></div><div><p className="text-xs text-muted-foreground">Configuration</p><p className="truncate">{process.config_path ?? "Not recorded"}</p></div></CardContent></Card>)}</div></DataPage>;
}

export function ModelsPage() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const query = useQuery({ queryKey: ["models"], queryFn: api.models });
  const switchModel = useMutation({
    mutationFn: (checkpointId: string) => api.switchModel(checkpointId),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["models"] });
      void client.invalidateQueries({ queryKey: ["dashboard"] });
    },
  });
  const unloadModel = useMutation({
    mutationFn: api.unloadModel,
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["models"] });
      void client.invalidateQueries({ queryKey: ["dashboard"] });
    },
  });
  if (query.isPending) return <LoadingState />;
  if (query.isError) return <ErrorState error={query.error} retry={() => query.refetch()} />;
  return (
    <DataPage title={t("resources.models.title")} description={t("resources.models.subtitle")}>
      <div className="grid gap-3">
        {query.data.length === 0 ? (
          <EmptyState title={t("resources.models.empty")} description={t("resources.models.subtitle")} />
        ) : (
          query.data.map((model) => (
            <Card key={model.id}>
              <CardContent className="grid gap-3 p-4 sm:grid-cols-4 sm:items-center">
                <div className="min-w-0">
                  <p className="truncate font-medium">{model.checkpoint_id}</p>
                  <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
                    <span className="rounded bg-muted px-1.5 py-0.5 uppercase text-foreground">
                      {t(`resources.models.stages.${model.stage}`, { defaultValue: model.stage })}
                    </span>
                    {model.step !== null && model.step !== undefined ? (
                      <span>#{formatNumber(model.step, 0)}</span>
                    ) : null}
                    <span>{formatBytes(model.weights_bytes)}</span>
                  </div>
                </div>
                <div>
                  <p className="text-sm">
                    {formatNumber(model.parameters, 0)} {t("resources.models.parameters")}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {t("resources.models.vocabulary")} {formatNumber(model.vocab_size, 0)} ·{" "}
                    {t("resources.models.context")} {formatNumber(model.context_length, 0)}
                  </p>
                </div>
                <div>
                  <div className="flex flex-wrap gap-1.5 text-xs">
                    <StatusBadge value={model.loaded ? "active" : model.available ? "ready" : "failed"} />
                    <span className="rounded border px-1.5 py-0.5">
                      {model.compatible
                        ? t("resources.models.compatible")
                        : t("resources.models.incompatible")}
                    </span>
                    {model.device ? (
                      <span className="rounded bg-muted px-1.5 py-0.5 font-mono uppercase">
                        {model.device}
                      </span>
                    ) : null}
                  </div>
                  {model.availability_reason ? (
                    <p className="mt-1 text-xs text-destructive">{model.availability_reason}</p>
                  ) : null}
                </div>
                <div className="flex justify-start gap-2 sm:justify-end">
                  {model.loaded ? (
                    <Button
                      variant="outline"
                      disabled={unloadModel.isPending}
                      onClick={() => unloadModel.mutate()}
                    >
                      {t("resources.models.unload")}
                    </Button>
                  ) : (
                    <Button
                      disabled={!model.compatible || !model.available || switchModel.isPending}
                      onClick={() => switchModel.mutate(model.checkpoint_id)}
                    >
                      {t("resources.models.load")}
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </DataPage>
  );
}

export function DatasetsPage() {
  const query = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  if (query.isPending) return <LoadingState />;
  if (query.isError) return <ErrorState error={query.error} retry={() => query.refetch()} />;
  return <DataPage title="Dataset Manager" description="Dataset manifests are read on the server; corpus contents are never loaded into the browser."><div className="grid gap-3">{query.data.length === 0 ? <EmptyState title="No manifests found" description="Processed datasets with a manifest.json will appear here." /> : query.data.map((dataset) => <Card key={dataset.path}><CardContent className="p-4"><p className="font-medium">{dataset.name}</p><p className="text-sm text-muted-foreground">{dataset.path} · {formatBytes(dataset.size_bytes)}</p><pre className="mt-3 overflow-auto rounded bg-muted p-3 text-xs">{JSON.stringify(dataset.manifest, null, 2)}</pre></CardContent></Card>)}</div></DataPage>;
}

export function ExperimentsPage() {
  const query = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  if (query.isPending) return <LoadingState />;
  if (query.isError) return <ErrorState error={query.error} retry={() => query.refetch()} />;
  return <DataPage title="Experiment Manager" description="History comes directly from experiments/registry.jsonl.">{query.data.length === 0 ? <EmptyState title="No experiments recorded" description="Run a sweep or evaluation to record results." /> : <div className="overflow-x-auto rounded-lg border"><table className="w-full text-left text-sm"><thead className="bg-muted text-xs text-muted-foreground"><tr><th className="p-3">Name</th><th className="p-3">Validation loss</th><th className="p-3">Perplexity</th><th className="p-3">Tokens/s</th></tr></thead><tbody>{query.data.map((experiment) => {
    const valLoss = experiment.results.validation_loss ?? experiment.results.test_loss;
    const ppl = experiment.results.test_perplexity ?? (valLoss !== undefined ? Math.exp(valLoss) : undefined);
    const speed = experiment.results.tokens_per_second ?? experiment.results.generation_tokens_per_second;
    return <tr className="border-t" key={experiment.name}><td className="p-3">{experiment.name}</td><td className="p-3">{formatNumber(valLoss, 4)}</td><td className="p-3">{formatNumber(ppl, 2)}</td><td className="p-3">{formatNumber(speed, 1)}</td></tr>;
  })}</tbody></table></div>}</DataPage>;
}

export function EvaluationPage() {
  const query = useQuery({ queryKey: ["evaluation"], queryFn: api.evaluations });
  if (query.isPending) return <LoadingState />;
  if (query.isError) return <ErrorState error={query.error} retry={() => query.refetch()} />;
  return <DataPage title="Evaluation Center" description="Recorded evaluation output is displayed without invented benchmark scores."><div className="grid gap-3">{query.data.length === 0 ? <EmptyState title="No evaluation results" description="Run the existing evaluate command to record a result." /> : query.data.map((evaluation) => <Card key={evaluation.path}><CardHeader><CardTitle>{evaluation.path}</CardTitle></CardHeader><CardContent><dl className="grid gap-2 sm:grid-cols-3">{Object.entries(evaluation.results).map(([key, value]) => <div key={key}><dt className="text-xs text-muted-foreground">{key}</dt><dd>{formatNumber(value, 4)}</dd></div>)}</dl></CardContent></Card>)}</div></DataPage>;
}

export function LogsPage() {
  const query = useQuery({ queryKey: ["logs"], queryFn: api.logs });
  if (query.isPending) return <LoadingState />;
  if (query.isError) return <ErrorState error={query.error} retry={() => query.refetch()} />;
  return <DataPage title="Logs and Events" description={`${query.data.total} records. Sensitive key-value values are redacted on the server.`}><Card><CardContent className="max-h-[70vh] overflow-auto p-0"><ul className="divide-y font-mono text-xs">{query.data.items.map((item, index) => <li className="p-3" key={`${item.source}-${index}`}><span className="mr-3 text-muted-foreground">{item.source}</span>{item.message}</li>)}</ul></CardContent></Card></DataPage>;
}
