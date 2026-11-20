import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BadgeCheck } from "lucide-react";
import { useTranslation } from "react-i18next";
import { EmptyState, ErrorState, LoadingState, PageHeader, Panel, Stat } from "@/components/data-state";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Tabs } from "@/components/ui/tabs";
import { useFormat } from "@/hooks/use-format";
import { useMetricFormat } from "@/hooks/use-metric-format";
import { useModelRegistry } from "@/hooks/use-runtime";
import { api, ApiError, type Model, type ModelRole } from "@/lib/api";
import { useToastStore } from "@/store/toast-store";

const ROLES: ModelRole[] = ["production", "best_evaluated", "latest_trained", "loaded"];

function Query<T>({ query, children }: { query: { isPending: boolean; isError: boolean; error: Error | null; data: T | undefined; refetch: () => unknown }; children: (data: T) => React.ReactNode }) {
  if (query.isPending) return <LoadingState />;
  if (query.isError || query.data === undefined) return <ErrorState error={query.error ?? new Error("unknown")} retry={() => void query.refetch()} />;
  return <>{children(query.data)}</>;
}

function modelReason(model: Model, t: (key: string) => string) {
  if (!model.compatible) return t(`modelIssue.${model.compatibility_issues[0] ?? "incompatible_model"}`);
  if (!model.available && model.availability_reason) return t(`modelIssue.${model.availability_reason.split(" ")[0]}`);
  return null;
}

export function ModelsPage() {
  const { t } = useTranslation();
  const format = useFormat();
  const metric = useMetricFormat();
  const client = useQueryClient();
  const push = useToastStore((state) => state.push);
  const query = useModelRegistry();
  const settle = () => void client.invalidateQueries({ queryKey: ["models"] });
  const fail = (error: Error) => push("error", error instanceof ApiError && error.message === "generation_active" ? t("models.generationActive") : error.message);
  const load = useMutation({ mutationFn: (id: string) => api.switchModel(id), onSuccess: settle, onError: fail });
  const unload = useMutation({ mutationFn: api.unloadModel, onSuccess: settle, onError: fail });
  return (
    <div className="grid gap-4">
      <PageHeader title={t("models.title")} subtitle={t("models.subtitle")} />
      <Query query={query}>
        {(registry) => (
          <>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              {ROLES.map((role) => {
                const model = registry.models.find((item) => item.id === registry.roles[role]);
                return (
                  <Panel key={role}>
                    <p className="label-caps">{t(`role.${role}`)}</p>
                    <p className="mt-1 truncate text-base font-semibold" title={model?.id}>{model?.display_name ?? t("roles.none")}</p>
                    <p className="mt-0.5 text-xs text-muted-foreground">{t(`roleHelp.${role}`)}</p>
                  </Panel>
                );
              })}
            </div>
            {registry.models.length === 0 ? <EmptyState title={t("models.empty")} /> : (
              <div className="overflow-x-auto rounded-xl border bg-card">
                <table className="data-table">
                  <thead><tr><th>{t("models.model")}</th><th>{t("models.stage")}</th><th>{t("models.roles")}</th><th>{t("models.parameters")}</th><th>{t("models.context")}</th><th>{t("models.size")}</th><th>{t("models.status")}</th><th /></tr></thead>
                  <tbody>
                    {registry.models.map((model) => {
                      const blocked = modelReason(model, t);
                      return (
                        <tr key={model.id}>
                          <td className="max-w-64"><p className="truncate font-medium" title={model.id}>{model.display_name}</p><p className="truncate text-xs text-muted-foreground" title={model.id}>{model.id}</p></td>
                          <td><span className="rounded bg-muted px-1.5 py-0.5 text-xs uppercase">{t(`modelStage.${model.stage}`, { defaultValue: model.stage })}</span>{model.verified ? <BadgeCheck size={14} className="ml-1.5 inline text-success" aria-label={t("roles.verified")} /> : null}</td>
                          <td><div className="flex flex-wrap gap-1">{model.roles.map((role) => <span key={role} className="rounded border px-1.5 py-0.5 text-xs">{t(`role.${role}`)}</span>)}</div></td>
                          <td className="tabular-nums">{format.compact(model.parameters)}</td>
                          <td className="tabular-nums">{format.number(model.context_length, 0)}</td>
                          <td className="tabular-nums">{format.bytes(model.weights_bytes)}</td>
                          <td>
                            <StatusBadge value={model.loaded ? "active" : blocked ? "failed" : "ready"} />
                            {blocked ? <p className="mt-1 text-xs text-destructive">{blocked}</p> : null}
                            {model.metrics?.["validation/loss"] !== undefined ? <p className="mt-1 text-xs text-muted-foreground">{metric.label("validation_loss")} {metric.value("validation_loss", model.metrics["validation/loss"])}</p> : null}
                          </td>
                          <td className="text-right">
                            {model.loaded ? <Button size="sm" variant="outline" disabled={unload.isPending} onClick={() => unload.mutate()}>{t("models.unload")}</Button> : <Button size="sm" disabled={blocked !== null || load.isPending} title={blocked ?? undefined} onClick={() => load.mutate(model.id)}>{t("models.load")}</Button>}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </Query>
    </div>
  );
}

export function ProcessesPage() {
  const { t } = useTranslation();
  const format = useFormat();
  const query = useQuery({ queryKey: ["processes"], queryFn: api.processes, staleTime: 30_000 });
  return (
    <div className="grid gap-4">
      <PageHeader title={t("processes.title")} subtitle={t("processes.subtitle")} />
      <Query query={query}>
        {(items) => items.length === 0 ? <EmptyState title={t("processes.empty")} /> : (
          <div className="overflow-x-auto rounded-xl border bg-card">
            <table className="data-table">
              <thead><tr><th>{t("processes.kind")}</th><th>PID</th><th>{t("processes.status")}</th><th>{t("processes.started")}</th><th>CPU</th><th>{t("processes.memory")}</th></tr></thead>
              <tbody>
                {items.map((item) => (
                  <tr key={item.id}>
                    <td className="font-medium">{t(`stage.${item.kind}`, { defaultValue: t(`modelStage.${item.kind}`, { defaultValue: item.kind }) })}{item.model_name ? <span className="ml-2 text-xs text-muted-foreground">{item.model_name}</span> : null}</td>
                    <td className="font-mono text-xs">{item.pid ?? "—"}</td>
                    <td><StatusBadge value={item.status} /></td>
                    <td>{format.dateTime(item.started_at)}</td>
                    <td className="tabular-nums">{format.percent(item.cpu_percent, 0)}</td>
                    <td className="tabular-nums">{format.bytes(item.memory_bytes)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Query>
    </div>
  );
}

export function DatasetsPage() {
  const { t } = useTranslation();
  const format = useFormat();
  const query = useQuery({ queryKey: ["datasets"], queryFn: api.datasets, staleTime: 120_000 });
  return (
    <div className="grid gap-4">
      <PageHeader title={t("datasets.title")} subtitle={t("datasets.subtitle")} />
      <Query query={query}>
        {(items) => items.length === 0 ? <EmptyState title={t("datasets.empty")} /> : (
          <div className="grid gap-4">
            {items.map((dataset) => {
              const sources = Object.entries((dataset.manifest.sources ?? {}) as Record<string, { dataset?: string; kept?: number; seen?: number; license?: string }>);
              return (
                <Panel key={dataset.path} title={dataset.name} actions={<span className="text-xs text-muted-foreground">{format.bytes(dataset.size_bytes)}</span>}>
                  <div className="overflow-x-auto">
                    <table className="data-table">
                      <thead><tr><th>{t("datasets.source")}</th><th>{t("datasets.origin")}</th><th>{t("datasets.kept")}</th><th>{t("datasets.seen")}</th><th>{t("datasets.license")}</th></tr></thead>
                      <tbody>{sources.map(([name, item]) => <tr key={name}><td className="font-medium">{name}</td><td>{item.dataset ?? "—"}</td><td className="tabular-nums">{format.number(item.kept, 0)}</td><td className="tabular-nums">{format.number(item.seen, 0)}</td><td>{item.license ?? "—"}</td></tr>)}</tbody>
                    </table>
                  </div>
                </Panel>
              );
            })}
          </div>
        )}
      </Query>
    </div>
  );
}

export function ExperimentsPage() {
  const { t } = useTranslation();
  const format = useFormat();
  const query = useQuery({ queryKey: ["experiments"], queryFn: api.experiments, staleTime: 60_000 });
  return (
    <div className="grid gap-4">
      <PageHeader title={t("experiments.title")} subtitle={t("experiments.subtitle")} />
      <Query query={query}>
        {(items) => items.length === 0 ? <EmptyState title={t("experiments.empty")} /> : (
          <div className="overflow-x-auto rounded-xl border bg-card">
            <table className="data-table">
              <thead><tr><th>{t("experiments.name")}</th><th>{t("metric.validation_loss")}</th><th>{t("metric.validation_perplexity")}</th><th>{t("metric.generation_tokens_per_second")}</th></tr></thead>
              <tbody>
                {items.map((item) => {
                  const loss = item.results.validation_loss ?? item.results.test_loss;
                  const speed = item.results.tokens_per_second ?? item.results.generation_tokens_per_second;
                  return <tr key={`${item.name}-${String(item.checkpoint ?? "")}`}><td className="font-medium">{item.name}</td><td className="tabular-nums">{format.number(loss, 4)}</td><td className="tabular-nums">{format.number(item.results.test_perplexity ?? (loss === undefined ? undefined : Math.exp(loss)), 2)}</td><td className="tabular-nums">{format.number(speed, 0)}</td></tr>;
                })}
              </tbody>
            </table>
          </div>
        )}
      </Query>
    </div>
  );
}

export function EvaluationPage() {
  const { t } = useTranslation();
  const metric = useMetricFormat();
  const query = useQuery({ queryKey: ["evaluation"], queryFn: api.evaluations, staleTime: 60_000 });
  return (
    <div className="grid gap-4">
      <PageHeader title={t("evaluation.title")} subtitle={t("evaluation.subtitle")} />
      <Query query={query}>
        {(items) => items.length === 0 ? <EmptyState title={t("evaluation.empty")} /> : (
          <div className="grid gap-4">
            {items.map((item) => (
              <Panel key={item.path} title={item.path}>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3 lg:grid-cols-5">
                  {Object.entries(item.results).slice(0, 15).map(([key, value]) => <Stat key={key} label={metric.label(key)} value={metric.value(key, value)} />)}
                </dl>
              </Panel>
            ))}
          </div>
        )}
      </Query>
    </div>
  );
}

export function LogsPage() {
  const { t } = useTranslation();
  const [category, setCategory] = useState<"all" | "pipeline" | "training">("all");
  const query = useQuery({ queryKey: ["logs", category], queryFn: () => api.logs(category === "all" ? undefined : category), staleTime: 15_000 });
  return (
    <div className="grid gap-4">
      <PageHeader title={t("logs.title")} subtitle={t("logs.subtitle")} actions={<Tabs value={category} onChange={setCategory} items={(["all", "pipeline", "training"] as const).map((value) => ({ value, label: t(`logs.${value}`) }))} />} />
      <Query query={query}>
        {(data) => data.items.length === 0 ? <EmptyState title={t("logs.empty")} /> : (
          <ul className="max-h-[70vh] divide-y overflow-auto rounded-xl border bg-card font-mono text-xs">
            {data.items.slice(0, 300).map((item, index) => <li key={index} className="grid gap-0.5 px-3 py-1.5 sm:grid-cols-[10rem_minmax(0,1fr)] sm:gap-3"><span className="truncate text-muted-foreground">{item.source.replace("experiments/", "")}</span><span className="break-words">{item.message}</span></li>)}
          </ul>
        )}
      </Query>
    </div>
  );
}
