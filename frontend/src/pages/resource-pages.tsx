import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { EmptyState, ErrorState, LoadingState, PageHeader, Panel } from "@/components/data-state";
import { Tabs } from "@/components/ui/tabs";
import { useFormat } from "@/hooks/use-format";
import { api } from "@/lib/api";

export function Query<T>({ query, children }: { query: { isPending: boolean; isError: boolean; error: Error | null; data: T | undefined; refetch: () => unknown }; children: (data: T) => React.ReactNode }) {
  if (query.isPending) return <LoadingState />;
  if (query.data === undefined) return <ErrorState error={query.error ?? new Error("unknown")} retry={() => void query.refetch()} />;
  return <>{children(query.data)}</>;
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
          <div className="overflow-x-auto border-y hairline">
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

export function LogsPage() {
  const { t } = useTranslation();
  const [category, setCategory] = useState<"all" | "pipeline" | "training">("all");
  const query = useQuery({ queryKey: ["logs", category], queryFn: () => api.logs(category === "all" ? undefined : category), staleTime: 15_000 });
  return (
    <div className="grid gap-4">
      <PageHeader title={t("logs.title")} subtitle={t("logs.subtitle")} actions={<Tabs value={category} onChange={setCategory} items={(["all", "pipeline", "training"] as const).map((value) => ({ value, label: t(`logs.${value}`) }))} />} />
      <Query query={query}>
        {(data) => data.items.length === 0 ? <EmptyState title={t("logs.empty")} /> : (
          <ul className="max-h-[70vh] divide-y overflow-auto border-y hairline font-mono text-xs">
            {data.items.slice(0, 300).map((item, index) => <li key={index} className="grid gap-0.5 px-3 py-1.5 sm:grid-cols-[10rem_minmax(0,1fr)] sm:gap-3"><span className="truncate text-muted-foreground">{item.source.replace("experiments/", "")}</span><span className="break-words">{item.message}</span></li>)}
          </ul>
        )}
      </Query>
    </div>
  );
}
