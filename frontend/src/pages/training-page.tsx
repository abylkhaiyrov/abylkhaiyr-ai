import { useEffect, useMemo, useState } from "react";
import { useQueries, useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { RunCharts } from "@/components/charts/run-charts";
import { EmptyState, ErrorState, LoadingState, PageHeader } from "@/components/data-state";
import { CheckpointsTable } from "@/components/training/checkpoints-table";
import { ProcessControls } from "@/components/training/process-controls";
import { RunList } from "@/components/training/run-list";
import { RunOverview } from "@/components/training/run-overview";
import { ConfigView, LogView } from "@/components/training/text-views";
import { Tabs } from "@/components/ui/tabs";
import { useActiveMetrics, useRuntime } from "@/hooks/use-runtime";
import { api, type Metric } from "@/lib/api";

type Tab = "overview" | "charts" | "checkpoints" | "compare" | "config" | "logs";

export function TrainingPage() {
  const { t } = useTranslation();
  const runtime = useRuntime().data;
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.runs, staleTime: 30_000 });
  const [selected, setSelected] = useState<string | null>(null);
  const [compared, setCompared] = useState<string[]>([]);
  const [tab, setTab] = useState<Tab>("overview");
  const activeKey = runtime?.pipeline?.running ? runtime.pipeline.current_stage : null;
  const pipelineState = runtime?.pipeline?.running ? `${runtime.pipeline.current_stage}:${runtime.pipeline.updated_at}` : "idle";

  useEffect(() => {
    if (selected === null && runs.data && runs.data.length > 0) {
      setSelected((runs.data.find((run) => run.status === "running") ?? runs.data[0]).name);
    }
  }, [runs.data, selected]);
  useEffect(() => {
    void runs.refetch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pipelineState]);

  const summary = runs.data?.find((run) => run.name === selected) ?? null;
  const detail = useQuery({ queryKey: ["run", selected], queryFn: () => api.run(selected ?? ""), enabled: selected !== null, staleTime: 30_000 });
  const live = summary?.status === "running";
  const metrics = useActiveMetrics(selected, live);
  const comparedNames = useMemo(() => (selected ? [selected, ...compared.filter((name) => name !== selected)] : []), [selected, compared]);
  const comparedQueries = useQueries({
    queries: comparedNames.map((name) => ({ queryKey: ["metrics", name], queryFn: () => api.runMetrics(name), staleTime: 60_000 })),
  });

  if (runs.isPending) return <LoadingState />;
  if (runs.isError) return <ErrorState error={runs.error} retry={() => void runs.refetch()} />;
  const stage = runtime?.pipeline?.stages.find((item) => item.key === activeKey && item.output_dir === summary?.output_dir) ?? null;
  const rows: Metric[] = metrics.data ?? [];
  const toggle = (name: string) => setCompared((previous) => (previous.includes(name) ? previous.filter((item) => item !== name) : [...previous, name]));
  const tabs: { value: Tab; label: string }[] = [
    { value: "overview", label: t("training.tabs.overview") },
    { value: "charts", label: t("training.tabs.charts") },
    { value: "checkpoints", label: t("training.tabs.checkpoints") },
    { value: "compare", label: t("training.tabs.compare") },
    { value: "config", label: t("training.tabs.config") },
    { value: "logs", label: t("training.tabs.logs") },
  ];

  return (
    <div className="grid gap-4">
      <PageHeader title={t("training.title")} subtitle={t("training.subtitle")} />
      <ProcessControls />
      <div className="grid items-start gap-4 lg:grid-cols-[18rem_minmax(0,1fr)]">
        <aside className="grid gap-2">
          <p className="label-caps">{t("training.experiments", { count: runs.data.length })}</p>
          {runs.data.length === 0 ? <EmptyState title={t("training.noRuns")} /> : <RunList runs={runs.data} selected={selected} compared={compared} onSelect={setSelected} onToggleCompare={toggle} />}
        </aside>
        <section className="grid min-w-0 gap-4">
          {summary && detail.data ? (
            <>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="truncate text-lg font-semibold">{summary.name}</h2>
                <Tabs value={tab} onChange={setTab} items={tabs} />
              </div>
              {tab === "overview" ? <RunOverview run={detail.data} rows={rows} stage={stage} /> : null}
              {tab === "charts" ? <RunCharts runs={[{ name: summary.name, rows }]} stage={summary.stage} markers={detail.data.checkpoints.map((item) => item.step ?? 0)} /> : null}
              {tab === "checkpoints" ? <CheckpointsTable run={detail.data} /> : null}
              {tab === "compare" ? (
                comparedNames.length < 2 ? <EmptyState title={t("training.compareHint")} /> : (
                  <RunCharts runs={comparedNames.map((name, index) => ({ name, rows: comparedQueries[index]?.data ?? [] }))} stage={summary.stage} />
                )
              ) : null}
              {tab === "config" ? <ConfigView run={detail.data} /> : null}
              {tab === "logs" ? <LogView name={summary.name} live={live} /> : null}
            </>
          ) : detail.isPending && selected ? <LoadingState /> : <EmptyState title={t("training.selectRun")} />}
        </section>
      </div>
    </div>
  );
}
