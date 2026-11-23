import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { CompletedTraining } from "@/components/dashboard/completed-training";
import { EventsCard } from "@/components/dashboard/events-card";
import { HardwareCard } from "@/components/dashboard/hardware-card";
import { HeroCard } from "@/components/dashboard/hero-card";
import { ModelRolesCard } from "@/components/dashboard/model-roles-card";
import { OverviewStats } from "@/components/dashboard/overview-stats";
import { RunSummaryCard } from "@/components/dashboard/run-summary-card";
import { EmptyState, ErrorState, LoadingState, PageHeader, Panel } from "@/components/data-state";
import { PipelineTimeline } from "@/components/pipeline/pipeline-timeline";
import { Select } from "@/components/ui/select";
import { useRuntime } from "@/hooks/use-runtime";
import { api, type RunEntry } from "@/lib/api";
import { pickRun } from "@/lib/runs";
import { cn } from "@/lib/utils";

export function DashboardPage() {
  const { t } = useTranslation();
  const query = useRuntime();
  const [selected, setSelected] = useState("");
  const catalog = useQuery({ queryKey: ["catalog-runs", "dashboard"], queryFn: () => api.catalogRuns({ limit: 100 }), staleTime: 30_000 });
  const run: RunEntry | null = query.data ? pickRun(query.data.runs, catalog.data?.items, selected) : null;
  const pipelines = useQuery({ queryKey: ["pipelines"], queryFn: api.pipelines, enabled: run?.kind === "pipeline", staleTime: 30_000 });
  if (query.isPending) return <LoadingState />;
  if (query.isLoadingError) return <ErrorState error={query.error} retry={() => void query.refetch()} />;
  const runtime = query.data;
  const livePipeline = runtime.pipeline?.id === run?.name ? runtime.pipeline : null;
  const pipeline = run?.kind === "pipeline" ? livePipeline ?? pipelines.data?.find((item) => item.id === run.name) ?? null : null;
  return (
    <div className="grid gap-8">
      <PageHeader title={t("dashboard.title")} subtitle={t("dashboard.subtitle")} actions={catalog.data?.items.length ? (
        <label className="grid gap-1 text-xs">{t("dashboard.run")}
          <Select value={selected} onChange={(event) => setSelected(event.target.value)}>
            <option value="">{t("dashboard.autoRun")}</option>
            {catalog.data.items.map((item) => <option key={item.id} value={item.id}>{item.name} · {t(`status.${item.status}`)}</option>)}
          </Select>
        </label>
      ) : null} />
      <OverviewStats runtime={runtime} />
      {run === null ? <EmptyState title={t("dashboard.noRuns")} description={t("dashboard.noRunsText")} /> : (
        <>
          {run.kind === "pipeline" ? <HeroCard pipeline={pipeline} /> : <RunSummaryCard run={run} />}
          <div className={cn("grid items-start gap-10", pipeline ? "xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]" : "lg:grid-cols-2")}>
            {pipeline ? (
              <div className="grid gap-8">
                <Panel title={t("dashboard.pipeline")}>
                  <PipelineTimeline pipeline={pipeline} />
                </Panel>
                <CompletedTraining pipeline={pipeline} />
              </div>
            ) : null}
            <div className={cn("grid gap-8", !pipeline && "lg:col-span-2 lg:grid-cols-2")}>
              <ModelRolesCard runtime={runtime} />
              <HardwareCard />
              <EventsCard events={runtime.events} />
            </div>
          </div>
        </>
      )}
      {run === null ? (
        <div className="grid items-start gap-8 xl:grid-cols-2">
          <ModelRolesCard runtime={runtime} />
          <HardwareCard />
        </div>
      ) : null}
    </div>
  );
}
