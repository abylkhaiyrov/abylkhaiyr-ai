import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { CompletedTraining } from "@/components/dashboard/completed-training";
import { EventsCard } from "@/components/dashboard/events-card";
import { HardwareCard } from "@/components/dashboard/hardware-card";
import { HeroCard } from "@/components/dashboard/hero-card";
import { ModelRolesCard } from "@/components/dashboard/model-roles-card";
import { ErrorState, LoadingState, PageHeader, Panel } from "@/components/data-state";
import { PipelineTimeline } from "@/components/pipeline/pipeline-timeline";
import { api } from "@/lib/api";
import { Select } from "@/components/ui/select";
import { useRuntime } from "@/hooks/use-runtime";

export function DashboardPage() {
  const { t } = useTranslation();
  const query = useRuntime();
  const history = useQuery({ queryKey: ["pipelines"], queryFn: api.pipelines });
  const [selected, setSelected] = useState("");
  if (query.isPending) return <LoadingState />;
  if (query.isLoadingError) return <ErrorState error={query.error} retry={() => void query.refetch()} />;
  const runtime = query.data;
  const pipeline = selected && selected !== runtime.pipeline?.id
    ? history.data?.find((item) => item.id === selected) ?? runtime.pipeline
    : runtime.pipeline;
  return (
    <div className="grid gap-8">
      <PageHeader title={t("dashboard.title")} subtitle={t("dashboard.subtitle")} actions={history.data?.length ? (
        <label className="grid gap-1 text-xs">{t("evaluation.experiment")}
          <Select value={selected} onChange={(event) => setSelected(event.target.value)}>
            <option value="">{t("dashboard.currentExperiment")}</option>
            {history.data.map((item) => <option key={item.id} value={item.id}>{item.model_name}</option>)}
          </Select>
        </label>
      ) : null} />
      <HeroCard pipeline={pipeline} />
      <div className="grid items-start gap-10 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div className="grid gap-8">
          {pipeline ? (
            <Panel title={t("dashboard.pipeline")}>
              <PipelineTimeline pipeline={pipeline} />
            </Panel>
          ) : null}
          <CompletedTraining pipeline={pipeline} />
        </div>
        <div className="grid gap-8">
          <ModelRolesCard runtime={runtime} />
          <HardwareCard />
          <EventsCard events={runtime.events} />
        </div>
      </div>
    </div>
  );
}
