import { useTranslation } from "react-i18next";
import { CompletedTraining } from "@/components/dashboard/completed-training";
import { EventsCard } from "@/components/dashboard/events-card";
import { HardwareCard } from "@/components/dashboard/hardware-card";
import { HeroCard } from "@/components/dashboard/hero-card";
import { ModelRolesCard } from "@/components/dashboard/model-roles-card";
import { ErrorState, LoadingState, PageHeader, Panel } from "@/components/data-state";
import { PipelineTimeline } from "@/components/pipeline/pipeline-timeline";
import { useRuntime } from "@/hooks/use-runtime";

export function DashboardPage() {
  const { t } = useTranslation();
  const query = useRuntime();
  if (query.isPending) return <LoadingState />;
  if (query.isError) return <ErrorState error={query.error} retry={() => void query.refetch()} />;
  const runtime = query.data;
  return (
    <div className="grid gap-4">
      <PageHeader title={t("dashboard.title")} subtitle={t("dashboard.subtitle")} />
      <HeroCard pipeline={runtime.pipeline} />
      <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div className="grid gap-4">
          {runtime.pipeline ? (
            <Panel title={t("dashboard.pipeline")}>
              <PipelineTimeline pipeline={runtime.pipeline} />
            </Panel>
          ) : null}
          <CompletedTraining pipeline={runtime.pipeline} />
        </div>
        <div className="grid gap-4">
          <ModelRolesCard runtime={runtime} />
          <HardwareCard />
          <EventsCard events={runtime.events} />
        </div>
      </div>
    </div>
  );
}
