import { useTranslation } from "react-i18next";
import { Stat } from "@/components/data-state";
import { StatusBadge } from "@/components/status-badge";
import type { RuntimeState } from "@/lib/api";

export function OverviewStats({ runtime }: { runtime: RuntimeState }) {
  const { t } = useTranslation();
  const { runs, models, production } = runtime;
  const bestVerified = models.best_evaluated?.verified ? models.best_evaluated : null;
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-4 border-b hairline pb-6 md:grid-cols-3 xl:grid-cols-6">
      <Stat label={t("dashboard.activity")} value={runs.active ? runs.active.name : t("dashboard.idle")} hint={runs.active ? <StatusBadge value="running" /> : undefined} />
      <Stat label={t("dashboard.latestExperiment")} value={runs.latest?.name ?? "—"} hint={runs.latest ? <StatusBadge value={runs.latest.status} /> : undefined} />
      <Stat label={t("dashboard.completedExperiments")} value={runs.completed_count} />
      <Stat label={t("dashboard.chatModel")} value={models.loaded?.display_name ?? t("dashboard.noChatModel")} />
      <Stat label={t("dashboard.lastTrained")} value={models.latest_trained?.display_name ?? "—"} />
      <Stat label={t("dashboard.bestVerified")} value={bestVerified?.display_name ?? "—"} hint={production?.ready ? undefined : t("dashboard.noProduction")} />
    </dl>
  );
}
