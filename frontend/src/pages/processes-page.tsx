import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { EmptyState, ErrorState, LoadingState, PageHeader, Panel } from "@/components/data-state";
import { ProcessTable } from "@/components/processes/process-table";
import { StatusBadge } from "@/components/status-badge";
import { useFormat } from "@/hooks/use-format";
import { api } from "@/lib/api";
import { groupProcesses, type ProcessGroup } from "@/lib/runs";

const SECTIONS: ProcessGroup[] = ["active", "queued", "completed", "failed"];

export function ProcessesPage() {
  const { t } = useTranslation();
  const format = useFormat();
  const processes = useQuery({ queryKey: ["processes"], queryFn: api.processes, staleTime: 30_000 });
  const history = useQuery({ queryKey: ["catalog-runs", "history"], queryFn: () => api.catalogRuns({ limit: 20 }), staleTime: 30_000 });
  const groups = useMemo(() => groupProcesses(processes.data ?? []), [processes.data]);
  if (processes.isPending) return <LoadingState />;
  if (processes.isLoadingError) return <ErrorState error={processes.error} retry={() => void processes.refetch()} />;
  return (
    <div className="grid gap-8">
      <PageHeader title={t("processes.title")} subtitle={t("processes.subtitle")} />
      {SECTIONS.map((group) => (
        <Panel key={group} title={`${t(`processes.groups.${group}`)} · ${groups[group].length}`}>
          {groups[group].length === 0 ? <p className="rounded-md border border-dashed px-3 py-2.5 text-[13px] text-muted-foreground">{t(`processes.empty.${group}`)}</p> : <ProcessTable items={groups[group]} live={group === "active" || group === "queued"} />}
        </Panel>
      ))}
      <Panel title={t("processes.history")}>
        {history.data?.items.length ? (
          <div className="overflow-x-auto border-y hairline">
            <table className="data-table">
              <thead><tr><th>{t("processes.name")}</th><th>{t("processes.kind")}</th><th>{t("processes.status")}</th><th>{t("processes.started")}</th><th>{t("processes.updated")}</th></tr></thead>
              <tbody>
                {history.data.items.map((run) => (
                  <tr key={run.id}>
                    <td className="max-w-64 truncate font-medium" title={run.name}>{run.name}</td>
                    <td>{t(run.kind === "pipeline" ? "processes.pipeline" : `modelStage.${run.stage}`, { defaultValue: run.stage })}</td>
                    <td><StatusBadge value={run.status} />{run.failure_reason ? <p className="mt-1 max-w-56 truncate text-xs text-destructive" title={run.failure_reason}>{run.failure_reason}</p> : null}</td>
                    <td>{format.dateTime(run.created_at)}</td>
                    <td>{format.dateTime(run.updated_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <EmptyState title={t("processes.empty.history")} />}
      </Panel>
    </div>
  );
}
