import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { DataPager } from "@/components/data-pager";
import { PageHeader, Panel } from "@/components/data-state";
import { ProcessTable } from "@/components/processes/process-table";
import { StatusBadge } from "@/components/status-badge";
import { useFormat } from "@/hooks/use-format";
import { api } from "@/lib/api";
import type { ProcessGroup } from "@/lib/runs";

const SECTIONS: ProcessGroup[] = ["active", "queued", "completed", "failed"];

function Section({ group, search }: { group: ProcessGroup; search: string }) {
  const { t } = useTranslation();
  const live = group === "active" || group === "queued";
  return (
    <DataPager queryKey={["processes", group, search]} load={(offset) => api.processPage({ group, q: search.trim() || undefined, offset, limit: live ? 50 : 20 })} empty={t(`processes.empty.${group}`)}>
      {(items, total) => (
        <div className="grid gap-2">
          <p className="text-xs text-muted-foreground" aria-live="polite">{t("history.shown", { shown: items.length, total })}</p>
          <ProcessTable items={items} live={live} />
        </div>
      )}
    </DataPager>
  );
}

export function ProcessesPage() {
  const { t } = useTranslation();
  const format = useFormat();
  const [search, setSearch] = useState("");
  const history = useQuery({ queryKey: ["catalog-runs", "history"], queryFn: () => api.catalogRuns({ limit: 20 }), staleTime: 30_000 });
  return (
    <div className="grid gap-8">
      <PageHeader title={t("processes.title")} subtitle={t("processes.subtitle")} actions={<label className="grid gap-1 text-xs font-medium">{t("history.search")}<input type="search" className="field w-64" value={search} onChange={(event) => setSearch(event.target.value)} /></label>} />
      {SECTIONS.map((group) => (
        <Panel key={group} title={t(`processes.groups.${group}`)}>
          <Section group={group} search={search} />
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
        ) : <p className="rounded-md border border-dashed px-3 py-2.5 text-[13px] text-muted-foreground">{t("processes.empty.history")}</p>}
      </Panel>
    </div>
  );
}
