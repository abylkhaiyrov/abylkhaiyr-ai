import { Fragment, useState } from "react";
import { ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import { StatusBadge } from "@/components/status-badge";
import { ProcessLogs } from "@/components/processes/process-logs";
import { useFormat } from "@/hooks/use-format";
import type { Process } from "@/lib/api";
import { cn } from "@/lib/utils";

function processName(item: Process): string {
  const source = item.config_path ?? item.output_dir ?? item.model_name;
  return source ? source.split("/").pop()?.replace(/\.ya?ml$/, "") ?? item.id : item.id;
}

export function ProcessTable({ items, live }: { items: Process[]; live: boolean }) {
  const { t } = useTranslation();
  const format = useFormat();
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div className="overflow-x-auto border-y hairline">
      <table className="data-table">
        <thead>
          <tr>
            <th className="w-8" />
            <th>{t("processes.name")}</th>
            <th>{t("processes.kind")}</th>
            <th>{t("processes.status")}</th>
            <th>{t("processes.started")}</th>
            <th>{live ? t("processes.running") : t("processes.finished")}</th>
            <th>{live ? "CPU" : t("processes.result")}</th>
            {live ? <th>{t("processes.memory")}</th> : null}
          </tr>
        </thead>
        <tbody>
          {items.map((item) => {
            const expanded = open === item.id;
            return (
              <Fragment key={item.id}>
                <tr>
                  <td>
                    <button type="button" aria-expanded={expanded} aria-label={t(expanded ? "processes.hideDetails" : "processes.showDetails")} onClick={() => setOpen(expanded ? null : item.id)} className="grid h-6 w-6 place-items-center rounded text-muted-foreground hover:bg-muted">
                      <ChevronRight size={14} className={cn("transition-transform", expanded && "rotate-90")} />
                    </button>
                  </td>
                  <td className="max-w-64"><p className="truncate font-medium" title={processName(item)}>{processName(item)}</p>{item.model_name ? <p className="truncate text-xs text-muted-foreground">{item.model_name}</p> : null}</td>
                  <td>{t(`stage.${item.kind}`, { defaultValue: t(`modelStage.${item.kind}`, { defaultValue: item.kind }) })}</td>
                  <td><StatusBadge value={item.status} />{item.error_reason ? <p className="mt-1 max-w-56 truncate text-xs text-destructive" title={item.error_reason}>{item.error_reason}</p> : null}</td>
                  <td>{format.dateTime(item.started_at)}</td>
                  <td className="tabular-nums">{live ? (item.pid === null ? "—" : `PID ${item.pid}`) : `${format.dateTime(item.completed_at)} · ${format.duration(item.duration_seconds)}`}</td>
                  <td className="tabular-nums">{live ? format.percent(item.cpu_percent, 0) : item.exit_code === null ? "—" : `${t("processes.exitCode")} ${item.exit_code}`}</td>
                  {live ? <td className="tabular-nums">{format.bytes(item.memory_bytes)}</td> : null}
                </tr>
                {expanded ? <tr><td /><td colSpan={live ? 7 : 6}><ProcessLogs item={item} /></td></tr> : null}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
