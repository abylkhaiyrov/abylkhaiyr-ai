import { AlertTriangle } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Panel } from "@/components/data-state";
import { useFormat } from "@/hooks/use-format";
import type { RunEntry } from "@/lib/api";
import { COMPARE_FIELDS, datasetsDiffer } from "@/lib/history";

export function RunCompare({ runs }: { runs: RunEntry[] }) {
  const { t } = useTranslation();
  const format = useFormat();
  const show = (key: string, value: string | number | null) => {
    if (value === null) return "—";
    if (key === "duration" && typeof value === "number") return format.duration(value);
    if (key === "status") return t(`status.${value}`);
    if (key === "stage") return t(`modelStage.${value}`, { defaultValue: String(value) });
    if (typeof value !== "number") return value;
    return format.number(value, key.includes("Loss") || key === "perplexity" ? 3 : 0);
  };
  return (
    <Panel title={t("history.compare")}>
      {datasetsDiffer(runs) ? <p role="note" className="mb-3 flex gap-2 text-xs text-warning"><AlertTriangle size={13} className="mt-0.5 shrink-0" />{t("history.datasetsDiffer")}</p> : null}
      <div className="overflow-x-auto border-y hairline">
        <table className="data-table">
          <thead><tr><th scope="col" />{runs.map((run) => <th scope="col" key={run.id} className="max-w-40 truncate" title={run.name}>{run.name}</th>)}</tr></thead>
          <tbody>
            {COMPARE_FIELDS.map((field) => (
              <tr key={field.key}>
                <th scope="row" className="text-left font-normal text-muted-foreground">{t(`history.fields.${field.key}`)}</th>
                {runs.map((run) => <td key={run.id} className="tabular-nums">{show(field.key, field.read(run))}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
