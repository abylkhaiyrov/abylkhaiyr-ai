import { useTranslation } from "react-i18next";
import { checkpointLabel, formatInterval, formatMetricValue, metricsOf, rowId, type SectionKey } from "@/lib/evaluation";
import type { EvaluationRow, MetricDefinition } from "@/lib/api";

type Props = { section: SectionKey; catalog: MetricDefinition[]; rows: EvaluationRow[] };

export function MetricSection({ section, catalog, rows }: Props) {
  const { t, i18n } = useTranslation();
  const definitions = metricsOf(catalog, section);
  return (
    <div className="overflow-x-auto border-y hairline">
      <table className="data-table">
        <thead>
          <tr>
            <th scope="col">{t("evaluation.metric")}</th>
            {rows.map((row) => <th scope="col" key={rowId(row)} className="max-w-40 truncate" title={row.checkpoint ?? undefined}>{checkpointLabel(row.checkpoint)}</th>)}
          </tr>
        </thead>
        <tbody>
          {definitions.map((definition) => (
            <tr key={definition.key}>
              <th scope="row" className="min-w-64 max-w-md text-left font-normal">
                <p className="font-medium">{t(`evalMetric.${definition.key}.name`)}</p>
                <p className="mt-0.5 text-xs text-muted-foreground">{t(`evalMetric.${definition.key}.about`)}</p>
                <p className="mt-0.5 text-xs text-muted-foreground">{t(`evalCenter.${definition.direction}`)}</p>
              </th>
              {rows.map((row) => {
                const shown = formatMetricValue(definition, row.results[definition.key], i18n.language);
                const interval = formatInterval(definition, row.intervals[definition.key], i18n.language);
                const samples = row.samples[definition.key];
                return (
                  <td key={rowId(row)} className="tabular-nums">
                    {shown ?? <span className="text-muted-foreground">{t("evalCenter.notVerified")}</span>}
                    {shown && samples ? <p className="text-xs text-muted-foreground">{t("evalCenter.samples", { count: samples })}{interval ? ` · ${t("evalCenter.interval")} ${interval}` : ""}</p> : null}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
