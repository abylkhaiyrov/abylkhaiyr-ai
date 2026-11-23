import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useTranslation } from "react-i18next";
import { EmptyState, ErrorState, LoadingState, Panel } from "@/components/data-state";
import { GateList } from "@/components/evaluation/gate-list";
import { Select } from "@/components/ui/select";
import { api, type Comparison, type MetricDefinition } from "@/lib/api";
import { checkpointLabel, formatDelta, formatMetricValue } from "@/lib/evaluation";
import { cn } from "@/lib/utils";

type Props = { group: string; checkpoints: string[]; catalog: MetricDefinition[] };

function Trend({ improved, delta }: { improved: boolean | null; delta: number | null }) {
  if (delta === null) return <span className="text-muted-foreground">—</span>;
  if (delta === 0 || improved === null) return <Minus size={13} aria-label="same" className="text-muted-foreground" />;
  const Icon = delta > 0 ? ArrowUp : ArrowDown;
  return <Icon size={13} aria-label={improved ? "improved" : "worse"} className={improved ? "text-success" : "text-destructive"} />;
}

function Result({ data, catalog }: { data: Comparison; catalog: MetricDefinition[] }) {
  const { t, i18n } = useTranslation();
  const [chartKey, setChartKey] = useState("");
  const defs = new Map(catalog.map((item) => [item.key, item]));
  const names = data.checkpoints.map((item) => checkpointLabel(item.checkpoint));
  const active = data.metrics.find((item) => item.key === chartKey) ?? data.metrics.find((item) => item.entries.every((entry) => entry.value !== null)) ?? data.metrics[0];
  const unit = active ? defs.get(active.key) : undefined;
  const bars = active ? names.map((name, index) => ({ name, value: active.entries[index].value })) : [];
  const leader = data.verdict.checkpoint ? checkpointLabel(data.verdict.checkpoint) : "";
  return (
    <div className="grid gap-8">
      <section className="grid gap-2 border-b hairline pb-6">
        <h3 className="text-[13px] font-semibold">{t("evalCenter.verdictTitle")}</h3>
        <p className={cn("max-w-2xl text-[15px]", data.verdict.status === "leader" ? "text-foreground" : "text-warning")}>{t(`evalCenter.verdict.${data.verdict.status}`, { name: leader })}</p>
        <ul className="grid gap-0.5 text-xs text-muted-foreground">{data.caveats.map((item) => <li key={item}>{t(`evalCenter.caveats.${item}`)}</li>)}</ul>
      </section>
      <Panel title={t("evalCenter.gates")}>
        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
          {data.checkpoints.map((item, index) => (
            <div key={item.checkpoint ?? index} className="grid content-start gap-2">
              <p className="truncate text-[13px] font-medium" title={item.checkpoint ?? undefined}>{names[index]}</p>
              <GateList gates={item.gates} />
            </div>
          ))}
        </div>
      </Panel>
      <Panel title={t("evaluation.comparison")}>
        <div className="overflow-x-auto border-y hairline">
          <table className="data-table">
            <thead><tr><th scope="col">{t("evaluation.metric")}</th>{names.map((name, index) => <th scope="col" key={data.checkpoints[index].checkpoint ?? index} className="max-w-40 truncate">{name}</th>)}</tr></thead>
            <tbody>
              {data.metrics.map((metric) => {
                const definition = defs.get(metric.key);
                return (
                  <tr key={metric.key}>
                    <th scope="row" className="text-left font-normal"><p className="font-medium">{t(`evalMetric.${metric.key}.name`, { defaultValue: metric.key })}</p><p className="text-xs text-muted-foreground">{t(`evalCenter.${metric.direction}`)}</p></th>
                    {metric.entries.map((entry, index) => {
                      const shown = definition ? formatMetricValue(definition, entry.value, i18n.language) : null;
                      const change = definition && index > 0 ? formatDelta(definition, entry.delta, i18n.language) : null;
                      return (
                        <td key={index} className="tabular-nums">
                          {shown ?? <span className="text-muted-foreground">{t("evalCenter.notVerified")}</span>}
                          {change ? <span className="ml-2 inline-flex items-center gap-1 text-xs"><Trend improved={entry.improved} delta={entry.delta} />{change}{entry.relative !== null ? ` (${new Intl.NumberFormat(i18n.language, { maximumFractionDigits: 0, signDisplay: "always" }).format(entry.relative * 100)}%)` : ""}</span> : null}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>
      {active && unit ? (
        <Panel title={t("evaluation.chart")} actions={<label className="grid gap-1 text-xs">{t("evalCenter.chartMetric")}<Select inline value={active.key} onChange={(event) => setChartKey(event.target.value)}>{data.metrics.map((item) => <option key={item.key} value={item.key}>{t(`evalMetric.${item.key}.name`, { defaultValue: item.key })}</option>)}</Select></label>}>
          <p className="mb-2 text-xs text-muted-foreground">{t(`evalCenter.${active.direction}`)}</p>
          <div className="h-56" role="img" aria-label={t(`evalMetric.${active.key}.name`, { defaultValue: active.key })}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={bars} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
                <CartesianGrid vertical={false} stroke="hsl(var(--border))" />
                <XAxis dataKey="name" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} tickLine={false} axisLine={false} interval={0} />
                <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} tickLine={false} axisLine={false} width={48} tickFormatter={(value: number) => formatMetricValue(unit, value, i18n.language) ?? ""} />
                <Tooltip cursor={{ fill: "hsl(var(--muted))" }} contentStyle={{ background: "hsl(var(--background))", border: "1px solid hsl(var(--border))", borderRadius: 8, fontSize: 12 }} formatter={(value: number) => formatMetricValue(unit, value, i18n.language) ?? t("evalCenter.notVerified")} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                  {bars.map((item, index) => <Cell key={item.name} fill={index === 0 ? "hsl(var(--muted-foreground))" : "hsl(var(--accent))"} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      ) : null}
    </div>
  );
}

export function ComparePanel({ group, checkpoints, catalog }: Props) {
  const { t } = useTranslation();
  const enabled = checkpoints.length >= 2;
  const query = useQuery({ queryKey: ["evaluation-compare", group, checkpoints], queryFn: () => api.compareEvaluations(group, checkpoints), enabled, staleTime: 60_000 });
  if (!enabled) return <EmptyState title={t("evalCenter.pick")} description={t("evalCenter.comparableNote")} />;
  if (query.isPending) return <LoadingState />;
  if (query.isError) return <ErrorState error={query.error} retry={() => void query.refetch()} />;
  return <Result data={query.data} catalog={catalog} />;
}
