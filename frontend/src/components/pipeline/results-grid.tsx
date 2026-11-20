import { Stat } from "@/components/data-state";
import { useMetricFormat } from "@/hooks/use-metric-format";

export function ResultsGrid({ entries, columns = "grid-cols-2 sm:grid-cols-3" }: { entries: [string, number | string][]; columns?: string }) {
  const metric = useMetricFormat();
  return (
    <dl className={`grid gap-x-4 gap-y-3 ${columns}`}>
      {entries.map(([key, value]) => (
        <Stat key={key} label={metric.label(key)} value={metric.value(key, value)} />
      ))}
    </dl>
  );
}
