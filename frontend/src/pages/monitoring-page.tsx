import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { SERIES_COLORS, type Point, type Series } from "@/components/charts/chart-utils";
import { TimeSeriesChart } from "@/components/charts/time-series-chart";
import { ErrorState, LoadingState, PageHeader, Panel, Stat } from "@/components/data-state";
import { ProgressBar } from "@/components/ui/progress-bar";
import { Tabs } from "@/components/ui/tabs";
import { useFormat } from "@/hooks/use-format";
import { useSystem, useSystemHistory } from "@/hooks/use-runtime";
import type { SystemSnapshot } from "@/lib/api";

const GIB = 1024 ** 3;
const RANGES = [5, 30, 60] as const;

function line(history: SystemSnapshot[], pick: (item: SystemSnapshot) => number | null): Point[] {
  const points: Point[] = [];
  for (const item of history) {
    const value = pick(item);
    if (value !== null) points.push({ x: new Date(item.recorded_at).getTime(), y: value });
  }
  return points;
}

export function MonitoringPage() {
  const { t } = useTranslation();
  const format = useFormat();
  const system = useSystem();
  const history = useSystemHistory();
  const [range, setRange] = useState<(typeof RANGES)[number]>(30);
  const windowed = useMemo(() => {
    const data = history.data ?? [];
    const last = data.length ? new Date(data[data.length - 1].recorded_at).getTime() : 0;
    return data.filter((item) => new Date(item.recorded_at).getTime() >= last - range * 60_000);
  }, [history.data, range]);
  if (system.isPending || history.isPending) return <LoadingState />;
  if (system.isLoadingError) return <ErrorState error={system.error} retry={() => void system.refetch()} />;
  const current = system.data;
  const totalGb = current.memory.total_bytes / GIB;
  const swapGb = Math.max(current.memory.swap_total_bytes / GIB, 1);
  const deviceUsed = windowed.some((item) => (item.gpu.allocated_bytes ?? 0) > 0 || (item.gpu.driver_allocated_bytes ?? 0) > 0);
  const gb = (value: number) => format.number(value, value >= 10 ? 1 : 2);
  const series = (key: string, label: string, color: number, points: Point[]): Series => ({ key, label, color: SERIES_COLORS[color], points });
  const clock = (value: number) => format.time(value);
  const common = { formatX: clock, goal: "none" as const, allowSmoothing: false };
  return (
    <div className="grid gap-4">
      <PageHeader
        title={t("monitoring.title")}
        subtitle={t("monitoring.subtitle")}
        actions={<Tabs value={String(range) as "5" | "30" | "60"} onChange={(value) => setRange(Number(value) as (typeof RANGES)[number])} items={RANGES.map((value) => ({ value: String(value) as "5" | "30" | "60", label: t("monitoring.minutes", { count: value }) }))} />}
      />
      <div className="grid gap-4 xl:grid-cols-2">
        <TimeSeriesChart {...common} title={t("chart.ram")} unit={t("units.gb")} fixedDomain={[0, Math.ceil(totalGb)]} formatY={gb} series={[series("ram", t("chart.ramUsed"), 0, line(windowed, (item) => item.memory.used_bytes / GIB)), series("proc", t("chart.ramProcess"), 2, line(windowed, (item) => item.memory.python_process_bytes / GIB))]} />
        <TimeSeriesChart {...common} title={t("chart.cpu")} unit="%" fixedDomain={[0, 100]} formatY={(value) => format.number(value, 0)} series={[series("cpu", t("chart.cpu"), 1, line(windowed, (item) => item.cpu.percent))]} />
        {deviceUsed ? <TimeSeriesChart {...common} title={t("chart.mps", { device: current.gpu.device.toUpperCase() })} unit={t("units.gb")} formatY={gb} series={[series("alloc", t("chart.deviceAllocated"), 3, line(windowed, (item) => (item.gpu.allocated_bytes === null ? null : item.gpu.allocated_bytes / GIB))), series("driver", t("chart.deviceDriver"), 4, line(windowed, (item) => (item.gpu.driver_allocated_bytes === null ? null : item.gpu.driver_allocated_bytes / GIB)))]} /> : <Panel title={t("chart.mps", { device: current.gpu.device.toUpperCase() })}><p className="text-sm text-muted-foreground">{t("chart.mpsUnavailable")}</p></Panel>}
        <TimeSeriesChart {...common} title={t("chart.swap")} unit={t("units.gb")} fixedDomain={[0, Math.ceil(swapGb)]} formatY={gb} series={[series("swap", t("chart.swap"), 2, line(windowed, (item) => item.memory.swap_used_bytes / GIB))]} />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title={t("monitoring.perCore")}>
          <div className="grid grid-cols-2 gap-x-4 gap-y-2.5 sm:grid-cols-3 lg:grid-cols-4">
            {current.cpu.per_core_percent.map((percent, index) => (
              <div key={index}>
                <div className="flex justify-between text-xs"><span className="text-muted-foreground">{t("monitoring.core", { index: index + 1 })}</span><span className="tabular-nums">{format.number(percent, 0)}%</span></div>
                <ProgressBar className="mt-1 h-1.5" percent={percent} label={t("monitoring.core", { index: index + 1 })} />
              </div>
            ))}
          </div>
        </Panel>
        <Panel title={t("monitoring.storage")}>
          <dl className="grid grid-cols-2 gap-4">
            <Stat label={t("monitoring.datasets")} value={format.bytes(current.disk.datasets_bytes)} />
            <Stat label={t("monitoring.checkpoints")} value={format.bytes(current.disk.checkpoints_bytes)} />
            <Stat label={t("monitoring.logs")} value={format.bytes(current.disk.logs_bytes)} />
            <Stat label={t("monitoring.database")} value={format.bytes(current.disk.database_bytes)} />
            <Stat label={t("monitoring.diskFree")} value={format.bytes(current.disk.free_bytes)} hint={`${t("hardware.of")} ${format.bytes(current.disk.total_bytes)}`} />
          </dl>
        </Panel>
      </div>
    </div>
  );
}
