import { useTranslation } from "react-i18next";
import { Panel } from "@/components/data-state";
import { ProgressBar } from "@/components/ui/progress-bar";
import { useFormat } from "@/hooks/use-format";
import { useSystem } from "@/hooks/use-runtime";

function Gauge({ label, value, caption, percent }: { label: string; value: string; caption?: string; percent: number | null }) {
  return (
    <div className="grid gap-1.5">
      <div className="flex items-baseline justify-between gap-2 text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium tabular-nums">{value}</span>
      </div>
      <ProgressBar percent={percent} label={label} tone={percent !== null && percent > 90 ? "danger" : percent !== null && percent > 75 ? "warning" : "primary"} />
      {caption ? <p className="text-xs text-muted-foreground tabular-nums">{caption}</p> : null}
    </div>
  );
}

export function HardwareCard() {
  const { t } = useTranslation();
  const format = useFormat();
  const { data } = useSystem();
  if (!data) return <Panel title={t("hardware.title")}><div className="h-24 animate-pulse rounded bg-muted" /></Panel>;
  const memoryPercent = (data.memory.used_bytes / data.memory.total_bytes) * 100;
  const swapPercent = data.memory.swap_total_bytes > 0 ? (data.memory.swap_used_bytes / data.memory.swap_total_bytes) * 100 : 0;
  return (
    <Panel title={t("hardware.title")}>
      <div className="grid gap-4">
        <Gauge label={t("hardware.cpu")} value={format.percent(data.cpu.percent, 0)} percent={data.cpu.percent} caption={t("hardware.cores", { count: data.cpu.logical_cores ?? 0 })} />
        <Gauge label={t("hardware.ram")} value={`${format.bytes(data.memory.used_bytes)} / ${format.bytes(data.memory.total_bytes)}`} percent={memoryPercent} />
        <Gauge label={t("hardware.swap")} value={format.bytes(data.memory.swap_used_bytes)} percent={swapPercent} caption={`${t("hardware.of")} ${format.bytes(data.memory.swap_total_bytes)}`} />
        <Gauge label={t("hardware.device", { device: data.gpu.device.toUpperCase() })} value={data.gpu.allocated_bytes === null ? t("common.unavailable") : format.bytes(data.gpu.allocated_bytes)} percent={null} caption={data.gpu.driver_allocated_bytes === null ? undefined : `${t("hardware.driver")}: ${format.bytes(data.gpu.driver_allocated_bytes)}`} />
      </div>
    </Panel>
  );
}
