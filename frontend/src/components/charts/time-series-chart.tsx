import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Brush, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState } from "@/components/data-state";
import { downsample, niceDomain, pivot, smooth, summarize, type Series, type Smoothing } from "@/components/charts/chart-utils";
import { useFormat } from "@/hooks/use-format";

type Marker = { x: number; label: string };

type TimeSeriesChartProps = {
  title: string;
  series: Series[];
  unit?: string;
  formatY?: (value: number) => string;
  formatX?: (value: number) => string;
  xLabel?: string;
  goal?: "min" | "max" | "none";
  markers?: Marker[];
  allowLog?: boolean;
  allowSmoothing?: boolean;
  fixedDomain?: [number, number];
  height?: number;
};

const SMOOTHING: Smoothing[] = ["off", "light", "strong"];
const TICK = { fontSize: 11, fill: "hsl(var(--muted-foreground))" };

function Summary({ series, goal, formatY, unit }: { series: Series[]; goal: "min" | "max" | "none"; formatY: (value: number) => string; unit: string }) {
  const { t } = useTranslation();
  return (
    <div className="grid gap-1 text-xs">
      {series.map((item) => {
        const stats = summarize(item.points, goal === "max" ? "max" : "min");
        if (!stats) return null;
        const cells: (readonly [string, number])[] = [
          [t("chart.start"), stats.start],
          [t("chart.current"), stats.current],
        ];
        if (goal !== "none") cells.push([goal === "min" ? t("chart.bestLow") : t("chart.bestHigh"), stats.best]);
        return (
          <div key={item.key} className="flex flex-wrap items-center gap-x-4 gap-y-0.5">
            <span className="flex min-w-0 items-center gap-1.5 font-medium">
              <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: item.color }} />
              <span className="truncate">{item.label}</span>
            </span>
            {cells.map(([label, value]) => (
              <span key={label} className="whitespace-nowrap text-muted-foreground">
                {label}: <span className="tabular-nums text-foreground">{formatY(value)}{unit ? ` ${unit}` : ""}</span>
              </span>
            ))}
          </div>
        );
      })}
    </div>
  );
}

export function TimeSeriesChart({
  title,
  series,
  unit = "",
  formatY,
  formatX,
  xLabel,
  goal = "min",
  markers = [],
  allowLog = false,
  allowSmoothing = true,
  fixedDomain,
  height = 220,
}: TimeSeriesChartProps) {
  const { t } = useTranslation();
  const format = useFormat();
  const [smoothing, setSmoothing] = useState<Smoothing>("off");
  const [logScale, setLogScale] = useState(false);
  const fmtY = useMemo(() => formatY ?? ((value: number) => format.number(value, Math.abs(value) >= 100 ? 0 : 3)), [formatY, format]);
  const fmtX = useMemo(() => formatX ?? ((value: number) => format.number(value, 0)), [formatX, format]);

  const prepared = useMemo(
    () => series.map((item) => ({ ...item, points: downsample(smooth(item.points, smoothing)) })),
    [series, smoothing],
  );
  const data = useMemo(() => pivot(prepared), [prepared]);
  const domain = useMemo(
    () => fixedDomain ?? niceDomain(prepared.flatMap((item) => item.points), logScale),
    [prepared, logScale, fixedDomain],
  );
  const tickWidth = useMemo(() => {
    const [low, high] = domain;
    const samples = [low, high].map((value) => (typeof value === "number" ? fmtY(value).length : 4));
    return Math.max(...samples) * 7 + 14;
  }, [domain, fmtY]);
  const hasData = series.some((item) => item.points.length > 0);

  return (
    <Card className="min-w-0 p-4">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold">{title}{unit ? <span className="ml-1.5 font-normal text-muted-foreground">({unit})</span> : null}</h3>
        <div className="flex flex-wrap items-center gap-1">
          {allowSmoothing
            ? SMOOTHING.map((mode) => (
                <Button key={mode} size="sm" variant={smoothing === mode ? "subtle" : "ghost"} aria-pressed={smoothing === mode} onClick={() => setSmoothing(mode)}>
                  {t(`chart.smoothing.${mode}`)}
                </Button>
              ))
            : null}
          {allowLog ? <Button size="sm" variant={logScale ? "subtle" : "ghost"} aria-pressed={logScale} onClick={() => setLogScale((value) => !value)}>{t("chart.log")}</Button> : null}
        </div>
      </div>
      {hasData ? (
        <>
          <div style={{ height }} className="w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="x" type="number" domain={["dataMin", "dataMax"]} tick={TICK} tickFormatter={fmtX} tickLine={false} axisLine={false} minTickGap={32} label={xLabel ? { value: xLabel, position: "insideBottomRight", offset: -2, fontSize: 10, fill: "hsl(var(--muted-foreground))" } : undefined} height={xLabel ? 32 : 24} />
                <YAxis width={tickWidth} tick={TICK} tickFormatter={fmtY} tickLine={false} axisLine={false} scale={logScale ? "log" : "linear"} domain={domain} allowDataOverflow />
                <Tooltip
                  cursor={{ stroke: "hsl(var(--muted-foreground))", strokeDasharray: "3 3" }}
                  content={({ active, payload, label }) =>
                    active && payload?.length ? (
                      <div className="rounded-lg border bg-card p-2.5 text-xs shadow-lg">
                        <p className="mb-1 font-medium">{xLabel ? `${xLabel}: ` : ""}{fmtX(Number(label))}</p>
                        {payload.map((entry) => (
                          <p key={String(entry.dataKey)} className="flex items-center gap-1.5 tabular-nums">
                            <span className="h-2 w-2 rounded-full" style={{ background: entry.color }} />
                            <span className="text-muted-foreground">{entry.name}</span>
                            <span className="ml-auto pl-3 font-medium">{fmtY(Number(entry.value))}{unit ? ` ${unit}` : ""}</span>
                          </p>
                        ))}
                      </div>
                    ) : null
                  }
                />
                {markers.slice(-24).map((marker) => (
                  <ReferenceLine key={marker.x} x={marker.x} stroke="hsl(var(--muted-foreground))" strokeDasharray="2 4" strokeOpacity={0.6} ifOverflow="hidden" />
                ))}
                {prepared.map((item) => (
                  <Line key={item.key} name={item.label} dataKey={item.key} stroke={item.color} strokeWidth={1.75} dot={false} activeDot={{ r: 3 }} connectNulls isAnimationActive={false} />
                ))}
                {data.length > 40 ? (
                  <Brush dataKey="x" height={18} stroke="hsl(var(--muted-foreground))" fill="hsl(var(--muted))" tickFormatter={fmtX} travellerWidth={8} />
                ) : null}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 border-t pt-2">
            <Summary series={series} goal={goal} formatY={fmtY} unit={unit} />
            {markers.length > 0 ? <p className="mt-1.5 text-[11px] text-muted-foreground">┆ {t("chart.checkpointMarkers")}</p> : null}
          </div>
        </>
      ) : (
        <EmptyState title={t("chart.noData")} />
      )}
    </Card>
  );
}
