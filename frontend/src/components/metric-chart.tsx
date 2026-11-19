import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Metric } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function MetricChart({ title, data, metric, color = "#14b8a6" }: { title: string; data: Metric[]; metric: string; color?: string }) {
  const points = data.filter((item) => typeof item[metric] === "number");
  return <Card><CardHeader><CardTitle>{title}</CardTitle></CardHeader><CardContent className="h-56"><ResponsiveContainer width="100%" height="100%"><LineChart data={points}><XAxis dataKey="step" tickLine={false} axisLine={false} /><YAxis width={44} tickLine={false} axisLine={false} /><Tooltip /><Line type="monotone" dataKey={metric} stroke={color} dot={false} strokeWidth={2} isAnimationActive={false} /></LineChart></ResponsiveContainer></CardContent></Card>;
}
