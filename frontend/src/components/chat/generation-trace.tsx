import { useState } from "react";
import { CheckCircle2, ChevronDown, ChevronRight, Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useFormat } from "@/hooks/use-format";
import type { ReasoningStageEvent } from "@/lib/api";
import type { ActiveGeneration } from "@/lib/generation";

function chips(event: ReasoningStageEvent, t: (key: string, options?: Record<string, unknown>) => string): string[] {
  const data = event.data ?? {};
  const result: string[] = [];
  if (typeof data.kind === "string") result.push(t(`trace.kinds.${data.kind}`, { defaultValue: data.kind }));
  if (typeof data.input_tokens === "number" && typeof data.context_limit === "number") result.push(`${data.input_tokens}/${data.context_limit}`);
  if (typeof data.dropped_messages === "number" && data.dropped_messages > 0) result.push(t("trace.dropped", { count: data.dropped_messages }));
  if (typeof data.attempt === "number" && typeof data.total === "number") result.push(`#${data.attempt}/${data.total}`);
  if (typeof data.tool === "string") result.push(data.tool);
  if (typeof data.tokens_used === "number") result.push(t("chat.tokens", { count: data.tokens_used }));
  if (typeof data.output_tokens === "number") result.push(t("chat.tokens", { count: data.output_tokens }));
  if (data.verified === true) result.push(t("trace.verified"));
  if (typeof data.device === "string") result.push(data.device.toUpperCase());
  if (typeof data.checkpoint_id === "string") result.push(data.checkpoint_id.split("/").slice(-1)[0]);
  return result;
}

export function GenerationTrace({ events, active }: { events: ReasoningStageEvent[]; active: ActiveGeneration | null }) {
  const { t } = useTranslation();
  const format = useFormat();
  const [open, setOpen] = useState(false);
  const queued = active?.status === "queued";
  const waiting = active && active.status === "running" && !active.text && events.length === 0;
  if (events.length === 0 && !queued && !waiting) return null;
  const running = active !== null && (queued || events.some((item) => item.status === "running") || !active.text);
  const summary = queued
    ? active.queuePosition > 0
      ? t("trace.queued", { position: active.queuePosition })
      : t("trace.waitingCompute")
    : waiting
      ? t("trace.preparing")
      : t("trace.steps", { count: events.length });
  return (
    <div className="mb-2 rounded-lg border bg-muted/40 text-xs">
      <button type="button" aria-expanded={open} onClick={() => setOpen((value) => !value)} className="flex w-full items-center gap-2 px-3 py-2 text-left">
        {running ? <Loader2 size={13} className="animate-spin text-info" /> : <CheckCircle2 size={13} className="text-success" />}
        <span className="font-medium">{summary}</span>
        {active?.queueReason ? <span className="truncate text-muted-foreground">· {t(`queueReason.${active.queueReason}`, { defaultValue: active.queueReason })}</span> : null}
        <span className="ml-auto text-muted-foreground">{open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}</span>
      </button>
      {open ? (
        <ol className="grid gap-1.5 border-t px-3 py-2.5">
          {events.map((event) => (
            <li key={`${event.stage}-${event.timestamp}`} className="flex flex-wrap items-center gap-x-2 gap-y-1">
              {event.status === "running" ? <Loader2 size={12} className="animate-spin text-info" /> : <CheckCircle2 size={12} className="text-success" />}
              <span>{t(`trace.stages.${event.stage}`, { defaultValue: event.stage })}</span>
              {chips(event, t as never).map((chip) => <span key={chip} className="rounded bg-background px-1.5 py-0.5 text-muted-foreground">{chip}</span>)}
              {event.timestamp ? <span className="ml-auto text-muted-foreground tabular-nums">{format.time(event.timestamp * 1000)}</span> : null}
            </li>
          ))}
        </ol>
      ) : null}
    </div>
  );
}
