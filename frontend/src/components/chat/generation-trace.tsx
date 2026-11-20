import { useState } from "react";
import { CheckCircle2, ChevronDown, ChevronRight, Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useFormat } from "@/hooks/use-format";
import type { ReasoningStageEvent } from "@/lib/api";
import type { ActiveGeneration } from "@/lib/generation";

type Translate = (key: string, options?: Record<string, unknown>) => string;

function chips(event: ReasoningStageEvent, t: Translate): string[] {
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

function phaseLabel(events: ReasoningStageEvent[], active: ActiveGeneration | null, t: Translate): string | null {
  if (active === null) return null;
  if (active.status === "queued") return active.queuePosition > 0 ? t("trace.queued", { position: active.queuePosition }) : t("trace.waitingCompute");
  const running = [...events].reverse().find((event) => event.status === "running");
  if (running) return t("trace.running", { stage: t(`trace.stages.${running.stage}`, { defaultValue: running.stage }) });
  if (!active.text) return events.length === 0 ? t("trace.preparing") : t("trace.generating");
  return t("trace.generating");
}

export function GenerationTrace({ events, active }: { events: ReasoningStageEvent[]; active: ActiveGeneration | null }) {
  const { t } = useTranslation();
  const format = useFormat();
  const [open, setOpen] = useState(false);
  const live = active !== null && ["queued", "running"].includes(active.status);
  const label = live ? phaseLabel(events, active, t as Translate) : t("trace.done", { count: events.length });
  if (!live && events.length === 0) return null;
  return (
    <div className="mb-3 text-xs">
      <button type="button" aria-expanded={open} onClick={() => setOpen((value) => !value)} className="flex items-center gap-1.5 rounded-md py-0.5 pr-1.5 text-muted-foreground hover:text-foreground">
        {live ? <Loader2 size={12} className="animate-spin text-accent" /> : <CheckCircle2 size={12} className="text-success" />}
        <span className="font-medium">{label}</span>
        {active?.queueReason ? <span className="hidden truncate sm:inline">· {t(`queueReason.${active.queueReason}`, { defaultValue: active.queueReason })}</span> : null}
        {events.length > 0 ? open ? <ChevronDown size={12} /> : <ChevronRight size={12} /> : null}
      </button>
      {open && events.length > 0 ? (
        <ol className="mt-1.5 grid gap-1.5 border-l hairline pl-3">
          {events.map((event) => (
            <li key={`${event.stage}-${event.timestamp}`} className="flex flex-wrap items-center gap-x-2 gap-y-1">
              {event.status === "running" ? <Loader2 size={11} className="animate-spin text-accent" /> : <CheckCircle2 size={11} className="text-success" />}
              <span className="text-foreground">{t(`trace.stages.${event.stage}`, { defaultValue: event.stage })}</span>
              {chips(event, t as Translate).map((chip) => <span key={chip} className="rounded bg-muted px-1.5 py-0.5 text-muted-foreground">{chip}</span>)}
              {event.timestamp ? <span className="ml-auto tabular-nums text-muted-foreground">{format.time(event.timestamp * 1000)}</span> : null}
            </li>
          ))}
        </ol>
      ) : null}
    </div>
  );
}
