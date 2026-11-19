import { useState } from "react";
import { useTranslation } from "react-i18next";
import { CheckCircle2, ChevronDown, ChevronRight, Cpu, Loader2, Sparkles } from "lucide-react";
import type { ReasoningStageEvent } from "@/lib/api";

type ReasoningPanelProps = {
  events: ReasoningStageEvent[];
  active?: boolean;
};

export function ReasoningPanel({ events, active = false }: ReasoningPanelProps) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(active);
  if (events.length === 0) return null;

  const isRunning = active || events.some((item) => item.status === "running");
  const isVerified = events.some(
    (item) =>
      (item.stage === "verification" || item.stage === "answer_formation") &&
      item.data?.verified === true,
  );

  function renderDetails(event: ReasoningStageEvent): string[] {
    const data = event.data ?? {};
    const chips: string[] = [];
    if (typeof data.kind === "string") {
      chips.push(t(`chat.reasoning.kinds.${data.kind}`, { defaultValue: data.kind }));
    }
    if (typeof data.input_tokens === "number" && typeof data.context_limit === "number") {
      chips.push(`${data.input_tokens}/${data.context_limit}`);
    }
    if (typeof data.attempt === "number" && typeof data.total === "number") {
      chips.push(`#${data.attempt}/${data.total}`);
    }
    if (typeof data.annotations_total === "number" && data.annotations_total > 0) {
      chips.push(`${String(data.annotations_correct ?? 0)}/${data.annotations_total}`);
    }
    if (typeof data.tool === "string") {
      chips.push(data.tool);
    }
    if (typeof data.tokens_used === "number") {
      chips.push(t("chat.tokensCount", { count: data.tokens_used }));
    }
    if (typeof data.output_tokens === "number") {
      chips.push(t("chat.tokensCount", { count: data.output_tokens }));
    }
    if (typeof data.score === "number") {
      chips.push(`${Math.round(data.score * 100)}%`);
    }
    if (data.verified === true) {
      chips.push(t("chat.reasoning.verified"));
    }
    if (typeof data.device === "string") {
      chips.push(data.device.toUpperCase());
    }
    return chips;
  }

  return (
    <div className="mb-3 overflow-hidden rounded-xl border bg-muted/40 text-xs transition-all">
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left transition-colors hover:bg-muted/70"
      >
        <span className="flex items-center gap-2 font-medium text-foreground">
          {isRunning ? (
            <Loader2 size={14} className="animate-spin text-primary" />
          ) : (
            <Sparkles size={14} className="text-primary" />
          )}
          <span>{isRunning ? t("chat.reasoning.running") : t("chat.reasoning.title")}</span>
          <span className="rounded-full bg-background/80 px-2 py-0.5 text-[11px] text-muted-foreground">
            {t("chat.reasoning.stepsCount", { count: events.length })}
          </span>
          {isVerified ? (
            <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[11px] font-medium text-emerald-600 dark:text-emerald-400">
              {t("chat.reasoning.verified")}
            </span>
          ) : null}
        </span>
        {open ? <ChevronDown size={14} className="text-muted-foreground" /> : <ChevronRight size={14} className="text-muted-foreground" />}
      </button>
      {open ? (
        <ol className="divide-y divide-border/60 border-t bg-background/60 px-3 py-1">
          {events.map((event, index) => {
            const chips = renderDetails(event);
            const runningStep = event.status === "running";
            return (
              <li
                key={`${event.stage}-${index}`}
                className="flex flex-wrap items-center justify-between gap-2 py-2"
              >
                <span className="flex items-center gap-2">
                  {runningStep ? (
                    <Loader2 size={13} className="animate-spin text-primary" />
                  ) : event.stage === "model_loading" ? (
                    <Cpu size={13} className="text-primary" />
                  ) : (
                    <CheckCircle2 size={13} className="text-emerald-500" />
                  )}
                  <span className={runningStep ? "font-medium text-foreground" : "text-muted-foreground"}>
                    {t(`chat.reasoning.stages.${event.stage}`, { defaultValue: event.stage })}
                  </span>
                </span>
                {chips.length > 0 ? (
                  <span className="flex flex-wrap gap-1">
                    {chips.map((chip, chipIndex) => (
                      <span
                        key={`${chip}-${chipIndex}`}
                        className="rounded-md border bg-muted/60 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground"
                      >
                        {chip}
                      </span>
                    ))}
                  </span>
                ) : null}
              </li>
            );
          })}
        </ol>
      ) : null}
    </div>
  );
}
