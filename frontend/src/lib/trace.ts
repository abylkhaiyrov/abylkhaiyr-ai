import type { ReasoningStageEvent } from "@/lib/api";

export type Translate = (key: string, options?: Record<string, unknown>) => string;

export function stageLabel(event: ReasoningStageEvent, t: Translate): string {
  const check = event.stage === "verification" && typeof event.data?.check === "string" ? `verification_${event.data.check}` : event.stage;
  return t(`trace.stages.${check}`, { defaultValue: event.stage });
}

export function doneLabel(events: ReasoningStageEvent[], t: Translate): string {
  const phases: string[] = [];
  for (const event of events) if (event.phase && !phases.includes(event.phase)) phases.push(event.phase);
  if (phases.length === 0) return t("trace.steps", { count: events.length });
  return t("trace.done", { summary: phases.map((phase) => t(`trace.phases.${phase}`, { defaultValue: phase })).join(" · ") });
}
