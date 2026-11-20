import type { ReasoningStageEvent } from "@/lib/api";
import type { GenerationEvent } from "@/lib/events";

export type GenerationAction = "send" | "edit" | "regenerate" | "continue";

export type ActiveGeneration = {
  requestId: string;
  chatId: string;
  action: GenerationAction;
  targetId: string | null;
  pendingContent: string | null;
  messageId: string | null;
  text: string;
  sequence: number;
  status: "queued" | "running" | "completed" | "cancelled" | "failed";
  queuePosition: number;
  queueReason: string | null;
  modelId: string | null;
  checkpointId: string | null;
  device: string | null;
  inputTokens: number | null;
  contextLimit: number | null;
  outputTokens: number | null;
  tokensPerSecond: number | null;
  firstTokenMs: number | null;
  stages: ReasoningStageEvent[];
  errorCode: string | null;
  errorDetail: string | null;
};

export const TERMINAL_STATUSES = new Set(["completed", "cancelled", "failed"]);

export function initialGeneration(requestId: string, chatId: string, action: GenerationAction, targetId: string | null, pendingContent: string | null): ActiveGeneration {
  return {
    requestId,
    chatId,
    action,
    targetId,
    pendingContent,
    messageId: null,
    text: "",
    sequence: 0,
    status: "queued",
    queuePosition: 0,
    queueReason: null,
    modelId: null,
    checkpointId: null,
    device: null,
    inputTokens: null,
    contextLimit: null,
    outputTokens: null,
    tokensPerSecond: null,
    firstTokenMs: null,
    stages: [],
    errorCode: null,
    errorDetail: null,
  };
}

export function isActive(generation: ActiveGeneration | null): generation is ActiveGeneration {
  return generation !== null && !TERMINAL_STATUSES.has(generation.status);
}

export function reduceGeneration(previous: ActiveGeneration | null, event: GenerationEvent): ActiveGeneration | null {
  if (event.type === "generation.snapshot") {
    const base = previous?.requestId === event.request_id ? previous : initialGeneration(event.request_id, event.chat_id, "send", null, null);
    return {
      ...base,
      messageId: event.message_id ?? base.messageId,
      text: event.text ?? base.text,
      sequence: event.sequence_id,
      status: (event.status as ActiveGeneration["status"]) ?? base.status,
      queuePosition: event.queue_position ?? 0,
      queueReason: event.queue_reason ?? null,
      modelId: event.model_id ?? base.modelId,
      checkpointId: event.checkpoint_id ?? base.checkpointId,
      device: event.device ?? base.device,
      stages: event.reasoning_events ?? base.stages,
    };
  }
  if (previous === null || previous.requestId !== event.request_id || event.sequence_id <= previous.sequence) return previous;
  const next: ActiveGeneration = {
    ...previous,
    sequence: event.sequence_id,
    messageId: event.message_id ?? previous.messageId,
    modelId: event.model_id ?? previous.modelId,
    checkpointId: event.checkpoint_id ?? previous.checkpointId,
    device: event.device ?? previous.device,
    inputTokens: event.input_tokens ?? previous.inputTokens,
    contextLimit: event.context_limit ?? previous.contextLimit,
    outputTokens: event.output_tokens ?? previous.outputTokens,
    tokensPerSecond: event.tokens_per_second ?? previous.tokensPerSecond,
    firstTokenMs: event.first_token_ms ?? previous.firstTokenMs,
  };
  switch (event.type) {
    case "generation.queued":
      return { ...next, status: "queued", queuePosition: event.queue_position ?? 0, queueReason: event.queue_reason ?? null };
    case "generation.started":
      return { ...next, status: "running", queuePosition: 0, text: event.text ?? next.text };
    case "generation.stage":
      return { ...next, stages: event.reasoning_events ?? next.stages };
    case "generation.delta":
      return { ...next, status: "running", text: next.text + (event.text ?? "") };
    case "generation.completed":
    case "generation.cancelled":
      return { ...next, status: event.type === "generation.completed" ? "completed" : "cancelled", text: event.text ?? next.text, stages: event.reasoning_events ?? next.stages };
    case "generation.failed":
      return { ...next, status: "failed", errorCode: event.code ?? "generation_failed", errorDetail: event.detail ?? null, text: event.text ?? next.text };
    default:
      return next;
  }
}
