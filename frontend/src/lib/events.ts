import { isRecord } from "@/lib/schema";
import type { AppNotification, ChatMessage, Metric, ModelRegistry, ReasoningStageEvent, RuntimeState, SystemSnapshot } from "@/lib/api";

type Envelope = { seq: number; ts: number };
type GenerationBase = {
  request_id: string;
  chat_id: string;
  message_id?: string | null;
  sequence_id: number;
  model_id?: string | null;
  checkpoint_id?: string | null;
  device?: string | null;
};

export type GenerationEvent = Envelope &
  GenerationBase & {
    type:
      | "generation.queued"
      | "generation.started"
      | "generation.prefill"
      | "generation.stage"
      | "generation.delta"
      | "generation.completed"
      | "generation.cancelled"
      | "generation.failed"
      | "generation.snapshot";
    status?: string;
    task_status?: string;
    text?: string;
    code?: string;
    detail?: string;
    action?: string;
    queue_position?: number;
    queue_reason?: string | null;
    input_tokens?: number;
    output_tokens?: number;
    context_limit?: number;
    elapsed_ms?: number;
    first_token_ms?: number | null;
    tokens_per_second?: number;
    user?: ChatMessage | null;
    assistant?: ChatMessage;
    generating_seconds?: number;
    stage?: string;
    stage_status?: string;
    stage_data?: Record<string, unknown>;
    reasoning_events?: ReasoningStageEvent[];
  };

export type ServerEvent =
  | (Envelope & { type: "hello"; connection_id: string })
  | (Envelope & { type: "heartbeat" | "pong" })
  | (Envelope & { type: "error"; code: string })
  | (Envelope & { type: "chat.ready"; request_ids: string[] })
  | (Envelope & { type: "runtime.update"; data: RuntimeState })
  | (Envelope & { type: "system.update"; data: SystemSnapshot })
  | (Envelope & { type: "models.update"; data: ModelRegistry })
  | (Envelope & { type: "metrics.append"; run: string; rows: Metric[]; reset: boolean })
  | (Envelope & { type: "notification.created"; notification: AppNotification })
  | (Envelope & { type: "notification.read" | "notification.deleted"; ids: string[] })
  | (Envelope & { type: "notification.count.updated"; unread: number })
  | (Envelope & { type: "session.revoked" })
  | GenerationEvent;

export function isGenerationEvent(event: ServerEvent): event is GenerationEvent {
  return event.type.startsWith("generation.");
}

export function parseServerEvent(data: unknown): ServerEvent | null {
  if (typeof data !== "string") return null;
  try {
    const event: unknown = JSON.parse(data);
    if (!isRecord(event) || typeof event.type !== "string" || !Number.isSafeInteger(event.seq)) return null;
    if (event.type.startsWith("generation.")) {
      if (typeof event.request_id !== "string" || typeof event.chat_id !== "string" || !Number.isSafeInteger(event.sequence_id)) return null;
    }
    if (event.type === "chat.ready" && (!Array.isArray(event.request_ids) || !event.request_ids.every((id) => typeof id === "string"))) return null;
    if (event.type === "metrics.append" && (!Array.isArray(event.rows) || !event.rows.every((row) => isRecord(row) && typeof row.step === "number") || typeof event.run !== "string")) return null;
    if (event.type.endsWith(".update") && !isRecord(event.data)) return null;
    if (event.type === "notification.created" && (!isRecord(event.notification) || typeof event.notification.id !== "string")) return null;
    if (event.type === "notification.count.updated" && !Number.isSafeInteger(event.unread)) return null;
    if ((event.type === "notification.read" || event.type === "notification.deleted") && !Array.isArray(event.ids)) return null;
    return event as ServerEvent;
  } catch {
    return null;
  }
}
