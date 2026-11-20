import type { Metric, ModelRegistry, ReasoningStageEvent, RuntimeState, SystemSnapshot } from "@/lib/api";

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
    stage?: string;
    stage_status?: string;
    stage_data?: Record<string, unknown>;
    reasoning_events?: ReasoningStageEvent[];
  };

export type ServerEvent =
  | (Envelope & { type: "hello"; connection_id: string })
  | (Envelope & { type: "heartbeat" | "pong" })
  | (Envelope & { type: "error"; code: string })
  | (Envelope & { type: "runtime.update"; data: RuntimeState })
  | (Envelope & { type: "system.update"; data: SystemSnapshot })
  | (Envelope & { type: "models.update"; data: ModelRegistry })
  | (Envelope & { type: "metrics.append"; run: string; rows: Metric[]; reset: boolean })
  | GenerationEvent;

export function isGenerationEvent(event: ServerEvent): event is GenerationEvent {
  return event.type.startsWith("generation.");
}
