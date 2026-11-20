import type { GenerationSettings } from "@/lib/api";

export const DEFAULT_SETTINGS: Required<GenerationSettings> = { mode: "standard", temperature: 0.7, top_p: 0.9, top_k: 40, max_tokens: 192, repetition_penalty: 1.1, context_messages: 0 };
