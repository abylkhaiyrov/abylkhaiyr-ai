import { useCallback, useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useStudioEvents } from "@/hooks/use-studio-events";
import type { GenerationSettings } from "@/lib/api";
import { initialGeneration, reduceGeneration, TERMINAL_STATUSES, type ActiveGeneration, type GenerationAction } from "@/lib/generation";

export type StartOptions = {
  chatId: string;
  action: GenerationAction;
  content?: string;
  targetId?: string;
  checkpointId?: string | null;
  settings: GenerationSettings;
};

export type GenerationFailure = { chatId: string; code: string; detail: string | null };

export function useChatGeneration() {
  const { send, subscribe } = useStudioEvents();
  const client = useQueryClient();
  const [active, setActive] = useState<ActiveGeneration | null>(null);
  const [failure, setFailure] = useState<GenerationFailure | null>(null);

  useEffect(
    () =>
      subscribe((event) => {
        setActive((previous) => reduceGeneration(previous, event));
        if (event.type === "generation.failed") setFailure({ chatId: event.chat_id, code: event.code ?? "generation_failed", detail: event.detail ?? null });
        if (event.type === "generation.snapshot" && event.status === "interrupted") {
          setFailure({ chatId: event.chat_id, code: "server_restarted", detail: null });
        }
        const recoveredTerminal = event.type === "generation.snapshot" && event.status !== "queued" && event.status !== "running";
        if (recoveredTerminal || event.type === "generation.completed" || event.type === "generation.cancelled" || event.type === "generation.failed") {
          const queryKey = ["chat", event.chat_id];
          void client.invalidateQueries({ queryKey }, { cancelRefetch: false }).then(() => {
            if (client.getQueryState(queryKey)?.status === "success") {
              setActive((previous) => (previous?.requestId === event.request_id && TERMINAL_STATUSES.has(previous.status) ? null : previous));
            }
          });
        }
        if (event.type === "generation.started") void client.invalidateQueries({ queryKey: ["chat", event.chat_id] }, { cancelRefetch: false });
      }),
    [subscribe, client],
  );

  const start = useCallback(
    (options: StartOptions) => {
      const requestId = crypto.randomUUID();
      const { settings } = options;
      const sent = send({
        type: "generation.start",
        request_id: requestId,
        chat_id: options.chatId,
        action: options.action,
        content: options.content,
        target_message_id: options.targetId,
        checkpoint_id: options.checkpointId ?? undefined,
        mode: settings.mode,
        profile: settings.profile ?? undefined,
        temperature: settings.temperature,
        top_p: settings.top_p,
        top_k: settings.top_k,
        max_tokens: settings.max_tokens,
        repetition_penalty: settings.repetition_penalty,
      });
      if (!sent) {
        setFailure({ chatId: options.chatId, code: "disconnected", detail: null });
        return false;
      }
      setFailure(null);
      setActive(initialGeneration(requestId, options.chatId, options.action, options.targetId ?? null, options.content ?? null));
      return true;
    },
    [send],
  );

  const cancel = useCallback(() => {
    if (active) send({ type: "generation.cancel", request_id: active.requestId });
  }, [active, send]);

  return { active, failure, start, cancel, clearFailure: () => setFailure(null) };
}
