import { createContext, useCallback, useContext, useEffect, useRef, type ReactNode, createElement } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { Dashboard, Metric, Process, ReasoningStageEvent } from "@/lib/api";
import { useUiStore } from "@/store/ui-store";

export type StudioEvent = {
  type: string;
  request_id?: string;
  chat_id?: string;
  message_id?: string;
  sequence_id?: number;
  text?: string;
  status?: string;
  code?: string;
  detail?: string;
  reason?: string;
  action?: string;
  model_id?: string | null;
  checkpoint_id?: string | null;
  device?: string | null;
  stage?: string;
  stage_status?: string;
  stage_data?: Record<string, unknown>;
  reasoning_events?: ReasoningStageEvent[];
  queue_position?: number;
  queue_reason?: string | null;
  input_tokens?: number;
  output_tokens?: number;
  context_limit?: number;
  elapsed_ms?: number;
  tokens_per_second?: number;
  user?: unknown;
  assistant?: unknown;
  data?: { dashboard: Dashboard; processes: Process[]; metrics: Metric[] };
};

type EventClient = {
  send: (event: Record<string, unknown>) => boolean;
  subscribe: (listener: (event: StudioEvent) => void) => () => void;
};

const EventContext = createContext<EventClient | null>(null);

export function StudioEventsProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const setConnection = useUiStore((state) => state.setConnection);
  const socket = useRef<WebSocket | null>(null);
  const listeners = useRef(new Set<(event: StudioEvent) => void>());
  const activeSnapshots = useRef(new Map<string, StudioEvent>());
  const send = useCallback((event: Record<string, unknown>) => {
    if (socket.current?.readyState !== WebSocket.OPEN) return false;
    socket.current.send(JSON.stringify(event));
    return true;
  }, []);
  const subscribe = useCallback((listener: (event: StudioEvent) => void) => {
    listeners.current.add(listener);
    for (const snapshot of activeSnapshots.current.values()) {
      listener(snapshot);
    }
    return () => {
      listeners.current.delete(listener);
    };
  }, []);

  useEffect(() => {
    let stopped = false;
    let retry = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let heartbeat: ReturnType<typeof setInterval> | undefined;
    function connect() {
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${protocol}//${location.host}/api/ws`);
      socket.current = ws;
      ws.onopen = () => {
        retry = 0;
        setConnection(true, new Date().toISOString());
        ws.send(JSON.stringify({ type: "subscribe", channels: ["studio", "chat"] }));
        void queryClient.invalidateQueries({ queryKey: ["chats"] });
        void queryClient.invalidateQueries({ queryKey: ["chat"] });
        void queryClient.invalidateQueries({ queryKey: ["models"] });
        heartbeat = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "ping" }));
          }
        }, 20_000);
      };
      ws.onmessage = (message) => {
        const event = JSON.parse(String(message.data)) as StudioEvent;
        if (event.type === "studio.snapshot" && event.data) {
          queryClient.setQueryData(["dashboard"], event.data.dashboard);
          queryClient.setQueryData(["system"], event.data.dashboard.system);
          queryClient.setQueryData(["processes"], event.data.processes);
          queryClient.setQueryData(["metrics"], event.data.metrics);
          setConnection(true, new Date().toISOString());
        }
        if (event.type === "generation.snapshot" && event.request_id) {
          if (event.status === "queued" || event.status === "running") {
            activeSnapshots.current.set(event.request_id, event);
          } else {
            activeSnapshots.current.delete(event.request_id);
          }
        }
        if (
          event.request_id &&
          ["generation.completed", "generation.cancelled", "generation.failed"].includes(event.type)
        ) {
          activeSnapshots.current.delete(event.request_id);
        }
        for (const listener of listeners.current) listener(event);
      };
      ws.onclose = () => {
        if (heartbeat) clearInterval(heartbeat);
        setConnection(false, null);
        if (!stopped) {
          timer = setTimeout(connect, Math.min(30_000, 500 * 2 ** retry) * (0.8 + Math.random() * 0.4));
          retry += 1;
        }
      };
      ws.onerror = () => ws.close();
    }
    connect();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
      if (heartbeat) clearInterval(heartbeat);
      socket.current?.close();
      socket.current = null;
    };
  }, [queryClient, setConnection]);

  return createElement(EventContext.Provider, { value: { send, subscribe } }, children);
}

export function useStudioEvents() {
  const value = useContext(EventContext);
  if (!value) throw new Error("StudioEventsProvider is missing");
  return value;
}
