import { useCallback, useEffect, useMemo, useRef, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { isGenerationEvent, type GenerationEvent, type ServerEvent } from "@/lib/events";
import type { Metric, SystemSnapshot } from "@/lib/api";
import { EventContext } from "@/hooks/use-studio-events";
import { useUiStore } from "@/store/ui-store";

const HEARTBEAT_MS = 20_000;
const STALE_MS = 50_000;
const MAX_HISTORY = 720;
const MAX_METRIC_ROWS = 3000;
const ACTIVE_STATUSES = new Set(["queued", "running"]);
const TERMINAL_TYPES = new Set(["generation.completed", "generation.cancelled", "generation.failed"]);

export function StudioEventsProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const setConnection = useUiStore((state) => state.setConnection);
  const socket = useRef<WebSocket | null>(null);
  const listeners = useRef(new Set<(event: GenerationEvent) => void>());
  const snapshots = useRef(new Map<string, GenerationEvent>());
  const lastSequence = useRef(new Map<string, number>());

  const send = useCallback((event: Record<string, unknown>) => {
    if (socket.current?.readyState !== WebSocket.OPEN) return false;
    socket.current.send(JSON.stringify(event));
    return true;
  }, []);

  const subscribe = useCallback((listener: (event: GenerationEvent) => void) => {
    listeners.current.add(listener);
    snapshots.current.forEach((snapshot) => listener(snapshot));
    return () => {
      listeners.current.delete(listener);
    };
  }, []);

  useEffect(() => {
    let stopped = false;
    let retry = 0;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    let heartbeat: ReturnType<typeof setInterval> | undefined;
    let lastSeen = Date.now();
    let connectionSeq = 0;

    function applyData(event: ServerEvent) {
      if (event.type === "runtime.update") queryClient.setQueryData(["runtime"], event.data);
      else if (event.type === "models.update") queryClient.setQueryData(["models"], event.data);
      else if (event.type === "system.update") {
        queryClient.setQueryData(["system"], event.data);
        queryClient.setQueryData<SystemSnapshot[]>(["system-history"], (previous) =>
          [...(previous ?? []), event.data].slice(-MAX_HISTORY),
        );
      } else if (event.type === "metrics.append") {
        queryClient.setQueryData<Metric[]>(["metrics", event.run], (previous) => {
          const byStep = new Map((previous ?? []).map((row) => [row.step, row]));
          event.rows.forEach((row) => byStep.set(row.step, { ...byStep.get(row.step), ...row }));
          return [...byStep.values()].sort((a, b) => a.step - b.step).slice(-MAX_METRIC_ROWS);
        });
      }
    }

    function applyGeneration(event: GenerationEvent) {
      const previous = lastSequence.current.get(event.request_id) ?? 0;
      if (event.type !== "generation.snapshot" && event.sequence_id <= previous) return;
      lastSequence.current.set(event.request_id, event.sequence_id);
      if (event.type === "generation.snapshot") {
        if (event.status && ACTIVE_STATUSES.has(event.status)) snapshots.current.set(event.request_id, event);
        else snapshots.current.delete(event.request_id);
      }
      if (TERMINAL_TYPES.has(event.type)) {
        snapshots.current.delete(event.request_id);
        void queryClient.invalidateQueries({ queryKey: ["chats"] });
      }
      if (event.type === "generation.started") void queryClient.invalidateQueries({ queryKey: ["chats"] });
      listeners.current.forEach((listener) => listener(event));
    }

    function connect() {
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${protocol}//${location.host}/api/ws`);
      socket.current = ws;
      connectionSeq = 0;
      ws.onopen = () => {
        retry = 0;
        lastSeen = Date.now();
        setConnection(true);
        ws.send(JSON.stringify({ type: "subscribe", channels: ["studio", "chat"] }));
        void queryClient.invalidateQueries({ queryKey: ["chats"] });
        void queryClient.invalidateQueries({ queryKey: ["chat"] });
        heartbeat = setInterval(() => {
          if (Date.now() - lastSeen > STALE_MS) ws.close();
          else if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "ping" }));
        }, HEARTBEAT_MS);
      };
      ws.onmessage = (message) => {
        const event = JSON.parse(String(message.data)) as ServerEvent;
        if (event.seq <= connectionSeq) return;
        connectionSeq = event.seq;
        lastSeen = Date.now();
        setConnection(true);
        if (isGenerationEvent(event)) applyGeneration(event);
        else applyData(event);
      };
      ws.onclose = (closeEvent) => {
        if (heartbeat) clearInterval(heartbeat);
        setConnection(false);
        if (closeEvent.code === 1008) void queryClient.invalidateQueries({ queryKey: ["me"] });
        if (!stopped) {
          retryTimer = setTimeout(connect, Math.min(15_000, 500 * 2 ** retry) * (0.8 + Math.random() * 0.4));
          retry += 1;
        }
      };
      ws.onerror = () => ws.close();
    }

    connect();
    return () => {
      stopped = true;
      if (retryTimer) clearTimeout(retryTimer);
      if (heartbeat) clearInterval(heartbeat);
      const ws = socket.current;
      socket.current = null;
      setConnection(false);
      if (ws) {
        ws.onmessage = null;
        ws.onclose = null;
        ws.onerror = null;
        if (ws.readyState === WebSocket.CONNECTING) {
          ws.onopen = () => ws.close();
        } else {
          ws.onopen = null;
          ws.close();
        }
      }
    };
  }, [queryClient, setConnection]);

  const value = useMemo(() => ({ send, subscribe }), [send, subscribe]);
  return <EventContext.Provider value={value}>{children}</EventContext.Provider>;
}
