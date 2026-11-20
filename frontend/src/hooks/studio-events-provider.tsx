import { useCallback, useEffect, useMemo, useRef, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { isGenerationEvent, type GenerationEvent, type ServerEvent } from "@/lib/events";
import type { AppNotification, Metric, SystemSnapshot } from "@/lib/api";
import { EventContext } from "@/hooks/use-studio-events";
import i18n from "@/i18n";
import { openStudioConnection } from "@/lib/studio-connection";
import { useToastStore } from "@/store/toast-store";
import { useUiStore } from "@/store/ui-store";

const MAX_HISTORY = 720;
const MAX_METRIC_ROWS = 3000;
const MAX_RECOVERABLE = 8;
const MAX_TERMINALS = 64;
const TERMINAL_TYPES = new Set(["generation.completed", "generation.cancelled", "generation.failed"]);
const LIVE_QUERY_KEYS = new Set(["me", "runtime", "models", "system"]);

function isTerminal(event: GenerationEvent) {
  return TERMINAL_TYPES.has(event.type) ||
    (event.type === "generation.snapshot" && event.status !== "queued" && event.status !== "running");
}

export function StudioEventsProvider({ children, enabled }: { children: ReactNode; enabled: boolean }) {
  const queryClient = useQueryClient();
  const setSocketStatus = useUiStore((state) => state.setSocketStatus);
  const connection = useRef<ReturnType<typeof openStudioConnection> | null>(null);
  const listeners = useRef(new Set<(event: GenerationEvent) => void>());
  const snapshots = useRef(new Map<string, GenerationEvent>());
  const terminals = useRef(new Set<string>());

  const send = useCallback((event: Record<string, unknown>) => {
    if (event.type !== "generation.start") return connection.current?.send(event) ?? false;
    if (typeof event.request_id !== "string" || typeof event.chat_id !== "string") return false;
    if (snapshots.current.size >= MAX_RECOVERABLE || terminals.current.has(event.request_id)) return false;
    for (const pending of snapshots.current.values()) {
      if (pending.request_id === event.request_id || pending.chat_id === event.chat_id) return false;
    }
    if (!connection.current?.send(event)) return false;
    snapshots.current.set(event.request_id, {
      type: "generation.snapshot",
      request_id: event.request_id,
      chat_id: event.chat_id,
      status: "queued",
      sequence_id: 0,
      seq: 0,
      ts: 0,
    });
    return true;
  }, []);

  const subscribe = useCallback((listener: (event: GenerationEvent) => void) => {
    listeners.current.add(listener);
    snapshots.current.forEach(listener);
    return () => {
      listeners.current.delete(listener);
    };
  }, []);

  useEffect(() => {
    function applyData(event: ServerEvent) {
      if (event.type === "runtime.update") {
        queryClient.setQueryData(["runtime"], event.data);
      } else if (event.type === "models.update") {
        queryClient.setQueryData(["models"], event.data);
        void queryClient.invalidateQueries({ queryKey: ["evaluation"] }, { cancelRefetch: false });
        void queryClient.invalidateQueries({ queryKey: ["pipelines"] }, { cancelRefetch: false });
      } else if (event.type === "system.update") {
        queryClient.setQueryData(["system"], event.data);
        queryClient.setQueryData<SystemSnapshot[]>(["system-history"], (previous) => {
          const history = previous ?? [];
          if (history.at(-1)?.recorded_at === event.data.recorded_at) return history;
          return [...history, event.data].slice(-MAX_HISTORY);
        });
      } else if (event.type === "notification.count.updated") {
        queryClient.setQueryData(["notifications", "count"], { unread: event.unread });
      } else if (event.type === "notification.read" || event.type === "notification.deleted") {
        void queryClient.invalidateQueries({ queryKey: ["notifications", "list"] });
      } else if (event.type === "notification.created") {
        void queryClient.invalidateQueries({ queryKey: ["notifications", "list"] });
        announce(event.notification);
      } else if (event.type === "session.revoked") {
        useToastStore.getState().push("info", i18n.t("auth.sessionRevoked"));
        void queryClient.invalidateQueries({ queryKey: ["me"], exact: true });
      } else if (event.type === "metrics.append") {
        queryClient.setQueryData<Metric[]>(["metrics", event.run], (previous) => {
          const byStep = new Map((previous ?? []).map((row) => [row.step, row]));
          event.rows.forEach((row) => byStep.set(row.step, { ...byStep.get(row.step), ...row }));
          return [...byStep.values()].sort((a, b) => a.step - b.step).slice(-MAX_METRIC_ROWS);
        });
      }
    }

    function announce(notification: AppNotification) {
      const params = new URLSearchParams(location.search);
      const viewing = location.pathname === "/chat" && params.get("c") === notification.chat_id && document.visibilityState === "visible";
      if (viewing || notification.type === "GENERATION_CANCELLED") return;
      const key = `notification.${notification.type}`;
      useToastStore.getState().push(
        notification.type === "GENERATION_FAILED" ? "error" : "info",
        i18n.exists(key) ? i18n.t(key) : i18n.t("notification.generic"),
        notification.chat_id ? { label: i18n.t("notification.open"), href: `/chat?c=${notification.chat_id}` } : undefined,
      );
    }

    function applyGeneration(event: GenerationEvent) {
      if (terminals.current.has(event.request_id)) return;
      const previous = snapshots.current.get(event.request_id);
      if (previous && event.sequence_id <= previous.sequence_id && event.type !== "generation.snapshot") return;
      if (previous && event.sequence_id < previous.sequence_id && !isTerminal(event)) return;
      if (isTerminal(event)) {
        snapshots.current.delete(event.request_id);
        terminals.current.add(event.request_id);
        if (terminals.current.size > MAX_TERMINALS) {
          const oldest = terminals.current.values().next().value;
          if (oldest !== undefined) terminals.current.delete(oldest);
        }
        void queryClient.invalidateQueries({ queryKey: ["chats"] }, { cancelRefetch: false });
      } else {
        const text = event.type === "generation.delta"
          ? (previous?.text ?? "") + (event.text ?? "")
          : event.text ?? previous?.text;
        snapshots.current.set(event.request_id, {
          ...previous,
          ...event,
          type: "generation.snapshot",
          text,
          status: event.status ?? (event.type === "generation.queued" ? "queued" : event.type === "generation.started" || event.type === "generation.delta" ? "running" : previous?.status ?? "running"),
          message_id: event.message_id ?? previous?.message_id,
        });
      }
      if (event.type === "generation.started") {
        void queryClient.invalidateQueries({ queryKey: ["chats"] }, { cancelRefetch: false });
      }
      listeners.current.forEach((listener) => listener(event));
    }

    let connectedBefore = false;
    const current = openStudioConnection({
      status: setSocketStatus,
      resume: () => [...snapshots.current.keys()].slice(0, MAX_RECOVERABLE),
      disconnected: () => {
        void queryClient.invalidateQueries({ queryKey: ["me"], exact: true }, { cancelRefetch: false });
      },
      receive: (event) => {
        if (event.type === "chat.ready") {
          const recovered = new Set(event.request_ids);
          snapshots.current.forEach((pending, id) => {
            if (!recovered.has(id)) {
              applyGeneration({ ...pending, type: "generation.failed", code: "server_restarted", sequence_id: pending.sequence_id + 1 });
            }
          });
          if (connectedBefore) {
            void queryClient.invalidateQueries({
              predicate: (query) => !LIVE_QUERY_KEYS.has(String(query.queryKey[0])),
            }, { cancelRefetch: false });
          }
          connectedBefore = true;
        } else if (isGenerationEvent(event)) {
          applyGeneration(event);
        } else {
          applyData(event);
        }
      },
    });
    connection.current = current;
    return () => {
      current.close();
      connection.current = null;
      setSocketStatus("disconnected");
    };
  }, [queryClient, setSocketStatus]);

  useEffect(() => {
    connection.current?.setEnabled(enabled);
  }, [enabled]);

  const reconnect = useCallback(() => connection.current?.reconnect(), []);
  const value = useMemo(() => ({ send, subscribe, reconnect }), [send, subscribe, reconnect]);
  return <EventContext.Provider value={value}>{children}</EventContext.Provider>;
}
