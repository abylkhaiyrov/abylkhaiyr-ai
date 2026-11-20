import { API_BASE } from "@/lib/http";
import { parseServerEvent, type ServerEvent } from "@/lib/events";
import type { SocketStatus } from "@/store/ui-store";

const HEARTBEAT_MS = 15_000;
const STALE_MS = 50_000;
const HANDSHAKE_MS = 15_000;
const STABLE_MS = 30_000;
const MAX_RETRIES = 6;

type ConnectionCallbacks = {
  status: (status: SocketStatus) => void;
  resume: () => string[];
  receive: (event: ServerEvent) => void;
  disconnected: () => void;
};

export function openStudioConnection(callbacks: ConnectionCallbacks) {
  let socket: WebSocket | null = null;
  let retryTimer: ReturnType<typeof setTimeout> | undefined;
  let retry = 0;
  let stopped = false;
  let enabled = false;
  let ready = false;
  let restartPending = false;
  let closeCurrent: (() => void) | undefined;

  function connect() {
    if (stopped || !enabled || socket) return;
    restartPending = false;
    callbacks.status(retry ? "reconnecting" : "connecting");
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${location.host}${API_BASE}/ws`);
    socket = ws;
    let sequence = 0;
    let connectedAt: number | null = null;
    let lastSeen = Date.now();
    const handshake = setTimeout(() => ws.close(), HANDSHAKE_MS);
    const heartbeat = setInterval(() => {
      if (Date.now() - lastSeen > STALE_MS) ws.close();
      else if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "ping" }));
    }, HEARTBEAT_MS);
    const clearTimers = () => {
      clearTimeout(handshake);
      clearInterval(heartbeat);
    };
    closeCurrent = () => {
      clearInterval(heartbeat);
      if (ws.readyState === WebSocket.CONNECTING) {
        ws.onopen = () => ws.close();
      } else {
        clearTimeout(handshake);
        ws.close();
      }
    };

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "subscribe", channels: ["studio", "chat"], resume: callbacks.resume() }));
    };
    ws.onmessage = (message) => {
      if (stopped || !enabled || restartPending) return;
      const event = parseServerEvent(message.data);
      if (!event || event.seq <= sequence) return;
      sequence = event.seq;
      lastSeen = Date.now();
      if (event.type === "chat.ready") {
        clearTimeout(handshake);
        ready = true;
        connectedAt = Date.now();
        callbacks.status("connected");
      }
      callbacks.receive(event);
    };
    ws.onclose = () => {
      clearTimers();
      socket = null;
      ready = false;
      closeCurrent = undefined;
      if (stopped || !enabled) return;
      if (restartPending) {
        restartPending = false;
        connect();
        return;
      }
      if (connectedAt !== null && Date.now() - connectedAt >= STABLE_MS) retry = 0;
      callbacks.disconnected();
      if (retry >= MAX_RETRIES) {
        callbacks.status("disconnected");
        return;
      }
      const delay = Math.min(15_000, 500 * 2 ** retry) * (0.8 + Math.random() * 0.4);
      retry += 1;
      callbacks.status("reconnecting");
      retryTimer = setTimeout(connect, delay);
    };
  }

  return {
    setEnabled(next: boolean) {
      if (enabled === next) return;
      enabled = next;
      if (enabled) {
        retry = 0;
        queueMicrotask(connect);
      } else {
        clearTimeout(retryTimer);
        ready = false;
        restartPending = true;
        closeCurrent?.();
        callbacks.status("disconnected");
      }
    },
    send(event: Record<string, unknown>) {
      if (!ready || socket?.readyState !== WebSocket.OPEN) return false;
      socket.send(JSON.stringify(event));
      return true;
    },
    reconnect() {
      if (!enabled) return;
      clearTimeout(retryTimer);
      retry = 0;
      if (socket) {
        if (ready) return;
        restartPending = true;
        closeCurrent?.();
      } else {
        connect();
      }
    },
    close() {
      stopped = true;
      ready = false;
      clearTimeout(retryTimer);
      closeCurrent?.();
    },
  };
}
