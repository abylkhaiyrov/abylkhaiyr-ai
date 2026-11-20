import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { openStudioConnection } from "@/lib/studio-connection";

class TestSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 3;
  static instances: TestSocket[] = [];
  readyState = TestSocket.CONNECTING;
  onopen?: () => void;
  onclose?: () => void;
  onmessage?: (event: { data: string }) => void;
  send = vi.fn();

  constructor() {
    TestSocket.instances.push(this);
  }

  open() {
    this.readyState = TestSocket.OPEN;
    this.onopen?.();
  }

  close() {
    this.readyState = TestSocket.CLOSED;
    this.onclose?.();
  }
}

function fixture() {
  const callbacks = {
    status: vi.fn(),
    resume: () => ["job"],
    receive: vi.fn(),
    disconnected: vi.fn(),
  };
  const connection = openStudioConnection(callbacks);
  return { connection, callbacks };
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
  TestSocket.instances = [];
  vi.stubGlobal("WebSocket", TestSocket);
  vi.stubGlobal("location", { protocol: "http:", host: "127.0.0.1:5173" });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("Studio connection", () => {
  it("gates generation on acknowledgement and ignores malformed or duplicate events", async () => {
    const { connection, callbacks } = fixture();
    connection.setEnabled(true);
    await Promise.resolve();
    const socket = TestSocket.instances[0];
    socket.open();
    expect(connection.send({ type: "generation.start" })).toBe(false);
    expect(socket.send).toHaveBeenCalledWith(JSON.stringify({
      type: "subscribe", channels: ["studio", "chat"], resume: ["job"],
    }));
    const ack = { data: '{"type":"chat.ready","request_ids":["job"],"seq":1}' };
    socket.onmessage?.({ data: "{invalid" });
    socket.onmessage?.(ack);
    socket.onmessage?.(ack);
    expect(callbacks.receive).toHaveBeenCalledTimes(1);
    expect(connection.send({ type: "generation.start" })).toBe(true);
    connection.close();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("stops after bounded retries and allows a manual retry", async () => {
    const { connection, callbacks } = fixture();
    connection.setEnabled(true);
    await Promise.resolve();
    for (let attempt = 0; attempt < 7; attempt += 1) {
      TestSocket.instances.at(-1)?.close();
      await vi.advanceTimersToNextTimerAsync();
    }
    expect(TestSocket.instances).toHaveLength(7);
    expect(callbacks.status).toHaveBeenLastCalledWith("disconnected");
    connection.reconnect();
    expect(TestSocket.instances).toHaveLength(8);
    connection.close();
    TestSocket.instances.at(-1)?.open();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("waits for the old socket to close when enabling before its handshake finishes", async () => {
    const { connection } = fixture();
    connection.setEnabled(true);
    await Promise.resolve();
    connection.setEnabled(false);
    connection.setEnabled(true);
    await Promise.resolve();
    expect(TestSocket.instances).toHaveLength(1);
    TestSocket.instances[0].open();
    expect(TestSocket.instances).toHaveLength(2);
    connection.close();
    TestSocket.instances[1].open();
  });
});
