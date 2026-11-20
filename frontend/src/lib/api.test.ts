import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("chat responses", () => {
  it("accepts a chat with the fields needed to render messages", async () => {
    const chat = {
      id: "chat",
      title: "Chat",
      messages: [{ id: "message", role: "assistant", content: "Hello", variants: [], reasoning_trace: [] }],
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(chat), {
      headers: { "content-type": "application/json" },
    })));

    await expect(api.chat("chat")).resolves.toEqual(chat);
  });

  it("rejects chat messages missing fields required by the interface", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      id: "chat",
      title: "Chat",
      messages: [{ id: "message", role: "assistant", content: "Hello" }],
    }), { headers: { "content-type": "application/json" } })));

    await expect(api.chat("chat")).rejects.toMatchObject({ code: "invalid_response" });
  });

  it("rejects chat summaries without an update time", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify([
      { id: "chat", title: "Chat", pinned: false },
    ]), { headers: { "content-type": "application/json" } })));

    await expect(api.chats()).rejects.toMatchObject({ code: "invalid_response" });
  });
});
