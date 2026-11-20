import { describe, expect, it } from "vitest";
import type { ChatSummary } from "@/lib/api";
import { buildRows } from "@/lib/chat-groups";

const chat = (id: string, updated: string, pinned = false): ChatSummary => ({
  id, title: id, model_name: null, checkpoint: null, system_prompt: "", pinned, generation_settings: {}, created_at: updated, updated_at: updated,
});

describe("buildRows", () => {
  it("groups chats by date with pinned first", () => {
    const now = new Date("2026-11-20T12:00:00Z").getTime();
    const rows = buildRows([chat("a", "2026-11-20T08:00:00"), chat("b", "2026-11-19T08:00:00"), chat("c", "2026-11-01T08:00:00"), chat("d", "2026-11-01T08:00:00", true)], now);
    const headers = rows.filter((row) => row.type === "header").map((row) => (row.type === "header" ? row.group : ""));
    expect(headers).toEqual(["pinned", "today", "yesterday", "older"]);
  });
});
