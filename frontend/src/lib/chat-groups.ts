import type { ChatSummary } from "@/lib/api";

export type ChatGroup = "pinned" | "today" | "yesterday" | "week" | "older";
export type ChatRow = { type: "header"; group: ChatGroup } | { type: "chat"; chat: ChatSummary };

const DAY = 86_400_000;
const ORDER: ChatGroup[] = ["pinned", "today", "yesterday", "week", "older"];

export function groupOf(chat: ChatSummary, now: number): ChatGroup {
  if (chat.pinned) return "pinned";
  const midnight = new Date(now).setHours(0, 0, 0, 0);
  const updated = new Date(chat.updated_at.endsWith("Z") ? chat.updated_at : `${chat.updated_at}Z`).getTime();
  if (updated >= midnight) return "today";
  if (updated >= midnight - DAY) return "yesterday";
  if (updated >= midnight - 7 * DAY) return "week";
  return "older";
}

export function buildRows(chats: ChatSummary[], now = Date.now()): ChatRow[] {
  const buckets = new Map<ChatGroup, ChatSummary[]>();
  for (const chat of chats) {
    const group = groupOf(chat, now);
    buckets.set(group, [...(buckets.get(group) ?? []), chat]);
  }
  return ORDER.flatMap((group) => {
    const items = buckets.get(group);
    return items ? [{ type: "header" as const, group }, ...items.map((chat) => ({ type: "chat" as const, chat }))] : [];
  });
}
