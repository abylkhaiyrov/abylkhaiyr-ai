import { promises as fs } from "node:fs";
import { expect, test } from "@playwright/test";
import { ADMIN_PASSWORD, adminApi, signIn } from "./support";

test.skip(!process.env.E2E_BACKEND, "requires a running Studio backend with a loaded model");
test.describe.configure({ mode: "serial" });

const queries = [
  "Что ты умеешь делать?",
  "Почему море солёное?",
  "Меня зовут Тимур. Запомни это.",
  "Как меня зовут?",
];

type Message = { role: string; content: string; status: string; checkpoint_id: string | null; input_tokens: number | null; generation_params: { profile?: string; max_tokens?: number } };

test("real queries through the Studio UI are answered by the loaded checkpoint", async ({ page, request, baseURL }, info) => {
  const admin = await adminApi(request, baseURL ?? "");
  const catalog = await admin.get<{ items: { id: string; available: boolean }[] }>("/models/catalog");
  const target = catalog.items.find((item) => item.available);
  expect(target).toBeTruthy();
  await admin.put("/models/allowed", { ids: [target?.id] });
  await admin.put("/inference/profiles/allowed", { ids: ["lite", "medium", "high", "very_high"] });
  await signIn(page, "admin", ADMIN_PASSWORD);
  await page.goto("/chat");
  const profile = page.getByLabel("Inference profile");
  await expect(profile).toBeVisible();
  await profile.selectOption("medium");
  for (const query of queries) {
    await page.getByLabel("Write a message…").fill(query);
    await page.getByRole("button", { name: "Send" }).click();
    await expect(page.getByTestId("messages").getByText(query, { exact: true }).first()).toBeVisible();
    await expect(page.getByRole("button", { name: "Send" })).toBeVisible({ timeout: 90_000 });
  }
  const chats = (await (await page.request.get("/api/chats")).json()) as { id: string }[];
  const chat = (await (await page.request.get(`/api/chats/${chats[0].id}`)).json()) as { messages: Message[] };
  const replies = chat.messages.filter((item) => item.role === "assistant");
  expect(replies).toHaveLength(queries.length);
  for (const reply of replies) {
    expect(reply.status).toBe("completed");
    expect(reply.content.trim().length).toBeGreaterThan(0);
    expect(reply.checkpoint_id).toBeTruthy();
    expect(reply.generation_params.profile).toBe("medium");
  }
  expect(replies[3].input_tokens ?? 0).toBeGreaterThan(replies[0].input_tokens ?? 0);
  const record = queries.map((query, index) => ({ query, answer: replies[index].content, checkpoint: replies[index].checkpoint_id, input_tokens: replies[index].input_tokens }));
  await fs.writeFile(info.outputPath("real-dialogue.json"), JSON.stringify(record, null, 1));
  await page.screenshot({ path: info.outputPath("real-dialogue.png"), fullPage: true });
});
