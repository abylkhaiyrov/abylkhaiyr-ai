import { execFileSync } from "node:child_process";
import { expect, test } from "@playwright/test";
import { signIn } from "./support";

const script = process.env.E2E_RESTART_SCRIPT;
test.skip(!process.env.E2E_BACKEND || !script, "requires a running Studio backend and E2E_RESTART_SCRIPT");

test("the interface recovers after the backend restarts", async ({ page }, info) => {
  test.setTimeout(180_000);
  await signIn(page);
  await page.goto("/");
  await expect(page.getByRole("status").filter({ hasText: "Live updates connected" })).toBeVisible();
  const created = await page.request.post("/api/chats", {
    data: { title: "Restart probe" },
    headers: { "X-CSRF-Token": (await page.context().cookies()).find((cookie) => cookie.name === "abylkhaiyr_csrf")?.value ?? "", origin: new URL(page.url()).origin },
  });
  expect(created.ok()).toBeTruthy();
  const chat = (await created.json()) as { id: string };
  await page.goto(`/chat?c=${chat.id}`);
  await expect(page.getByLabel("Write a message…")).toBeVisible();
  const failures: string[] = [];
  page.on("pageerror", (error) => failures.push(error.message));

  const run = (action: "stop" | "start") => execFileSync(script ?? "", [action], { timeout: 120_000 });
  run("stop");
  try {
    await expect(page.getByRole("status").filter({ hasText: /reconnecting|paused|unavailable/i })).toBeVisible({ timeout: 60_000 });
    await expect(page.locator("main")).toBeVisible();
    await page.screenshot({ path: info.outputPath("backend-down.png") });
  } finally {
    run("start");
  }

  await expect(page.getByRole("status").filter({ hasText: "Live updates connected" })).toBeVisible({ timeout: 90_000 });
  await expect(page.getByLabel("Write a message…")).toBeVisible();
  expect(failures).toEqual([]);
  const listed = await (await page.request.get("/api/chats")).json();
  expect(listed.some((item: { id: string }) => item.id === chat.id)).toBeTruthy();
  await page.screenshot({ path: info.outputPath("backend-recovered.png") });
});
