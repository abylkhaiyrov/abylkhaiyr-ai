import { expect, test } from "@playwright/test";
import { adminApi, signIn } from "./support";

test.skip(!process.env.E2E_BACKEND, "requires a running Studio backend");
test.describe.configure({ mode: "serial" });

type Catalog = { items: { id: string; display_name: string; available: boolean }[] };
const stamp = Date.now().toString(36);
const user = { username: `e2e${stamp}`, email: `e2e${stamp}@example.com`, password: "e2e-user-password-1" };

test("a new user sees an empty state until an admin enables a model", async ({ page, request, baseURL }, info) => {
  const admin = await adminApi(request, baseURL ?? "");
  await admin.put("/models/allowed", { ids: [] });
  await page.addInitScript(() => localStorage.setItem("abylkhaiyr-language", "en"));
  await page.goto("/register");
  await page.getByLabel("First name").fill("E2E");
  await page.getByLabel("Last name").fill("User");
  await page.getByLabel("Email").fill(user.email);
  await page.getByLabel("Username").fill(user.username);
  await page.getByLabel("Password", { exact: true }).fill(user.password);
  await page.getByLabel("Repeat password").fill(user.password);
  await page.getByRole("button", { name: /create account/i }).click();
  await expect(page.getByText("No models are available")).toBeVisible();
  await expect(page.getByLabel("Write a message…")).toHaveCount(0);
  await page.screenshot({ path: info.outputPath("user-empty.png") });
});

test("an allowed model is selectable, the profile is saved and the reply streams", async ({ page, request, baseURL }, info) => {
  const admin = await adminApi(request, baseURL ?? "");
  const catalog = await admin.get<Catalog>("/models/catalog");
  const target = catalog.items.find((item) => item.available);
  expect(target).toBeTruthy();
  await admin.put("/models/allowed", { ids: [target?.id] });
  await admin.put("/inference/profiles/allowed", { ids: ["lite", "medium", "high", "very_high"] });
  await signIn(page, user.username, user.password);
  await page.goto("/");
  await expect(page.getByRole("button", { name: new RegExp(target?.display_name ?? "") })).toBeVisible();
  const profile = page.getByLabel("Inference profile");
  await expect(profile).toBeVisible();
  await expect(profile.locator("option", { hasText: "Very High" })).toBeDisabled();
  await profile.selectOption("lite");
  await page.getByLabel("Write a message…").fill("Hello");
  await page.getByRole("button", { name: "Send" }).click();
  const messages = page.getByTestId("messages");
  await expect(messages.getByText("Hello")).toBeVisible();
  await expect(page.getByRole("button", { name: "Send" })).toBeVisible({ timeout: 90_000 });
  await page.screenshot({ path: info.outputPath("user-chat.png") });
  const chats = await (await page.request.get("/api/chats")).json();
  const chat = await (await page.request.get(`/api/chats/${chats[0].id}`)).json();
  const reply = chat.messages.find((item: { role: string }) => item.role === "assistant");
  expect(reply.status).toBe("completed");
  expect(reply.content.length).toBeGreaterThan(0);
  expect(reply.generation_params.profile).toBe("lite");
  expect(reply.generation_params.max_tokens).toBeLessThanOrEqual(128);
});

test("revoking the model removes it from the user", async ({ page, request, baseURL }) => {
  const admin = await adminApi(request, baseURL ?? "");
  await admin.put("/models/allowed", { ids: [] });
  await signIn(page, user.username, user.password);
  await page.goto("/");
  await expect(page.getByText("No models are available")).toBeVisible();
});
