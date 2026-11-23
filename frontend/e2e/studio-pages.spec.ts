import { expect, test, type Page } from "@playwright/test";
import { signIn } from "./support";

test.skip(!process.env.E2E_BACKEND, "requires a running Studio backend");

const ADMIN_ROUTES = ["/", "/chat", "/training", "/monitoring", "/history", "/processes", "/models", "/datasets", "/experiments", "/evaluation", "/logs", "/users", "/notifications", "/settings", "/profile"];
const USER_ROUTES = ["/chat", "/notifications", "/settings", "/profile"];
const THEMES = ["light", "dark"] as const;

async function audit(page: Page, route: string, label: string) {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(route);
  await page.waitForLoadState("networkidle");
  await expect(page.locator("main")).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow, `${label} ${route} overflows horizontally`).toBeLessThanOrEqual(0);
  const blank = await page.evaluate(() => (document.querySelector("main")?.textContent ?? "").trim().length);
  expect(blank, `${label} ${route} rendered nothing`).toBeGreaterThan(0);
  expect(errors, `${label} ${route} threw`).toEqual([]);
}

for (const theme of THEMES) {
  test(`admin pages render without overflow, ${theme} theme`, async ({ page }, info) => {
    test.setTimeout(180_000);
    await page.addInitScript((value) => localStorage.setItem("abylkhaiyr-theme", value), theme);
    await signIn(page);
    for (const route of ADMIN_ROUTES) {
      await audit(page, route, `${info.project.name}/${theme}`);
      if (route === "/" || route === "/evaluation") await page.screenshot({ path: info.outputPath(`${theme}-${route.replace("/", "") || "overview"}.png`) });
    }
  });
}

test("regular user pages stay within the user layout", async ({ page, request, baseURL }, info) => {
  test.setTimeout(120_000);
  const user = { username: process.env.E2E_USER ?? "e2e_pages_user", password: "e2e-user-password-1" };
  const created = await request.post("/api/auth/register", { data: { firstName: "Page", lastName: "Audit", email: `${user.username}@example.com`, username: user.username, password: user.password }, headers: { origin: baseURL ?? "" } });
  expect([201, 409, 429], `register: ${created.status()}`).toContain(created.status());
  await signIn(page, user.username, user.password);
  for (const route of USER_ROUTES) await audit(page, route, `${info.project.name}/user`);
  for (const route of ["/training", "/models", "/users", "/evaluation", "/history"]) {
    await page.goto(route);
    await expect(page).toHaveURL(/\/chat/);
  }
  expect((await page.request.get("/api/runtime")).status()).toBe(403);
  expect((await page.request.get("/api/catalog/runs")).status()).toBe(403);
  expect((await page.request.get("/api/evaluation/groups")).status()).toBe(403);
});
