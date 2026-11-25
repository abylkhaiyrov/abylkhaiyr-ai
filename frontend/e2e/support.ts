import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import { join } from "node:path";
import { expect, type APIRequestContext, type BrowserContext, type Page } from "@playwright/test";

export const ADMIN_PASSWORD = process.env.E2E_ADMIN_PASSWORD ?? "admin";

export async function adminApi(request: APIRequestContext, baseURL: string) {
  const file = sessionFile(process.env.E2E_BASE_URL ?? "dev", "admin");
  let cookies = await cachedCookies(file);
  if (!cookies) {
    const login = await request.post("/api/auth/login", { data: { identifier: "admin", password: ADMIN_PASSWORD }, headers: { origin: baseURL } });
    expect(login.ok()).toBeTruthy();
    cookies = (await request.storageState()).cookies as NonNullable<typeof cookies>;
    await fs.mkdir(".auth", { recursive: true });
    await fs.writeFile(file, JSON.stringify(cookies));
  }
  const csrf = cookies.find((cookie) => cookie.name === "abylkhaiyr_csrf")?.value ?? "";
  const headers = { "X-CSRF-Token": csrf, origin: baseURL, cookie: cookies.map((cookie) => `${cookie.name}=${cookie.value}`).join("; ") };
  return {
    get: async <T>(path: string) => (await request.get(`/api${path}`, { headers })).json() as Promise<T>,
    put: async <T>(path: string, data: unknown) => {
      const response = await request.put(`/api${path}`, { data, headers });
      expect(response.ok()).toBeTruthy();
      return response.json() as Promise<T>;
    },
  };
}

const SESSION_TTL_MS = 8 * 60_000;

function sessionFile(base: string, identifier: string) {
  return join(".auth", `${createHash("sha1").update(`${base}|${identifier}`).digest("hex").slice(0, 16)}.json`);
}

async function cachedCookies(file: string) {
  try {
    const stat = await fs.stat(file);
    if (Date.now() - stat.mtimeMs > SESSION_TTL_MS) return null;
    return JSON.parse(await fs.readFile(file, "utf8")) as Parameters<BrowserContext["addCookies"]>[0];
  } catch {
    return null;
  }
}

export async function signIn(page: Page, identifier = "admin", password = ADMIN_PASSWORD, locale: "en" | "ru" = "en") {
  await page.addInitScript((value) => localStorage.setItem("abylkhaiyr-language", value), locale);
  const file = sessionFile(process.env.E2E_BASE_URL ?? "dev", identifier);
  const cookies = await cachedCookies(file);
  if (cookies) {
    await page.context().addCookies(cookies);
    await page.goto("/");
    if (!new URL(page.url()).pathname.startsWith("/login")) {
      await expect(page.locator("main")).toBeVisible();
      return;
    }
  }
  await page.goto("/login");
  await page.getByLabel(/username or email|логин или email|имя пользователя/i).fill(identifier);
  await page.getByLabel(/^(password|пароль)$/i).fill(password);
  await page.getByRole("button", { name: /sign in|войти/i }).click();
  await page.waitForURL((url) => !url.pathname.startsWith("/login"));
  await expect(page.locator("main")).toBeVisible();
  await fs.mkdir(".auth", { recursive: true });
  await fs.writeFile(file, JSON.stringify(await page.context().cookies()));
}
