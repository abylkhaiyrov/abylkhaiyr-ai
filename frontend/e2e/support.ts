import { expect, type APIRequestContext, type Page } from "@playwright/test";

export const ADMIN_PASSWORD = process.env.E2E_ADMIN_PASSWORD ?? "admin";

export async function adminApi(request: APIRequestContext, baseURL: string) {
  const login = await request.post("/api/auth/login", { data: { identifier: "admin", password: ADMIN_PASSWORD }, headers: { origin: baseURL } });
  expect(login.ok()).toBeTruthy();
  const state = await request.storageState();
  const csrf = state.cookies.find((cookie) => cookie.name === "abylkhaiyr_csrf")?.value ?? "";
  const headers = { "X-CSRF-Token": csrf, origin: baseURL };
  return {
    get: async <T>(path: string) => (await request.get(`/api${path}`)).json() as Promise<T>,
    put: async <T>(path: string, data: unknown) => {
      const response = await request.put(`/api${path}`, { data, headers });
      expect(response.ok()).toBeTruthy();
      return response.json() as Promise<T>;
    },
  };
}

export async function signIn(page: Page, identifier = "admin", password = ADMIN_PASSWORD, locale: "en" | "ru" = "en") {
  await page.addInitScript((value) => localStorage.setItem("abylkhaiyr-language", value), locale);
  await page.goto("/login");
  await page.getByLabel(/username or email|имя пользователя/i).fill(identifier);
  await page.getByLabel(/^(password|пароль)$/i).fill(password);
  await page.getByRole("button", { name: /sign in|войти/i }).click();
  await page.waitForURL((url) => !url.pathname.startsWith("/login"));
  await expect(page.getByRole("link").first()).toBeVisible();
}
