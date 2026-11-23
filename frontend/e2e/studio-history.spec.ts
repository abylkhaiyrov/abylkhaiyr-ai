import { expect, test } from "@playwright/test";
import { signIn } from "./support";

test.skip(!process.env.E2E_BACKEND, "requires a running Studio backend");

test("training history is virtualized, filterable, keyboard accessible and comparable", async ({ page }, info) => {
  await signIn(page);
  await page.goto("/history");
  await expect(page.getByRole("heading", { name: "Training history" })).toBeVisible();
  const list = page.getByRole("list", { name: "Runs" });
  await expect(list).toBeVisible();
  const total = Number((await page.getByText(/Showing \d+ of \d+/).innerText()).match(/of (\d+)/)?.[1]);
  expect(total).toBeGreaterThan(5);
  const rendered = await list.getByRole("listitem").count();
  expect(rendered).toBeLessThanOrEqual(total);
  const viewportRows = Math.floor((page.viewportSize()?.height ?? 0) * 0.68 / 64);
  if (total > viewportRows + 12) expect(rendered).toBeLessThan(total);

  const first = list.getByRole("button").first();
  await first.focus();
  await page.keyboard.press("ArrowDown");
  await expect(list.locator('[data-row="1"]')).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("main").getByRole("complementary")).toBeVisible();
  await expect(page.locator("main").getByRole("complementary").getByText("Dataset", { exact: true }).first()).toBeVisible();
  await page.screenshot({ path: info.outputPath("history-detail.png"), fullPage: true });

  await list.evaluate((element) => element.scrollTo(0, element.scrollHeight));
  await expect(list.locator(`[aria-posinset="${total}"]`)).toBeVisible();

  await list.evaluate((element) => element.scrollTo(0, 0));
  await page.getByLabel(/^Compare /).nth(0).check();
  await page.getByLabel(/^Compare /).nth(1).check();
  await expect(page.getByText("Run comparison")).toBeVisible();
  await page.screenshot({ path: info.outputPath("history-compare.png"), fullPage: true });

  await page.getByLabel("Search by name or model").fill("v4");
  await expect(page.getByText(/Showing \d+ of \d+/)).toContainText("Showing");
  const filtered = Number((await page.getByText(/Showing \d+ of \d+/).innerText()).match(/of (\d+)/)?.[1]);
  expect(filtered).toBeLessThan(total);
  await page.getByLabel("Search by name or model").fill("no-such-run-xyz");
  await expect(page.getByText("No runs match these filters")).toBeVisible();
});

test("run pages are stable and non-overlapping", async ({ page }) => {
  await signIn(page);
  const read = async (offset: number) => (await (await page.request.get(`/api/catalog/runs?limit=7&offset=${offset}`)).json()) as { items: { id: string }[]; total: number };
  const first = await read(0);
  const second = await read(7);
  const all = await (await page.request.get("/api/catalog/runs?limit=200")).json();
  expect([...first.items, ...second.items].map((item) => item.id)).toEqual(all.items.slice(0, first.items.length + second.items.length).map((item: { id: string }) => item.id));
  expect(new Set([...first.items, ...second.items].map((item) => item.id)).size).toBe(first.items.length + second.items.length);
});
