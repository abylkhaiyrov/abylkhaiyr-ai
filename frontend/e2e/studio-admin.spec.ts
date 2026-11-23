import { expect, test } from "@playwright/test";
import { signIn as login } from "./support";

test.skip(!process.env.E2E_BACKEND, "requires a running Studio backend");

test("dashboard defaults to the latest real run, not a stale pipeline", async ({ page }, info) => {
  await login(page);
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Overview" })).toBeVisible();
  await expect(page.getByText("Latest experiment")).toBeVisible();
  await expect(page.getByText("No production model assigned").first()).toBeVisible();
  await page.screenshot({ path: info.outputPath("dashboard.png"), fullPage: true });
});

test("processes page groups jobs and keeps history", async ({ page }, info) => {
  await login(page);
  await page.goto("/processes");
  for (const title of ["Active processes", "Queue", "Completed jobs", "Failed jobs", "Run history"]) {
    await expect(page.getByText(title, { exact: false }).first()).toBeVisible();
  }
  await page.screenshot({ path: info.outputPath("processes.png"), fullPage: true });
});

test("models page separates the catalog from the checkpoint registry", async ({ page }, info) => {
  await login(page);
  await page.goto("/models");
  await expect(page.getByRole("tab", { name: "Model catalog" })).toBeVisible();
  await expect(page.getByText("Visible to users").first()).toBeVisible();
  await expect(async () => {
    await page.getByRole("tab", { name: "Checkpoint registry" }).click();
    await expect(page.getByRole("tab", { name: "Checkpoint registry" })).toHaveAttribute("aria-selected", "true", { timeout: 2000 });
  }).toPass({ timeout: 15_000 });
  await expect(page.getByRole("columnheader", { name: "Stage" })).toBeVisible();
  await expect(page.getByText(/Showing 50 of \d+/)).toBeVisible();
  const before = await page.locator("tbody tr").count();
  await page.getByRole("button", { name: "Show more" }).click();
  await expect.poll(() => page.locator("tbody tr").count()).toBeGreaterThan(before);
  await page.getByLabel("Search by name or model").fill("tiny_v31");
  await expect(page.getByText(/Showing \d+ of \d+/)).toBeVisible();
  expect(await page.locator("tbody tr").count()).toBeLessThan(before);
  await page.screenshot({ path: info.outputPath("models.png"), fullPage: true });
});

test("catalog and experiments load one server page at a time", async ({ page }) => {
  await login(page);
  await page.goto("/models");
  await expect(page.getByText(/Showing 30 of \d+/)).toBeVisible();
  const first = await page.locator("tbody tr").count();
  expect(first).toBeLessThanOrEqual(30);
  await page.getByRole("button", { name: "Show more" }).click();
  await expect.poll(() => page.locator("tbody tr").count()).toBeGreaterThan(first);
  await page.getByLabel("Search by name or model").fill("no-such-model-xyz");
  await expect(page.getByText("No checkpoints found")).toBeVisible();

  await page.goto("/experiments");
  await expect(page.getByText(/Showing \d+ of \d+/)).toBeVisible();
  const total = Number((await page.getByText(/Showing \d+ of \d+/).innerText()).match(/of (\d+)/)?.[1]);
  const rows = await page.locator("tbody tr").count();
  expect(rows).toBeLessThanOrEqual(30);
  if (total > 30) {
    await page.getByRole("button", { name: "Show more" }).click();
    await expect.poll(() => page.locator("tbody tr").count()).toBeGreaterThan(rows);
  }
});
