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
  await page.getByRole("tab", { name: "Checkpoint registry" }).click();
  await expect(page.getByRole("columnheader", { name: "Stage" })).toBeVisible();
  await page.screenshot({ path: info.outputPath("models.png"), fullPage: true });
});
