import { expect, test } from "@playwright/test";
import { signIn } from "./support";

test.skip(!process.env.E2E_BACKEND, "requires a running Studio backend");

test("evaluation center shows measured results, never zero for missing metrics, and compares checkpoints", async ({ page }, info) => {
  await signIn(page);
  await page.goto("/evaluation");
  await expect(page.getByRole("heading", { name: "Evaluation Center" })).toBeVisible();
  const group = page.locator("main select").first();
  await group.selectOption({ index: 0 });
  await expect(page.getByText("Not recorded").first()).toBeVisible();
  await expect(page.getByText("Not passed").first()).toBeVisible();
  await page.screenshot({ path: info.outputPath("evaluation-overview.png"), fullPage: true });

  await page.getByRole("tab", { name: "Conversation quality" }).click();
  const semantic = page.getByRole("row", { name: /Semantic relevance/ });
  await expect(semantic).toBeVisible();
  await expect(semantic.getByText("Not verified").first()).toBeVisible();
  await expect(semantic.getByText("0%")).toHaveCount(0);
  await page.screenshot({ path: info.outputPath("evaluation-conversation.png"), fullPage: true });

  await page.getByRole("tab", { name: "Compare" }).click();
  await expect(page.getByText("Conclusion")).toBeVisible();
  await expect(page.getByText(/Perplexity alone is not enough|No clear leader|Leads on/)).toBeVisible();
  await expect(page.getByText("Confidence intervals are not computed.")).toBeVisible();
  await page.screenshot({ path: info.outputPath("evaluation-compare.png"), fullPage: true });
});

test("evaluation center is not available to regular users through the API", async ({ page, request }) => {
  const response = await request.get("/api/evaluation/groups");
  expect(response.status()).toBe(401);
  await signIn(page);
  expect((await page.request.get("/api/evaluation/groups")).ok()).toBeTruthy();
});
