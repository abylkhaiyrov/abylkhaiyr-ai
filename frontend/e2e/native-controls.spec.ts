import { expect, test, type Page } from "@playwright/test";

type Theme = "light" | "dark";

async function openLogin(page: Page, theme: Theme) {
  await page.addInitScript((value) => localStorage.setItem("abylkhaiyr-theme", value), theme);
  await page.goto("/login");
  await page.locator("select").first().waitFor();
}

function luminance(rgb: string) {
  const [r, g, b] = (rgb.match(/\d+(\.\d+)?/g) ?? ["0", "0", "0"]).slice(0, 3).map(Number).map((value) => {
    const channel = value / 255;
    return channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a: string, b: string) {
  const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (high + 0.05) / (low + 0.05);
}

for (const theme of ["light", "dark"] as const) {
  test.describe(`native select popup, ${theme} theme`, () => {
    test("declares the colour scheme so the OS popup matches the theme", async ({ page }) => {
      await openLogin(page, theme);
      const scheme = await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme);
      expect(scheme).toBe(theme);
    });

    test("select and option are opaque with readable contrast", async ({ page }) => {
      await openLogin(page, theme);
      const styles = await page.locator("select").first().evaluate((select) => {
        const option = select.querySelector("option") as HTMLOptionElement;
        const own = getComputedStyle(select);
        const item = getComputedStyle(option);
        return { selectBg: own.backgroundColor, selectColor: own.color, optionBg: item.backgroundColor, optionColor: item.color };
      });
      for (const color of [styles.selectBg, styles.optionBg]) {
        expect(color).not.toBe("rgba(0, 0, 0, 0)");
        expect(color).not.toBe("transparent");
      }
      expect(contrast(styles.optionColor, styles.optionBg)).toBeGreaterThanOrEqual(4.5);
      expect(contrast(styles.selectColor, styles.selectBg)).toBeGreaterThanOrEqual(4.5);
    });

    test("language select is keyboard operable", async ({ page }) => {
      await openLogin(page, theme);
      const select = page.locator("select").first();
      await select.focus();
      await select.selectOption("en");
      await expect(select).toHaveValue("en");
      await expect(page.locator("html")).toHaveAttribute("lang", "en");
    });
  });
}

test("no horizontal overflow on the login page", async ({ page }) => {
  await openLogin(page, "dark");
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});

test("login page screenshot", async ({ page }, info) => {
  await openLogin(page, "dark");
  await page.screenshot({ path: info.outputPath("login-dark.png"), fullPage: true });
});
