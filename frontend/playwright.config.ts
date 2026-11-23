import { defineConfig, devices } from "@playwright/test";

const port = 5173;
const external = process.env.E2E_BASE_URL;

const projects = [
  { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  { name: "firefox", use: { ...devices["Desktop Firefox"] } },
  { name: "webkit", use: { ...devices["Desktop Safari"] } },
  { name: "chrome", use: { ...devices["Desktop Chrome"], channel: "chrome" } },
  { name: "edge", use: { ...devices["Desktop Edge"], channel: "msedge" } },
  { name: "mobile-375", use: { ...devices["Desktop Chrome"], viewport: { width: 375, height: 812 } } },
  { name: "hidpi-200", use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 720 }, deviceScaleFactor: 2 } },
  { name: "uhd-4k", use: { ...devices["Desktop Chrome"], viewport: { width: 3840, height: 2160 } } },
];

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["html", { open: "never" }]] : "list",
  expect: { timeout: 10_000 },
  use: { baseURL: external ?? `http://127.0.0.1:${port}`, trace: "retain-on-failure", screenshot: "only-on-failure" },
  projects,
  webServer: external
    ? undefined
    : {
        command: "npm run dev",
        url: `http://127.0.0.1:${port}`,
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
      },
});
