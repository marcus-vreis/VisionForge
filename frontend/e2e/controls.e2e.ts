import { expect, test } from "@playwright/test";
import path from "node:path";
import { realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createServer, type ViteDevServer } from "vite";

let server: ViteDevServer;
let url: string;
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

test.beforeAll(async () => {
  server = await createServer({
    root,
    cacheDir: path.resolve(root, "../.codex/accessibility-vite-cache"),
    server: { host: "127.0.0.1", port: 0, open: false, fs: { allow: [root, realpathSync(path.join(root, "node_modules"))] } },
    plugins: [{
      name: "controls-test-page",
      configureServer(vite) {
        vite.middlewares.use("/__controls", async (_request, response) => {
          const html = await vite.transformIndexHtml("/__controls", '<html><body><div id="root"></div><script type="module" src="/e2e/controls.fixture.tsx"></script></body></html>');
          response.setHeader("Content-Type", "text/html");
          response.end(html);
        });
      },
    }],
  });
  await server.listen();
  url = server.resolvedUrls!.local[0];
});

test.after all(async () => { await server?.close(); });

test("labels focus unique inputs and retain feedback across keyboard edits", async ({ page }) => {
  await page.goto(`${url}__controls`);
  const first = page.getByRole("region", { name: "Primeiro card" });
  const second = page.getByRole("region", { name: "Segundo card" });
  const input = first.getByRole("textbox", { name: "CSV de grupos", exact: true });
  const repeated = second.getByLabel("CSV de grupos", { exact: true });
  const id = await input.getAttribute("id");
  expect(id).toBeTruthy();
  expect(await repeated.getAttribute("id")).not.toBe(id);
  await expect(input).toHaveAccessibleDescription("Obrigatório Use um arquivo local. Informe o CSV.");
  await expect(input).toHaveAttribute("aria-invalid", "true");
  await first.locator("label").filter({ hasText: "CSV de grupos" }).click();
  await expect(input).toBeFocused();
  await input.pressSequentially("groups.csv");
  await expect(input).toHaveAttribute("id", id!);
  await expect(input).not.toHaveAttribute("aria-invalid");
  await expect(input).toHaveAccessibleDescription("Obrigatório Use um arquivo local.");
  await second.locator("label").click();
  await expect(repeated).toBeFocused();
  await first.locator("label").filter({ hasText: "Épocas" }).click();
  const number = first.getByRole("textbox", { name: "Épocas", exact: true });
  await expect(number).toBeFocused();
  await number.fill("7");
  await number.press("Enter");
  await expect(number).toHaveValue("7");
  await number.focus();
  await number.press("Tab");
  const toggle = first.getByRole("switch", { name: "AMP", exact: true });
  await expect(toggle).toBeFocused();
  await toggle.press("Space");
  await expect(toggle).toBeChecked();
  await first.locator("label").filter({ hasText: "AMP" }).click();
  await expect(toggle).toBeFocused();
  await expect(toggle).not.toBeChecked();
});

test("dropdown and segmented choices expose names, keyboard selection and state", async ({ page }) => {
  await page.goto(`${url}__controls`);
  const select = page.getByRole("button", { name: "Modelo", exact: true });
  await page.locator("label").filter({ hasText: "Modelo" }).click();
  await expect(select).toBeFocused();
  await expect(select).toHaveAttribute("aria-expanded", "true");
  // The portaled options follow the second card in document tab order.
  for (let step = 0; step < 5; step++) await page.keyboard.press("Tab");
  await expect(page.getByRole("group", { name: "Modelo", exact: true }).getByRole("button", { name: "sgd" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(select).toHaveText("sgd");
  await expect(select).toBeFocused();
  await expect(select).toHaveAttribute("aria-expanded", "false");
  await select.press("Space");
  for (let step = 0; step < 4; step++) await page.keyboard.press("Tab");
  await expect(page.getByRole("group", { name: "Modelo", exact: true }).getByRole("button", { name: "adam" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(select).toBeFocused();
  await expect(select).toHaveAttribute("aria-expanded", "false");
  const segmented = page.getByRole("group", { name: "Otimizador", exact: true });
  await expect(segmented.getByRole("button", { name: "sgd", pressed: true })).toBeVisible();
  const adam = segmented.getByRole("button", { name: "adam" });
  await adam.focus();
  await adam.press("Space");
  await expect(adam).toHaveAttribute("aria-pressed", "true");
  await expect(segmented.getByRole("button", { name: "sgd" })).toHaveAttribute("aria-pressed", "false");
});
