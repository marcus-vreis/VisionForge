import { expect, test } from "@playwright/test";
import { enter } from "./helpers";

/** GUI-13: skip link to the content and 44 px touch targets in the header, on a phone. */
test.use({ viewport: { width: 375, height: 812 } });

test("o primeiro Tab cai no link que pula para o conteúdo", async ({ page }) => {
  await enter(page);
  await page.keyboard.press("Tab");
  const skip = page.getByRole("link", { name: "Pular para o conteúdo" });
  await expect(skip).toBeFocused();
  await expect(skip).toHaveAttribute("href", "#vf-main");
  await page.keyboard.press("Enter");
  await expect(page.locator("main#vf-main")).toBeFocused();
});

test("botões e links do cabeçalho têm ao menos 44x44 px a 375x812", async ({ page }) => {
  await enter(page);
  const targets = page.locator("header.vf-app-header").locator("button, a");
  const count = await targets.count();
  expect(count).toBeGreaterThan(0);
  const small: string[] = [];
  for (let i = 0; i < count; i++) {
    const target = targets.nth(i);
    if (!(await target.isVisible())) continue;
    const box = await target.boundingBox();
    if (box && (box.width < 44 || box.height < 44)) {
      small.push(`${await target.innerText()} ${box.width}x${box.height}`);
    }
  }
  expect(small).toEqual([]);
});
