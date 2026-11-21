import { expect, test } from "@playwright/test";
import { enter, openStep } from "./helpers";

// At 800 px the Avançado grid of Treinamento was 877 px wide in a 657 px card:
// three `1fr` columns never shrink below their widest field (the optimizer
// buttons, "Early stop (paciência)"), so the third column ran off the card.
test.use({ viewport: { width: 800, height: 900 } });

test("a grade do Avançado cabe no cartão a 800 px", async ({ page }) => {
  await enter(page);
  await openStep(page, "Treinamento");
  const toggle = page.getByRole("button", { name: /avançado/i });
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-expanded", "true");

  const gridId = await toggle.getAttribute("aria-controls");
  expect(gridId).toBeTruthy();
  const grid = page.locator(`[id="${gridId}"]`);
  await expect(grid).toBeVisible();

  // What runs past the grid's right edge, so a failure names the culprit: the
  // fallback monospace font differs between systems, and so do the widths.
  const past = await grid.evaluate((el) => {
    const edge = el.getBoundingClientRect().right;
    return [...el.querySelectorAll("*")]
      .filter((child) => child.getBoundingClientRect().right > edge + 0.5)
      .map((child) => `${child.tagName} "${(child.textContent ?? "").trim().slice(0, 40)}"`);
  });
  expect(past).toEqual([]);
});
