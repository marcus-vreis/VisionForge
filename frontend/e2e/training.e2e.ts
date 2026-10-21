import { expect, test } from "@playwright/test";

import { enter, findInHistory, prepareClassification, train } from "./helpers";

// One server, one training slot: the tests run in file order and each waits for
// its own training to end, so the next one never queues behind it.
test.describe.configure({ mode: "serial" });

test.beforeEach(async ({ page }) => {
  await enter(page);
  await page.locator("[data-tour='tabs']").getByText("Classificação", { exact: true }).click();
});

test("trains one epoch on the sample dataset and finds the run in the history", async ({ page }) => {
  await prepareClassification(page, "e2e_vertical", 1);
  const sheet = await train(page);

  await expect(sheet.getByText(/^treinando · /)).toBeVisible();
  await expect(sheet.getByText("treino concluído", { exact: true })).toBeVisible({ timeout: 120_000 });

  await sheet.getByRole("button", { name: "Fechar", exact: true }).click();
  const card = await findInHistory(page, "e2e_vertical");
  await expect(card.getByText(/^completed$/i)).toBeVisible();
  await expect(card.getByText("1 epoch", { exact: true })).toBeVisible();
});

test("stops a running training at the end of the current epoch", async ({ page }) => {
  await prepareClassification(page, "e2e_stop", 200);
  const sheet = await train(page);

  // Stop is offered once the first epoch says where this run can stop.
  await expect(sheet.getByText(/^> epoch 1\/200 · /)).toBeVisible({ timeout: 60_000 });
  await sheet.getByRole("button", { name: "■ Parar" }).click();
  await sheet.getByRole("button", { name: "Parar o treino" }).click();

  await expect(sheet.getByText("Treino interrompido", { exact: true })).toBeVisible({ timeout: 60_000 });
  await expect(sheet.getByText(/interrompido na época \d+\/200/)).toBeVisible();
});

test("rides out a few seconds without the server and still ends as completed", async ({ page }) => {
  await prepareClassification(page, "e2e_offline", 60);
  const sheet = await train(page);
  await expect(sheet.getByText(/^treinando · /)).toBeVisible();

  // The server "goes away" for the run's status poll and live stream only; the
  // training itself keeps going on the server.
  const down = (route: { abort: () => Promise<void> }) => route.abort();
  await page.route("**/api/experiment/status", down);
  await page.route("**/api/experiment/events", down);

  await expect(sheet.getByText("reconectando…", { exact: true })).toBeVisible({ timeout: 20_000 });
  await page.waitForTimeout(3_000);
  await expect(sheet.getByText("treino falhou", { exact: true })).toHaveCount(0);

  await page.unroute("**/api/experiment/status", down);
  await page.unroute("**/api/experiment/events", down);

  // The overlay picks the run up again and follows it to its end.
  await expect(sheet.getByText(/^treinando · /)).toBeVisible({ timeout: 20_000 });
  await expect(sheet.getByText("treino concluído", { exact: true })).toBeVisible({ timeout: 120_000 });
  await expect(sheet.getByText("treino falhou", { exact: true })).toHaveCount(0);

  await sheet.getByRole("button", { name: "Fechar", exact: true }).click();
  const card = await findInHistory(page, "e2e_offline");
  await expect(card.getByText(/^completed$/i)).toBeVisible();
});
