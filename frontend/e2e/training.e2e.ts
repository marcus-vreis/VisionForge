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

test("tells the user when the server stops answering during a training", async ({ page }) => {
  await prepareClassification(page, "e2e_offline", 60);
  const sheet = await train(page);
  await expect(sheet.getByText(/^treinando · /)).toBeVisible();

  // The server "goes away" for the run's status poll and live stream only; the
  // training itself keeps going on the server.
  const down = (route: { abort: () => Promise<void> }) => route.abort();
  await page.route("**/api/experiment/status", down);
  await page.route("**/api/experiment/events", down);

  await expect(sheet.getByText("treino falhou", { exact: true })).toBeVisible({ timeout: 20_000 });
  await expect(sheet.getByText(/Não foi possível conectar ao servidor/).first()).toBeVisible();

  await page.waitForTimeout(3_000);
  await page.unroute("**/api/experiment/status", down);
  await page.unroute("**/api/experiment/events", down);

  // The overlay does not reconnect once the poll has failed (it stays on the
  // error); what comes back is the rest of the app: the run finished on the
  // server and the History lists it.
  await sheet.getByRole("button", { name: "Fechar", exact: true }).click();
  await expect(async () => {
    await page.keyboard.press("Escape");
    const card = await findInHistory(page, "e2e_offline");
    await expect(card.getByText(/^completed$/i)).toBeVisible({ timeout: 2_000 });
  }).toPass({ timeout: 90_000 });
});
