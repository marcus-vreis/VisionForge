import { expect, test } from "@playwright/test";

import { enter, findInHistory, prepareClassification, train } from "./helpers";

// One server, one training slot: only the queue test overlaps two submissions;
// every test waits for its own training to end before the next one starts.
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
  await prepareClassification(page, "e2e_offline", 8);
  // Install before submission: routing cannot break an EventSource that has
  // already connected. Both real requests must fail during this outage.
  const down = (route: { abort: () => Promise<void> }) => route.abort();
  await page.route("**/api/experiment/status", down);
  await page.route("**/api/experiment/events", down);
  const streamFailed = page.waitForEvent("requestfailed", {
    predicate: (request) => request.url().endsWith("/api/experiment/events"),
  });
  const pollFailed = page.waitForEvent("requestfailed", {
    predicate: (request) => request.url().endsWith("/api/experiment/status"),
  });
  const sheet = await train(page);
  await streamFailed;
  await pollFailed;

  await expect(sheet.getByText("reconectando…", { exact: true })).toBeVisible({ timeout: 20_000 });
  await page.waitForTimeout(3_000);
  await expect(sheet.getByText("treino falhou", { exact: true })).toHaveCount(0);

  await page.unroute("**/api/experiment/status", down);
  await page.unroute("**/api/experiment/events", down);

  // The overlay picks the run up again and follows it to its end.
  await expect(sheet.getByText("reconectando…", { exact: true })).toHaveCount(0, { timeout: 20_000 });
  await expect(sheet.getByText("treino concluído", { exact: true })).toBeVisible({ timeout: 120_000 });
  await expect(sheet.getByText("treino falhou", { exact: true })).toHaveCount(0);

  await sheet.getByRole("button", { name: "Fechar", exact: true }).click();
  const card = await findInHistory(page, "e2e_offline");
  await expect(card.getByText(/^completed$/i)).toBeVisible();
});

test("queues a second browser's run and starts it after the active run stops", async ({ page, browser }) => {
  // The waiting submission has its own hook and stream, as with two researchers
  // sharing one server. Its completion must never be confused with the first's.
  const waitingContext = await browser.newContext();
  const waitingPage = await waitingContext.newPage();
  try {
    await enter(waitingPage, "Queue");
    await waitingPage.locator("[data-tour='tabs']").getByText("Classificação", { exact: true }).click();
    await prepareClassification(waitingPage, "e2e_queued", 8);
    await prepareClassification(page, "e2e_queue_active", 200);
    const active = await train(page);
    await expect(active.getByText(/^> epoch 1\/200 · /)).toBeVisible({ timeout: 60_000 });

    const waiting = await train(waitingPage);
    await expect(waiting.getByText(/^na fila · /)).toBeVisible();
    await expect(waiting.getByText(/aguardando a GPU — 1º na fila/)).toBeVisible();
    await expect(waiting.getByText("treino concluído", { exact: true })).toHaveCount(0);

    await active.getByRole("button", { name: "■ Parar" }).click();
    await active.getByRole("button", { name: "Parar o treino" }).click();
    await expect(active.getByText("Treino interrompido", { exact: true })).toBeVisible({ timeout: 60_000 });
    await expect(waiting.getByText(/^treinando · /)).toBeVisible({ timeout: 60_000 });
    await expect(waiting.getByText("treino concluído", { exact: true })).toBeVisible({ timeout: 120_000 });

    await waiting.getByRole("button", { name: "Fechar", exact: true }).click();
    const card = await findInHistory(waitingPage, "e2e_queued");
    await expect(card.getByText(/^completed$/i)).toBeVisible();
    await expect(card.getByText("8 epochs", { exact: true })).toBeVisible();
  } finally {
    await waitingContext.close();
  }
});

test("reconnects the live stream while status polling stays available", async ({ page }) => {
  await prepareClassification(page, "e2e_stream", 200);
  const down = (route: { abort: () => Promise<void> }) => route.abort();
  await page.route("**/api/experiment/events", down);
  const streamFailed = page.waitForEvent("requestfailed", {
    predicate: (request) => request.url().endsWith("/api/experiment/events"),
  });
  const healthyPoll = page.waitForResponse((response) =>
    response.url().endsWith("/api/experiment/status") && response.ok(),
  );
  const sheet = await train(page);
  await streamFailed;
  await healthyPoll;
  await expect(sheet.getByText("reconectando…", { exact: true })).toHaveCount(0);

  const reconnected = page.waitForResponse((response) =>
    response.url().endsWith("/api/experiment/events") && response.ok(),
  );
  await page.unroute("**/api/experiment/events", down);
  await reconnected;
  await expect(sheet.getByText(/^> epoch \d+\/200 · /).first()).toBeVisible({ timeout: 60_000 });
  await sheet.getByRole("button", { name: "■ Parar" }).click();
  await sheet.getByRole("button", { name: "Parar o treino" }).click();
  await expect(sheet.getByText("Treino interrompido", { exact: true })).toBeVisible({ timeout: 60_000 });
});
