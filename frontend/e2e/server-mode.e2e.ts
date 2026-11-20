import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Browser, type BrowserContext } from "@playwright/test";

import { pt } from "../src/i18n/pt";
import { enter, findInHistory, openStep, prepareClassification, train } from "./helpers";

/**
 * The GUI on a server several people share (`visionforge gui --server`, ADR-212).
 *
 * serve.mjs starts that server with this machine as the trusted proxy, so the
 * browser plays the login proxy: a context names its account in the header
 * oauth2-proxy would set. Two accounts, Ana and Bia, share the one server.
 */

// The server's own disk: nobody's dataset folder, and not a shared one.
const work = path.join(path.dirname(fileURLToPath(import.meta.url)), ".work-server");

const USER_HEADER = "X-Forwarded-User";
const ANA = "ana@lab.org";
const BIA = "bia@lab.org";
const RUN_NAME = "e2e_server";

// One server and one training slot; the later tests read what the earlier left.
test.describe.configure({ mode: "serial" });

// The default page and `request` are Ana's.
test.use({ extraHTTPHeaders: { [USER_HEADER]: ANA } });

/** A browser signed in as another account, as the proxy would present it. */
async function signedInAs(browser: Browser, account: string): Promise<BrowserContext> {
  return browser.newContext({
    baseURL: test.info().project.use.baseURL,
    locale: "pt-BR",
    extraHTTPHeaders: { [USER_HEADER]: account },
  });
}

test("a request the proxy did not sign is refused", async ({ playwright }) => {
  const anonymous = await playwright.request.newContext({
    baseURL: test.info().project.use.baseURL,
    // Not the file's default: this client carries no account at all.
    extraHTTPHeaders: {},
  });
  // The proxy's own liveness probe carries no account.
  expect((await anonymous.get("/api/health")).status()).toBe(200);
  expect((await anonymous.get("/api/session")).status()).toBe(401);
  expect((await anonymous.get("/api/runs")).status()).toBe(401);
  await anonymous.dispose();
});

test("the first visit asks the name and keeps it in the person's own profile", async ({
  page,
  request,
}) => {
  await enter(page, "Ana");
  const chip = page.getByTitle(pt.profile.menuTitle);
  await expect(chip).toContainText("Ana");

  await chip.click();
  const menu = page.getByRole("dialog", { name: pt.profile.menuTitle });
  await expect(menu.getByText(pt.server.account(ANA), { exact: true })).toBeVisible();
  // Her own integrations, each sent with her own key; local MLflow has no
  // reader on a server, and the profile is the account: none to create.
  await expect(menu.getByRole("switch", { name: pt.wandb.label })).toBeVisible();
  await expect(menu.getByRole("switch", { name: pt.comet.label })).toBeVisible();
  await expect(menu.getByRole("switch", { name: pt.mlflow.label })).toHaveCount(0);
  await expect(menu.getByRole("button", { name: pt.profile.createTitle })).toHaveCount(0);
  await expect(menu.getByRole("button", { name: pt.profile.switchProfile })).toHaveCount(0);

  const listed = (await (await request.get("/api/profiles")).json()).profiles;
  expect(listed).toHaveLength(1);
  expect(listed[0].name).toBe("Ana");
  expect(listed[0].is_default).toBe(false);

  // The name lives on the server: a second load enters without asking.
  await page.reload();
  await expect(page.getByTitle(pt.profile.menuTitle)).toContainText("Ana");
  await expect(page.getByRole("textbox", { name: "Seu nome" })).toBeHidden();
});

test("a person trains on a dataset of their own folder and the run is theirs alone", async ({
  page,
  request,
  browser,
}) => {
  await enter(page, "Ana");
  await page.locator("[data-tour='tabs']").getByText("Classificação", { exact: true }).click();

  // No folder picker here: it would open a dialog on the server's own screen.
  // The page names the folders a dataset may live in.
  await openStep(page, "Dados");
  await expect(page.getByText(pt.server.foldersTitle, { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Procurar", exact: true })).toHaveCount(0);

  await prepareClassification(page, RUN_NAME, 1);
  // The sample dataset was written inside her own dataset folder.
  const session = await (await request.get("/api/session")).json();
  const own: string = session.dataset_dirs[0];
  expect(own).toContain(session.profile);
  await openStep(page, "Dados");
  const datasetPath = await page.evaluate(() =>
    [...document.querySelectorAll("input")]
      .map((input) => input.value)
      .find((value) => value.includes("exemplo-classificacao")),
  );
  expect(datasetPath).toContain(session.profile);
  await openStep(page, "Revisão");

  const sheet = await train(page);
  await expect(sheet.getByText("treino concluído", { exact: true })).toBeVisible({ timeout: 120_000 });
  // The results open as the run's page in the History; Esc steps back out.
  await sheet.getByRole("button", { name: /Ver resultados/ }).click();
  const history = page.getByTestId("history-overlay");
  await expect(history.getByRole("heading", { name: RUN_NAME, exact: true })).toBeVisible();
  await page.keyboard.press("Escape");
  await page.keyboard.press("Escape");
  await expect(history).toHaveCount(0);
  const card = await findInHistory(page, RUN_NAME);
  await expect(card.getByText("1 epoch", { exact: true })).toBeVisible();

  const mine: { run_id: string; experiment_name: string }[] = await (await request.get("/api/runs")).json();
  const run = mine.find((r) => r.experiment_name === RUN_NAME);
  expect(run).toBeDefined();

  // Bia shares the server and sees none of it.
  const context = await signedInAs(browser, BIA);
  const bia = await context.newPage();
  await enter(bia, "Bia");
  await expect(bia.getByTitle(pt.profile.menuTitle)).toContainText("Bia");
  await bia.locator("[data-tour='history']").click();
  await expect(bia.getByText(pt.history.emptyTitle, { exact: true })).toBeVisible();
  expect(await (await context.request.get("/api/runs")).json()).toHaveLength(0);
  expect((await context.request.get(`/api/runs/${run?.run_id}`)).status()).toBe(404);
  const theirs = (await (await context.request.get("/api/profiles")).json()).profiles;
  expect(theirs).toHaveLength(1);
  expect(theirs[0].name).toBe("Bia");
  await context.close();
});

test("a person deletes a dataset of their own folder, after being asked twice", async ({
  page,
  request,
}) => {
  // The sample dataset of the training above sits in Ana's own folder.
  const before = (await (await request.get("/api/datasets/own")).json()).datasets;
  expect(before.map((d: { name: string }) => d.name)).toEqual(["exemplo-classificacao"]);

  await enter(page, "Ana");
  await page.locator("[data-tour='datasets']").click();
  await page.getByRole("tab", { name: pt.datasetsOverlay.tabMine }).click();
  const list = page.getByRole("region", { name: pt.ownDatasets.title });
  await expect(list.getByText("exemplo-classificacao", { exact: true })).toBeVisible();

  // The first click only asks; cancelling keeps the folder.
  const remove = list.getByRole("button", {
    name: pt.ownDatasets.deleteLabel("exemplo-classificacao"),
  });
  await remove.click();
  await expect(list.getByText(pt.ownDatasets.confirm, { exact: true })).toBeVisible();
  await list.getByRole("button", { name: pt.ownDatasets.confirmNo }).click();
  expect((await (await request.get("/api/datasets/own")).json()).datasets).toHaveLength(1);

  await remove.click();
  await list.getByRole("button", { name: pt.ownDatasets.confirmYes }).click();
  await expect(
    list.getByText(pt.ownDatasets.deleted("exemplo-classificacao"), { exact: true }),
  ).toBeVisible();
  await expect(list.getByText(pt.ownDatasets.empty, { exact: true })).toBeVisible();
  expect((await (await request.get("/api/datasets/own")).json()).datasets).toHaveLength(0);
});

test("the server refuses what is outside a person's folders or off for a server", async ({
  request,
}) => {
  const session = await (await request.get("/api/session")).json();
  expect(session.server_mode).toBe(true);
  expect(session.user).toBe(ANA);
  // Two folders: her own and the one the operator shares.
  expect(session.dataset_dirs).toHaveLength(2);

  // A folder of the server's disk that is neither hers nor shared.
  const outside = await request.post("/api/dataset/stats", { data: { base_dir: work } });
  expect(outside.status()).toBe(422);
  // The profile is the account, and nobody edits another one.
  expect((await request.post("/api/profiles", { data: { name: "Outra" } })).status()).toBe(404);
  const other = await request.put("/api/profiles/default", { data: { name: "Outra" } });
  expect(other.status()).toBe(404);
  const mlflow = await request.put(`/api/profiles/${session.profile}`, {
    data: { settings: { mlflow: { enabled: true } } },
  });
  expect(mlflow.status()).toBe(422);
  // A dialog on the server's own screen.
  expect((await request.post("/api/dataset/pick", { data: {} })).status()).toBe(404);
});
