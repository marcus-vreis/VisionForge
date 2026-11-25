import { randomBytes } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { crc32 } from "node:zlib";
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
  // The page does not call itself "local" on a shared server.
  await expect(page).toHaveTitle(pt.app.pageTitleServer);
  await expect(page.getByText(/shared ai studio/)).toBeVisible();
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

/** A .zip of `entries`, stored without compression: enough for a tiny dataset. */
function storedZip(entries: Record<string, Buffer>): Buffer {
  const pieces: Buffer[] = [];
  const index: Buffer[] = [];
  let offset = 0;
  for (const [entry, data] of Object.entries(entries)) {
    const name = Buffer.from(entry, "utf8");
    const crc = crc32(data);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0x0800, 6); // UTF-8 names
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(data.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(name.length, 26);
    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 4);
    central.writeUInt16LE(20, 6);
    central.writeUInt16LE(0x0800, 8);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(data.length, 20);
    central.writeUInt32LE(data.length, 24);
    central.writeUInt16LE(name.length, 28);
    central.writeUInt32LE(offset, 42);
    pieces.push(local, name, data);
    index.push(central, name);
    offset += local.length + name.length + data.length;
  }
  const directory = Buffer.concat(index);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(index.length / 2, 8);
  end.writeUInt16LE(index.length / 2, 10);
  end.writeUInt32LE(directory.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...pieces, directory, end]);
}

test("a person sends a dataset .zip of their own and resumes it after a failure", async ({
  page,
  request,
}) => {
  const zip = storedZip({
    "frutas/train/maca/1.jpg": randomBytes(2048),
    "frutas/val/maca/2.jpg": randomBytes(1024),
  });
  // The first part is lost on the way, as when the connection drops.
  let dropped = false;
  await page.route("**/api/uploads/*/parts/*", async (route) => {
    if (dropped) return route.continue();
    dropped = true;
    return route.abort("connectionreset");
  });

  await enter(page, "Ana");
  await page.locator("[data-tour='datasets']").click();
  await page.getByRole("tab", { name: pt.datasetsOverlay.tabMine }).click();
  const mine = page.getByRole("region", { name: pt.ownDatasets.title });
  await mine.locator("input[type='file']").setInputFiles({
    name: "frutas.zip",
    mimeType: "application/zip",
    buffer: zip,
  });

  const resume = mine.getByRole("button", { name: pt.ownDatasets.uploadResume });
  await expect(resume).toBeVisible();
  await expect(mine.getByRole("alert")).toBeVisible();
  await resume.click();

  await expect(mine.getByText(pt.ownDatasets.uploaded("frutas"), { exact: true })).toBeVisible();
  await expect(mine.getByText("frutas", { exact: true })).toBeVisible();
  const own = (await (await request.get("/api/datasets/own")).json()).datasets;
  expect(own).toEqual([expect.objectContaining({ name: "frutas", size_bytes: 3072 })]);
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
