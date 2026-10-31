import { expect, test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { enter, fieldByLabel, findInHistory, openStep } from "./helpers";

const work = path.join(path.dirname(fileURLToPath(import.meta.url)), ".work");

test("explicit image audit, persisted K-fold groups and experiment bundle download", async ({ page }) => {
  await enter(page);
  await page.locator("[data-tour='tabs']").getByText("Classificação", { exact: true }).click();

  // Fixtures live only in the server's disposable work folder; no training or weights.
  const name = `evidence_${Date.now()}`;
  const dataset = path.join(work, "datasets", name);
  const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGMUCUhhYGBgYgADAAk6AMwAweHWAAAAAElFTkSuQmCC", "base64");
  for (const split of ["train", "val", "test"]) {
    const folder = path.join(dataset, split, "class_a");
    mkdirSync(folder, { recursive: true });
    writeFileSync(path.join(folder, "same.png"), png);
  }
  writeFileSync(path.join(dataset, "groups.csv"), "path,group\ntrain/class_a/same.png,patient1\nval/class_a/same.png,patient1\ntest/class_a/same.png,patient1\n");

  const auditRequests: string[] = [];
  page.on("request", (request) => {
    if (request.url().endsWith("/api/dataset/audit")) auditRequests.push(request.postData() ?? "");
  });
  await fieldByLabel(page, "Pasta base do dataset").fill(dataset);
  const audit = page.getByTestId("dataset-audit");
  await expect(audit.getByRole("button", { name: "Auditar imagens", exact: true })).toBeEnabled();
  await expect(audit.getByTestId("audit-result")).toHaveCount(0);
  expect(auditRequests).toHaveLength(0);
  await audit.locator("input").fill("groups.csv");
  // Forward to the real server, holding only delivery so pending UI is observable.
  let releaseAudit = () => {};
  const auditGate = new Promise<void>((resolve) => { releaseAudit = resolve; });
  await page.route("**/api/dataset/audit", async (route) => {
    const result = await route.fetch();
    await auditGate;
    await route.fulfill({ response: result });
  }, { times: 1 });
  const response = page.waitForResponse((res) => res.url().endsWith("/api/dataset/audit"));
  await audit.getByRole("button", { name: "Auditar imagens", exact: true }).click();
  await expect(audit.getByRole("button", { name: "Auditando imagens…", exact: true })).toBeDisabled();
  releaseAudit();
  expect((await response).status()).toBe(200);
  await expect(audit.getByTestId("audit-result")).toContainText("Grupos de duplicatas exatas: 1");
  await expect(audit.getByTestId("audit-result")).toContainText("Grupos entre splits: 1");
  await expect(audit).toContainText("Similaridade visual não é verificada.");
  expect(JSON.parse(auditRequests[0])).toMatchObject({ base_dir: dataset, splits: { train: "train", val: "val", test: "test" }, group_manifest: "groups.csv", strict_groups: false });
  await audit.getByText(/^Detalhes/).click();
  await expect(audit.getByTestId("audit-result")).toContainText("patient1");

  // A changed dataset invalidates the old result; HTTP failures stay visible.
  await fieldByLabel(page, "Pasta base do dataset").fill(path.join(dataset, "missing"));
  await expect(audit.getByTestId("audit-result")).toHaveCount(0);
  await audit.getByRole("button", { name: "Auditar imagens", exact: true }).click();
  await expect(audit.getByRole("alert")).toContainText("Não foi possível auditar as imagens.");
  await fieldByLabel(page, "Pasta base do dataset").fill(dataset);

  await openStep(page, "Estratégia");
  await page.getByRole("button", { name: "K-Fold (CV)", exact: true }).click();
  const cv = page.getByTestId("classification-cross-validation");
  await expect(cv).toBeVisible();
  const groupField = cv.getByLabel("CSV de grupos (opcional)", { exact: true });
  await expect(groupField).toHaveCount(1);
  await groupField.fill("groups.csv");
  // Switching strategy preserves the classification configuration.
  await page.getByRole("button", { name: "Treino simples", exact: true }).click();
  await expect(cv).toHaveCount(0);
  await page.getByRole("button", { name: "K-Fold (CV)", exact: true }).click();
  await expect(groupField).toHaveValue("groups.csv");
  await openStep(page, "Revisão");
  const configDownload = page.waitForEvent("download");
  await page.getByTitle("Exportar configuração atual como arquivo .yaml").click();
  const configStream = await (await configDownload).createReadStream();
  const configChunks: Buffer[] = [];
  for await (const chunk of configStream) configChunks.push(Buffer.from(chunk));
  expect(Buffer.concat(configChunks).toString("utf8")).toContain("group_manifest: groups.csv");

  const slug = await page.evaluate(() => localStorage.getItem("vf.profile") ?? "default");
  const models = slug === "default" ? path.join(work, "outputs", "models") : path.join(work, "outputs", "profiles", slug, "models");
  const runId = `${name}_20261021_120000`;
  const runDir = path.join(models, name, runId);
  mkdirSync(runDir, { recursive: true });
  writeFileSync(path.join(runDir, "predictions.csv"), "label,prediction\na,a\n");
  writeFileSync(path.join(runDir, "private.env"), "not exported");
  writeFileSync(path.join(runDir, "run.json"), JSON.stringify({
    experiment: name, timestamp: "2026-10-21T12:00:00", status: "completed", schema_version: 1,
    config: { name, task: "multiclass", block: "classification", model: { name: "resnet18", pretrained: false, num_classes: 2 }, data: { base_dir: dataset }, training: { epochs: 1 } },
    history: [], metrics: { total_epochs: 1, test_accuracy: 1 }, environment: {},
    artifacts: { predictions: "predictions.csv" },
  }));
  const card = await findInHistory(page, name);
  await card.click();
  let releaseBundle = () => {};
  const bundleGate = new Promise<void>((resolve) => { releaseBundle = resolve; });
  await page.route(`**/api/runs/${runId}/bundle`, async (route) => {
    const result = await route.fetch();
    await bundleGate;
    await route.fulfill({ response: result });
  }, { times: 1 });
  const bundleRequest = page.waitForRequest((req) => req.url().endsWith(`/api/runs/${runId}/bundle`));
  const bundleDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Baixar bundle do experimento", exact: true }).click();
  await expect(page.getByRole("button", { name: "Preparando bundle…", exact: true })).toBeDisabled();
  releaseBundle();
  expect((await bundleRequest).headers()["x-vf-profile"] ?? "default").toBe(slug);
  const downloaded = await bundleDownload;
  expect(downloaded.suggestedFilename()).toMatch(/\.zip$/);
  const stream = await downloaded.createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream) chunks.push(Buffer.from(chunk));
  const zip = Buffer.concat(chunks);
  expect(zip.subarray(0, 2).toString()).toBe("PK");
  for (const file of ["run.json", "manifest.json", "predictions.csv"]) expect(zip.includes(Buffer.from(file))).toBe(true);
  expect(zip.includes(Buffer.from("private.env"))).toBe(false);

  // The already-open detail must expose a later refusal from the real API.
  writeFileSync(path.join(runDir, "run.json"), "[]");
  await page.getByRole("button", { name: "Baixar bundle do experimento", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("Não foi possível baixar o bundle.");
});
