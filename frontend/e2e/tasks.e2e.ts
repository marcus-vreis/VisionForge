import { expect, test, type Page } from "@playwright/test";

import { enter, fieldByLabel, openStep, train } from "./helpers";

/**
 * The first training of the four tasks that are not classification, the way
 * someone new would do it: open the tab, click "usar dataset de exemplo",
 * train a couple of epochs on the CPU and open the result.
 *
 * training.e2e.ts covers classification in depth; this file is the proof that
 * the same path works on every other tab with nothing typed but the epochs and
 * the name.
 */

// One server, one training slot.
test.describe.configure({ mode: "serial" });

interface TaskCase {
  /** The tab's label. */
  tab: string;
  /** The folder the sample dataset is created in. */
  folder: string;
  /** The labels the result must show: the task's own metrics, named for a
   *  person. They used to appear as the raw run.json keys (TEST_R2, VAL_MIOU). */
  metrics: string[];
  epochs: number;
  /** Something to pick in the Modelo step before training. */
  model?: (page: Page) => Promise<void>;
}

/** Random initialization: no weights are downloaded. */
async function withoutPretrainedWeights(page: Page): Promise<void> {
  const pretrained = page.getByRole("switch", { name: "Pesos pré-treinados", exact: true });
  if (await pretrained.isChecked()) await pretrained.click();
  await expect(pretrained).toHaveText(/desativado/i);
}

const CASES: TaskCase[] = [
  {
    tab: "Regressão",
    folder: "exemplo-regressao",
    metrics: ["R² (teste)", "R² (val)", "MSE (teste)"],
    epochs: 2,
    model: withoutPretrainedWeights,
  },
  {
    tab: "Segmentação",
    folder: "exemplo-segmentacao",
    metrics: ["mIoU (teste)", "mIoU (val)", "Melhor mIoU (val)"],
    epochs: 2,
    model: withoutPretrainedWeights,
  },
  {
    tab: "Anomalia",
    folder: "exemplo-anomalia",
    // No "(val)": anomaly has no validation split.
    metrics: ["AUROC (teste)", "AUROC (melhor época)", "Melhor AUROC"],
    epochs: 2,
  },
  {
    tab: "Detecção de Objeto",
    folder: "exemplo-deteccao",
    metrics: ["mAP@50", "mAP@50-95"],
    epochs: 1,
    model: withoutPretrainedWeights,
  },
];

async function useSampleDataset(page: Page, folder: string): Promise<void> {
  await openStep(page, "Dados");
  // A click can land while the panel re-renders and be lost; clicking again is
  // safe, since the server reuses a sample folder that exists.
  await expect(async () => {
    await page.getByRole("button", { name: "usar dataset de exemplo" }).click();
    await page.waitForFunction(
      (name) => [...document.querySelectorAll("input")].some((i) => i.value.includes(name)),
      folder,
      { timeout: 5_000 },
    );
  }).toPass({ timeout: 30_000 });
}

for (const task of CASES) {
  test(`${task.tab}: trains on the sample dataset and shows the result`, async ({ page }) => {
    const name = `e2e_${task.folder.replace("exemplo-", "")}`;
    await enter(page);
    await page.locator("[data-tour='tabs']").getByText(task.tab, { exact: true }).click();

    await useSampleDataset(page, task.folder);

    await openStep(page, "Treinamento");
    const epochs = fieldByLabel(page, "Épocas");
    await epochs.fill(String(task.epochs));
    await epochs.press("Enter");
    await expect(fieldByLabel(page, "Épocas")).toHaveValue(String(task.epochs));

    if (task.model) {
      await openStep(page, "Modelo");
      await task.model(page);
    }

    const device = page.locator("[data-tour='device']");
    await device.getByRole("button").first().click();
    await device.getByRole("button", { name: /Treinar usando processador/ }).click();

    await openStep(page, "Revisão");
    await fieldByLabel(page, "Nome do experimento").fill(name);
    await expect(page.locator("[data-tour='train']")).toBeEnabled();

    const sheet = await train(page);
    await expect(sheet.getByText("treino concluído", { exact: true })).toBeVisible({
      timeout: 240_000,
    });

    // "Ver resultados" opens the run's own page in the History.
    await sheet.getByRole("button", { name: /Ver resultados/ }).click();
    const history = page.getByTestId("history-overlay");
    await expect(history.getByRole("heading", { name, exact: true })).toBeVisible({
      timeout: 15_000,
    });
    for (const label of task.metrics) {
      await expect(history.getByText(label, { exact: true }).first()).toBeVisible();
    }
    await expect(history.getByText("Curvas por época", { exact: true })).toBeVisible();
  });
}
