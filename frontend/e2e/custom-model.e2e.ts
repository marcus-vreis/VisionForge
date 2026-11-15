import { expect, test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { enter, openStep, train } from "./helpers";

/**
 * A model of the researcher's own (docs/custom/MODELS.md), used from the
 * interface: a file in `user_models/`, a YAML that names it in
 * `model.custom_model`, imported in the Revisão step.
 *
 * The form has no field for it, and it used to train the model while every
 * screen still said resnet18. The screens now name the model that trains.
 */

const here = path.dirname(fileURLToPath(import.meta.url));
// The server's work folder (serve.mjs): `user_models/` is read from its cwd.
const work = path.join(here, ".work");

const MODEL_FILE = `import torch.nn as nn

from visionforge.models.registry import register_model


@register_model("my_net")
def build(num_outputs: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, num_outputs),
    )
`;

test("a YAML that names a custom model trains it and every screen says so", async ({ page }) => {
  mkdirSync(path.join(work, "user_models"), { recursive: true });
  writeFileSync(path.join(work, "user_models", "my_net.py"), MODEL_FILE);

  await enter(page);
  await page.locator("[data-tour='tabs']").getByText("Classificação", { exact: true }).click();
  await openStep(page, "Dados");
  await expect(async () => {
    await page.getByRole("button", { name: "usar dataset de exemplo" }).click();
    await page.waitForFunction(
      () =>
        [...document.querySelectorAll("input")].some((i) =>
          i.value.includes("exemplo-classificacao"),
        ),
      null,
      { timeout: 5_000 },
    );
  }).toPass({ timeout: 30_000 });
  const baseDir = await page
    .locator("input")
    .evaluateAll(
      (inputs) =>
        (inputs as HTMLInputElement[]).find((i) => i.value.includes("exemplo-classificacao"))!
          .value,
    );

  const config = path.join(work, "custom_model.yaml");
  writeFileSync(
    config,
    [
      "schema_version: 2",
      "name: e2e_custom_model",
      "task: binary",
      "model:",
      "  custom_model: my_net",
      "  num_classes: 1",
      "training:",
      "  epochs: 1",
      "  batch_size: 4",
      "data:",
      `  base_dir: '${baseDir.replaceAll("\\", "/")}'`,
      "  num_workers: 0",
      "device:",
      "  kind: cpu",
      "",
    ].join("\n"),
  );

  await openStep(page, "Revisão");
  await page.locator('input[type="file"][accept=".yaml,.yml"]').setInputFiles(config);
  await expect(page.getByText("custom_model.yaml importado")).toBeVisible();

  // The review names the model that will train, not the builtin default.
  const review = page.locator("main");
  await expect(review.getByText("Modelo próprio", { exact: true })).toBeVisible();
  await expect(review.getByText("my_net", { exact: true }).first()).toBeVisible();

  // So does the Modelo step, with the way back to a builtin architecture.
  await openStep(page, "Modelo");
  const note = page.locator("[data-other-model='custom_model']");
  await expect(note).toContainText("my_net");
  await expect(note.getByRole("button", { name: "Usar a arquitetura abaixo" })).toBeVisible();

  await openStep(page, "Revisão");
  const sheet = await train(page);
  await expect(sheet.getByText("treino concluído", { exact: true })).toBeVisible({
    timeout: 120_000,
  });

  // The History lists the run under the model it trained.
  await sheet.getByRole("button", { name: /Ver resultados/ }).click();
  const history = page.getByTestId("history-overlay");
  await expect(history.getByRole("heading", { name: "e2e_custom_model", exact: true })).toBeVisible({
    timeout: 15_000,
  });
  await page.keyboard.press("Escape");
  await expect(history.getByText("my_net", { exact: true }).first()).toBeVisible();
  await expect(history.getByText("resnet18", { exact: true })).toHaveCount(0);
});
