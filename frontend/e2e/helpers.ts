import { expect, type Locator, type Page } from "@playwright/test";

/** Opens the GUI and gets past the first-visit welcome by typing a name. */
export async function enter(page: Page, name = "E2E"): Promise<void> {
  // The guide offers itself on a first visit and after a first training; it is
  // not what these tests drive, so it is dismissed whenever it covers the page.
  const closeGuide = page.getByRole("button", { name: "Fechar o guia" });
  await page.addLocatorHandler(closeGuide, () => closeGuide.click());
  await page.goto("/");
  // The field exists from the start but stays inert until the intro reaches the question.
  await expect(page.getByText("Qual é o seu nome?")).toBeVisible();
  const field = page.getByRole("textbox", { name: "Seu nome" });
  await expect(async () => {
    await field.fill(name, { timeout: 1_000 });
    await expect(field).toHaveValue(name, { timeout: 500 });
  }).toPass();
  await page.getByRole("button", { name: /Entrar/ }).click();
  await expect(field).toBeHidden();
}

/** The input associated with a form field's visible label. */
export function fieldByLabel(page: Page, label: string): Locator {
  return page.getByLabel(label, { exact: true }).first();
}

/** Opens a step of the task configuration from its rail (ADR-156). */
export async function openStep(page: Page, label: string): Promise<void> {
  const tab = page
    .getByRole("navigation", { name: "Etapas da configuração" })
    .getByRole("button", { name: label });
  await tab.click();
  await expect(tab).toHaveAttribute("aria-current", "step");
}

/**
 * Prepares a classification run on the sample dataset, on the CPU.
 *
 * The device is chosen in the UI even though the server is started without a
 * visible GPU: choosing it is part of the flow under test. It ends on the
 * Revisão step, the only one where Treinar is unlocked (ADR-156).
 */
export async function prepareClassification(page: Page, name: string, epochs: number): Promise<void> {
  await openStep(page, "Dados");
  // A click can land while the panel re-renders and be lost (seen once in four
  // runs). Clicking again is safe: the server reuses a sample folder that exists.
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

  await openStep(page, "Treinamento");
  const epochField = fieldByLabel(page, "Épocas");
  await epochField.fill(String(epochs));
  await epochField.press("Enter");
  await expect(fieldByLabel(page, "Épocas")).toHaveValue(String(epochs));

  await openStep(page, "Modelo");
  // Random initialization keeps this real training independent of the network
  // and of any ImageNet weights cached on the developer's machine.
  const pretrained = page.getByRole("switch", { name: "Pesos pré-treinados", exact: true });
  if (await pretrained.isChecked()) {
    await pretrained.click();
  }
  await expect(pretrained).toHaveText(/desativado/i);

  const device = page.locator("[data-tour='device']");
  await device.getByRole("button").first().click();
  await device.getByRole("button", { name: /Treinar usando processador/ }).click();

  // The dataset is set, but only the review unlocks the run.
  await expect(page.locator("[data-tour='train']")).toBeDisabled();
  await openStep(page, "Revisão");
  await fieldByLabel(page, "Nome do experimento").fill(name);
  await expect(page.locator("[data-tour='train']")).toBeEnabled();
}

/**
 * Opens the History from the bottom bar and narrows it to one run.
 *
 * The overlay shows the queue's job id (`<name>_<timestamp>`), which is not the
 * id the History lists, so the run is found by its experiment name.
 */
export async function findInHistory(page: Page, name: string): Promise<Locator> {
  await page.locator("[data-tour='history']").click();
  await expect(page.getByText("Treinamentos recentes")).toBeVisible();
  await page.getByPlaceholder(/buscar por nome, arquitetura ou run_id/).fill(name);
  // Inside the overlay: the step rail behind it also shows the dataset's name.
  const card = page.getByTestId("history-overlay").getByText(name, { exact: true }).locator("xpath=ancestor::div[.//*[contains(., 'epoch')]][1]");
  await expect(card).toBeVisible();
  return card;
}

/** Presses the centre "Treinar" button and returns the training overlay. */
export async function train(page: Page): Promise<Locator> {
  await page.locator("[data-tour='train']").click();
  const sheet = page.locator("[data-tour='training-sheet']");
  await expect(sheet).toBeVisible();
  return sheet;
}
