import { rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
import { pt } from "../src/i18n/pt";

// serve.mjs runs the server from this folder; profiles live under its outputs/.
const work = path.join(path.dirname(fileURLToPath(import.meta.url)), ".work");

interface Profile {
  slug: string;
  name: string;
  is_default: boolean;
  settings: { mlflow: { enabled: boolean } };
}

let createdSlug: string | null = null;

// The suite shares one server: leave no second profile and no MLflow flag behind,
// or the next tests open on the profile list instead of entering directly.
test.afterEach(async ({ request }) => {
  await request.put("/api/profiles/default", { data: { settings: { mlflow: { enabled: false } } } });
  if (createdSlug) rmSync(path.join(work, "outputs", "profiles", createdSlug), { recursive: true, force: true });
  createdSlug = null;
});

/** With two profiles the list shows on every load: pick one by its name. */
async function pick(page: Page, name: string): Promise<void> {
  await expect(page.getByText(pt.profile.who, { exact: true })).toBeVisible();
  const option = page
    .getByRole("group", { name: pt.profile.listLabel })
    .getByRole("button", { name, exact: true });
  await expect(option).toBeVisible();
  await option.click();
  await expect(page.getByText(pt.profile.who, { exact: true })).toBeHidden();
}

/** The MLflow switch of the profile in use, in the header's profile menu. */
async function mlflowSwitch(page: Page) {
  const chip = page.getByTitle(pt.profile.menuTitle);
  await chip.click();
  const menu = page.getByRole("dialog", { name: pt.profile.menuTitle });
  await expect(menu).toBeVisible();
  return menu.getByRole("switch", { name: pt.mlflow.label });
}

async function closeMenu(page: Page): Promise<void> {
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog", { name: pt.profile.menuTitle })).toBeHidden();
}

test("profile MLflow switches persist independently across reloads", async ({ page, request }) => {
  // Choosing a profile with no name asks for one; give the default its name first.
  const before: Profile[] = (await (await request.get("/api/profiles")).json()).profiles;
  const fallback = before.find((p) => p.is_default);
  expect(fallback).toBeDefined();
  if (!fallback?.name || fallback.name === "default") {
    expect((await request.put("/api/profiles/default", { data: { name: "E2E" } })).status()).toBe(200);
  }
  const listed: Profile[] = (await (await request.get("/api/profiles")).json()).profiles;
  const defaultName = listed.find((p) => p.is_default)?.name ?? "";
  // No "(padrão)" suffix: the default is listed by its name like any profile.
  expect(defaultName).not.toBe("");

  const created = await request.post("/api/profiles", { data: { name: "MLflow E2E" } });
  expect(created.status()).toBe(201);
  const named: Profile = await created.json();
  createdSlug = named.slug;

  // The guide offers itself after a first entry; it is not what this test drives.
  const closeGuide = page.getByRole("button", { name: "Fechar o guia" });
  await page.addLocatorHandler(closeGuide, () => closeGuide.click());
  await page.goto("/");
  await pick(page, named.name);
  let toggle = await mlflowSwitch(page);
  await expect(toggle).not.toBeChecked();
  const saved = page.waitForResponse((res) => res.url().endsWith(`/api/profiles/${named.slug}`) && res.request().method() === "PUT");
  await toggle.click();
  expect((await saved).status()).toBe(200);
  await expect(toggle).toBeChecked();
  await closeMenu(page);

  // A reload with two profiles shows the list again, the last one preselected.
  await page.reload();
  await expect(page.getByText(pt.profile.who, { exact: true })).toBeVisible();
  await expect(
    page.getByRole("group", { name: pt.profile.listLabel }).getByRole("button", { name: named.name, exact: true }),
  ).toHaveAttribute("aria-current", "true");
  await pick(page, defaultName);
  toggle = await mlflowSwitch(page);
  await expect(toggle).not.toBeChecked();
  const defaultSaved = page.waitForResponse((res) => res.url().endsWith("/api/profiles/default") && res.request().method() === "PUT");
  await toggle.click();
  expect((await defaultSaved).status()).toBe(200);
  await expect(toggle).toBeChecked();
  await closeMenu(page);

  await page.reload();
  await pick(page, named.name);
  toggle = await mlflowSwitch(page);
  await expect(toggle).toBeChecked();
  const namedSaved = page.waitForResponse((res) => res.url().endsWith(`/api/profiles/${named.slug}`) && res.request().method() === "PUT");
  await toggle.click();
  expect((await namedSaved).status()).toBe(200);
  await expect(toggle).not.toBeChecked();
  await closeMenu(page);

  await page.reload();
  await pick(page, defaultName);
  await expect(await mlflowSwitch(page)).toBeChecked();

  const profiles: Profile[] = (await (await request.get("/api/profiles")).json()).profiles;
  expect(profiles.find((p) => p.slug === "default")?.settings.mlflow.enabled).toBe(true);
  expect(profiles.find((p) => p.slug === named.slug)?.settings.mlflow.enabled).toBe(false);
});
