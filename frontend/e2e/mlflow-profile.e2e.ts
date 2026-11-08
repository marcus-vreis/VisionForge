import { rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";
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
// or the next tests open on the profile list instead of the name question.
test.afterEach(async ({ request }) => {
  await request.put("/api/profiles/default", { data: { settings: { mlflow: { enabled: false } } } });
  if (createdSlug) rmSync(path.join(work, "outputs", "profiles", createdSlug), { recursive: true, force: true });
  createdSlug = null;
});

test("profile MLflow toggles persist independently across reloads", async ({ page, request }) => {
  const created = await request.post("/api/profiles", { data: { name: "MLflow E2E" } });
  expect(created.status()).toBe(201);
  const named: Profile = await created.json();
  createdSlug = named.slug;
  const listed: Profile[] = (await (await request.get("/api/profiles")).json()).profiles;
  const fallback = listed.find((p) => p.is_default);
  expect(fallback).toBeDefined();
  // Earlier tests may already have given the default profile a name.
  const defaultLabel = fallback?.name ? pt.profile.defaultNamed(fallback.name) : pt.profile.defaultName;

  await page.goto("/");
  await expect(page.getByText(pt.profile.who, { exact: true })).toBeVisible();
  const rows = page.getByRole("group", { name: pt.profile.listLabel }).locator(":scope > div");
  const defaultRow = rows.filter({ has: page.getByRole("button", { name: defaultLabel, exact: true }) });
  const namedRow = rows.filter({ has: page.getByRole("button", { name: named.name, exact: true }) });
  const defaultToggle = defaultRow.getByRole("checkbox", { name: pt.mlflow.label });
  const namedToggle = namedRow.getByRole("checkbox", { name: pt.mlflow.label });
  await expect(defaultToggle).not.toBeChecked();
  await expect(namedToggle).not.toBeChecked();

  const saved = page.waitForResponse((res) => res.url().endsWith(`/api/profiles/${named.slug}`) && res.request().method() === "PUT");
  await namedToggle.click();
  expect((await saved).status()).toBe(200);
  await expect(namedToggle).toBeChecked();
  await expect(defaultToggle).not.toBeChecked();
  await page.reload();
  await expect(namedToggle).toBeChecked();
  await expect(defaultToggle).not.toBeChecked();

  const defaultSaved = page.waitForResponse((res) => res.url().endsWith("/api/profiles/default") && res.request().method() === "PUT");
  await defaultToggle.click();
  expect((await defaultSaved).status()).toBe(200);
  await expect(defaultToggle).toBeChecked();
  const namedSaved = page.waitForResponse((res) => res.url().endsWith(`/api/profiles/${named.slug}`) && res.request().method() === "PUT");
  await namedToggle.click();
  expect((await namedSaved).status()).toBe(200);
  await page.reload();
  await expect(defaultToggle).toBeChecked();
  await expect(namedToggle).not.toBeChecked();

  const profiles: Profile[] = (await (await request.get("/api/profiles")).json()).profiles;
  expect(profiles.find((p) => p.slug === "default")?.settings.mlflow.enabled).toBe(true);
  expect(profiles.find((p) => p.slug === named.slug)?.settings.mlflow.enabled).toBe(false);
});
