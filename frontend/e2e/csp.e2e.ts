import { expect, test } from "@playwright/test";
import { enter, openStep } from "./helpers";

/**
 * The page the server delivers runs under its Content-Security-Policy and talks
 * to no host but this server (SEC-2). A policy that blocks something
 * the SPA needs shows here as a violation; a font or script loaded from another
 * site shows as a request to another host.
 */
test("o app roda sob a CSP e não fala com outro host", async ({ page, baseURL }) => {
  const own = new URL(baseURL ?? "http://127.0.0.1").host;
  const foreign: string[] = [];
  const blocked: string[] = [];

  page.on("request", (request) => {
    const url = new URL(request.url());
    if (["data:", "blob:"].includes(url.protocol)) return;
    if (url.host !== own) foreign.push(request.url());
  });
  page.on("console", (message) => {
    if (/Content.Security.Policy/i.test(message.text())) blocked.push(message.text());
  });
  await page.addInitScript(() => {
    document.addEventListener("securitypolicyviolation", (event) => {
      console.error(
        `Content-Security-Policy violation: ${event.violatedDirective} ${event.blockedURI}`,
      );
    });
  });

  const response = await page.goto("/");
  const headers = response?.headers() ?? {};
  expect(headers["content-security-policy"]).toContain("default-src 'self'");
  expect(headers["x-frame-options"]).toBe("DENY");

  // Light, dark and the theme script that picks one before the bundle loads.
  expect(await page.evaluate(() => document.documentElement.dataset.theme)).toMatch(
    /^(light|dark)$/,
  );

  await enter(page);
  for (const step of ["Treinamento", "Modelo", "Revisão", "Dados"]) {
    await openStep(page, step);
  }
  // A POST from the page itself must pass the Host/Origin guard under the
  // page's Referrer-Policy: the Origin it carries depends on that policy.
  const created = page.waitForResponse(
    (r) => r.request().method() === "POST" && new URL(r.url()).pathname.startsWith("/api/"),
  );
  await page.getByRole("button", { name: "usar dataset de exemplo" }).click();
  expect((await created).status()).toBeLessThan(400);

  expect(blocked).toEqual([]);
  expect(foreign).toEqual([]);
});
