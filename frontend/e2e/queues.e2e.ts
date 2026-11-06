import { expect, test, type Page, type Route } from "@playwright/test";

import { enter } from "./helpers";

// Two queues side by side (ADR-163 phase 4), in the real built SPA. The suite
// has no Colab: `/api/queue` and `/api/colab/status` answer with a local job and
// a Colab one running at once, as the server reports them, and the stop request
// is caught before it reaches the server.
const COLAB = "colab:ana@gmail.com";

const job = (run_id: string, label: string, lane: string, stop_at: string) => ({
  run_id,
  label,
  task: "classification",
  strategy: "simple",
  submitted_at: new Date().toISOString(),
  stop_at,
  profile: "default",
  profile_name: "",
  lane,
});

const local = job("local-run", "e2e_local_run", "local", "epoch");
const colab = job("colab-run", "e2e_colab_run", COLAB, "vm");
const colabNext = job("colab-next", "e2e_colab_next", COLAB, "vm");

async function serveTwoQueues(page: Page, stops: string[]): Promise<void> {
  await page.route(
    (url) => url.pathname === "/api/queue",
    (route: Route) =>
      route.fulfill({
        json: {
          active: local,
          pending: [],
          lanes: [
            { lane: "local", active: local, pending: [] },
            { lane: COLAB, active: colab, pending: [colabNext] },
          ],
        },
      }),
  );
  await page.route(
    (url) => url.pathname.startsWith("/api/queue/"),
    (route: Route) => {
      stops.push(decodeURIComponent(new URL(route.request().url()).pathname.slice("/api/queue/".length)));
      return route.fulfill({ json: { run_id: "x", status: "cancelling" } });
    },
  );
  await page.route(
    (url) => url.pathname === "/api/colab/status",
    (route: Route) =>
      route.fulfill({
        json: {
          pinned_version: "0",
          installed_version: "0",
          install_state: "idle",
          install_error: null,
          logged_in: true,
          account: "ana@gmail.com",
          login_pending: false,
        },
      }),
  );
}

test("shows the local queue and the Colab one, and stops only the chosen job", async ({ page }) => {
  const stops: string[] = [];
  await serveTwoQueues(page, stops);
  await enter(page);

  // A Colab job waiting counts in the badge like a local one.
  await page.getByRole("button", { name: "fila", exact: true }).click();

  const localQueue = page.getByRole("region", { name: "Local · esta máquina" });
  const colabQueue = page.getByRole("region", { name: "Colab · ana@gmail.com" });
  await expect(localQueue.getByText("e2e_local_run")).toBeVisible();
  await expect(colabQueue.getByText("e2e_colab_run")).toBeVisible();
  await expect(colabQueue.getByText("e2e_colab_next")).toBeVisible();
  await expect(localQueue.getByText("e2e_colab_run")).toHaveCount(0);
  // Each row says where its job runs.
  await expect(localQueue.getByText(/^local · Classificação · /)).toBeVisible();
  await expect(colabQueue.getByText(/^Colab · Classificação · /)).toHaveCount(2);

  await colabQueue.getByRole("button", { name: "■ parar" }).click();
  await expect.poll(() => stops).toEqual(["colab-run"]);
  // The local job was not asked to stop.
  expect(stops).not.toContain("local-run");
});
