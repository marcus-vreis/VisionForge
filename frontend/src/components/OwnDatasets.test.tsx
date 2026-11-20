import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it } from "vitest";

import type { OwnDataset } from "../api/client";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import { LOCAL_SESSION, diskSize, setSession, type Session } from "../lib/session";
import { DatasetsOverlay } from "./DatasetsOverlay";
import { OwnDatasetsView } from "./OwnDatasets";

const GIB = 1024 ** 3;
const SERVER: Session = {
  ...LOCAL_SESSION,
  server_mode: true,
  user: "ana@lab.org",
  profile: "ana",
  quota: { used_bytes: 5.2 * GIB, limit_bytes: 5 * GIB },
};
const BEANS: OwnDataset = { name: "beans", path: "/srv/ana/datasets/beans", size_bytes: 3 * GIB };
const APPLES: OwnDataset = {
  name: "apples",
  path: "/srv/ana/datasets/apples",
  size_bytes: 12.3 * 1024 ** 2,
};

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };
const noop = () => {};

function view(fields: Partial<Parameters<typeof OwnDatasetsView>[0]> = {}): string {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={lang}>
      <OwnDatasetsView
        load="ready"
        datasets={[APPLES, BEANS]}
        asking={null}
        deleting={null}
        error={null}
        notice={null}
        onAsk={noop}
        onDelete={noop}
        {...fields}
      />
    </LanguageContext.Provider>,
  );
}

afterEach(() => setSession(LOCAL_SESSION));

describe("the person's own datasets on a shared server (ADR-227)", () => {
  it("lists each one with its size and path, and a delete button of its own", () => {
    const markup = view();
    expect(markup).toContain("12,3 MB");
    expect(markup).toContain("3,0 GB");
    expect(markup).toContain(BEANS.path);
    expect(markup).toContain(`aria-label="${pt.ownDatasets.deleteLabel("beans")}"`);
    expect(markup).toContain(`aria-label="${pt.ownDatasets.deleteLabel("apples")}"`);
    expect(markup).not.toContain(pt.ownDatasets.confirm);
  });

  it("shows the folder's use of the quota when the server has one", () => {
    expect(view()).not.toContain("data-quota");
    setSession(SERVER);
    const markup = view();
    expect(markup).toContain(pt.server.quota("5,2 GB", "5,0 GB"));
    expect(markup).toContain('data-quota="full"');
  });

  it("says so while loading, when the folder is empty and when it cannot be read", () => {
    expect(view({ load: "loading", datasets: [] })).toContain(pt.ownDatasets.loading);
    expect(view({ datasets: [] })).toContain(pt.ownDatasets.empty);
    const failed = view({ load: "failed", datasets: [] });
    expect(failed).toContain(pt.ownDatasets.loadFailed);
    expect(failed).toContain('role="alert"');
  });

  it("asks in the row before deleting, and only in that row", () => {
    const markup = view({ asking: "beans" });
    expect(markup).toContain(pt.ownDatasets.confirm);
    expect(markup).toContain(pt.ownDatasets.confirmYes);
    expect(markup).toContain(pt.ownDatasets.confirmNo);
    // The other row keeps its delete button; the asking one trades it.
    expect(markup).toContain(`aria-label="${pt.ownDatasets.deleteLabel("apples")}"`);
    expect(markup).not.toContain(`aria-label="${pt.ownDatasets.deleteLabel("beans")}"`);
  });

  it("holds every button while one dataset is being deleted", () => {
    const markup = view({ asking: "beans", deleting: "beans" });
    expect(markup).toContain(pt.ownDatasets.deleting);
    expect(markup.match(/<button[^>]*disabled/g)).toHaveLength(3);
  });

  it("shows what was deleted and the server's reason when it refuses", () => {
    expect(view({ notice: pt.ownDatasets.deleted("beans") })).toContain(
      pt.ownDatasets.deleted("beans"),
    );
    const reason = "Você tem um treino rodando ou na fila.";
    const refused = view({ error: reason });
    expect(refused).toContain(reason);
    expect(refused).toContain('role="alert"');
  });
});

describe("the datasets sheet", () => {
  const sheet = () =>
    renderToStaticMarkup(
      <LanguageContext.Provider value={lang}>
        <DatasetsOverlay open onClose={noop} />
      </LanguageContext.Provider>,
    );

  it("offers 'my datasets' only on a shared server", () => {
    expect(sheet()).not.toContain(pt.datasetsOverlay.tabMine);
    setSession(SERVER);
    expect(sheet()).toContain(pt.datasetsOverlay.tabMine);
    // An older server that lists the route as closed does not offer it.
    setSession({ ...SERVER, closed: ["GET /api/datasets/own"] });
    expect(sheet()).not.toContain(pt.datasetsOverlay.tabMine);
  });
});

describe("diskSize", () => {
  it("picks the unit that reads best", () => {
    expect(diskSize(12.3 * 1024 ** 2, "pt-BR")).toBe("12,3 MB");
    expect(diskSize(GIB, "pt-BR")).toBe("1,0 GB");
    // A small dataset is not "0,0 MB".
    expect(diskSize(340 * 1024, "pt-BR")).toBe("340,0 KB");
    expect(diskSize(0, "en-US")).toBe("0.0 KB");
  });
});
