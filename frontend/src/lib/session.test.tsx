import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchSession } from "../api/client";
import { BottomBar } from "../components/BottomBar";
import { ProfileMenu } from "../components/ProfileMenu";
import { SampleDatasetButton } from "../components/SampleDatasetButton";
import { ServerFoldersNote } from "../components/ServerFoldersNote";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import { offeredGuides } from "./guides";
import { DEFAULT_PROFILE_INFO } from "./profile";
import {
  LOCAL_SESSION,
  ROUTE,
  bootSession,
  datasetsOffered,
  getSession,
  openStrategies,
  parseSession,
  routeOpen,
  setSession,
  strategyRoute,
  type Session,
} from "./session";
import { tourSteps } from "./tour";

const SERVER: Session = {
  server_mode: true,
  user: "ana@lab.org",
  profile: "ana-lab-org-0123456789",
  dataset_dirs: ["/srv/vf/outputs/profiles/ana-lab-org-0123456789/datasets", "/srv/shared"],
  closed: [
    ROUTE.pickFolder,
    ROUTE.sampleDataset,
    ROUTE.datasetDownload,
    ROUTE.datasetLab,
    ROUTE.profileCreate,
    ROUTE.profileUpdate,
    "POST /api/detection/sweep",
    "POST /api/detection/replicates",
    "POST /api/classification/replicates",
  ],
  closed_blocks: ["cross_validation", "grid_search", "random_search"],
};

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };
const render = (element: React.ReactNode) =>
  renderToStaticMarkup(<LanguageContext.Provider value={lang}>{element}</LanguageContext.Provider>);
const noop = () => {};

afterEach(() => {
  setSession(LOCAL_SESSION);
  vi.unstubAllGlobals();
});

describe("parseSession", () => {
  it("reads what a shared server sends", () => {
    expect(parseSession({ ...SERVER })).toEqual(SERVER);
  });

  it.each([null, undefined, "x", 3, {}, { server_mode: false, closed: ["GET /api/runs"] }])(
    "reads %j as a local install, with nothing closed",
    (raw) => {
      expect(parseSession(raw)).toEqual(LOCAL_SESSION);
    },
  );

  it("drops what is not text from the lists", () => {
    const parsed = parseSession({ server_mode: true, closed: ["a", 1, null], dataset_dirs: "x" });
    expect(parsed.closed).toEqual(["a"]);
    expect(parsed.dataset_dirs).toEqual([]);
    expect(parsed.user).toBeNull();
  });
});

describe("fetchSession", () => {
  const reply = (status: number, json: unknown) =>
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify(json), {
        status,
        headers: { "Content-Type": "application/json" },
      }),
    );

  it("asks the server", async () => {
    const fetchMock = reply(200, SERVER);
    vi.stubGlobal("fetch", fetchMock);
    expect(await fetchSession()).toEqual(SERVER);
    expect(String(fetchMock.mock.calls[0][0])).toBe("/api/session");
  });

  it("reads an older server, which has no such route, as a local install", async () => {
    vi.stubGlobal("fetch", reply(404, { detail: "Not Found" }));
    expect(await fetchSession()).toEqual(LOCAL_SESSION);
  });

  it("reads a server that cannot be reached as a local install", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));
    expect(await fetchSession()).toEqual(LOCAL_SESSION);
  });
});

describe("what a session offers", () => {
  it("a local install offers every route", () => {
    for (const route of Object.values(ROUTE)) expect(routeOpen(LOCAL_SESSION, route)).toBe(true);
  });

  it("a shared server closes what it lists, and only that", () => {
    expect(routeOpen(SERVER, ROUTE.pickFolder)).toBe(false);
    expect(routeOpen(SERVER, ROUTE.runTest)).toBe(true);
    expect(routeOpen(SERVER, null)).toBe(true);
  });

  it("names the route of a strategy that has one of its own", () => {
    expect(strategyRoute("regression", "cv")).toBe("POST /api/regression/cv");
    expect(strategyRoute("anomaly", "replicatedComparison")).toBe(
      "POST /api/anomaly/replicated-comparison",
    );
    expect(strategyRoute("detection", "simple")).toBeNull();
    // Classification sends these inside its run.
    expect(strategyRoute("classification", "grid_search")).toBeNull();
  });

  it("keeps the strategies the server can run", () => {
    const all = ["simple", "sweep", "replicates", "replicatedComparison"].map((value) => ({ value }));
    expect(openStrategies(LOCAL_SESSION, "detection", all)).toEqual(all);
    expect(openStrategies(SERVER, "detection", all).map((s) => s.value)).toEqual([
      "simple",
      "replicatedComparison",
    ]);
    const blocks = ["classification", "cross_validation", "grid_search", "replicates"].map(
      (value) => ({ value }),
    );
    expect(openStrategies(SERVER, "classification", blocks).map((s) => s.value)).toEqual([
      "classification",
    ]);
    expect(
      openStrategies({ ...SERVER, closed_blocks: [] }, "classification", blocks).map((s) => s.value),
    ).toEqual(["classification", "cross_validation", "grid_search"]);
    // A block is classification's: another task's strategy of the same name stays.
    expect(
      openStrategies({ ...SERVER, closed_blocks: ["simple"] }, "regression", [{ value: "simple" }]),
    ).toHaveLength(1);
  });

  it("has a datasets sheet while the download or the Laboratory answers", () => {
    expect(datasetsOffered(LOCAL_SESSION)).toBe(true);
    expect(datasetsOffered(SERVER)).toBe(false);
    expect(
      datasetsOffered({ ...SERVER, closed: SERVER.closed.filter((r) => r !== ROUTE.datasetLab) }),
    ).toBe(true);
  });

  it("drops the tour's stop on a datasets sheet that is not there", () => {
    expect(tourSteps(pt).map((s) => s.anchor)).toContain("datasets");
    expect(tourSteps(pt, false).map((s) => s.anchor)).not.toContain("datasets");
    expect(tourSteps(pt, false)).toHaveLength(tourSteps(pt).length - 1);
  });

  it("offers the first-training guide only where the sample dataset can be made", () => {
    expect(offeredGuides(LOCAL_SESSION).map((g) => g.id)).toEqual(["tour", "firstTraining"]);
    expect(offeredGuides(SERVER).map((g) => g.id)).toEqual(["tour"]);
  });
});

describe("the screen on a shared server", () => {
  const menu = () =>
    render(
      <ProfileMenu
        name="Ana"
        profile={{ ...DEFAULT_PROFILE_INFO, name: "Ana" }}
        shared={false}
        onProfileSaved={noop}
        onChangeName={noop}
        onSwitchProfile={noop}
        onNewProfile={noop}
        initialOpen
      />,
    );

  const bar = () =>
    render(
      <BottomBar
        onHistory={noop}
        onDatasets={noop}
        onQueue={noop}
        onTrain={noop}
        disabled={false}
        historyCount={0}
        queuedCount={0}
        serverRunning={false}
        selection={{ kind: "cpu", gpu_ids: [] }}
        onSelectionChange={noop}
        isRunning={false}
      />,
    );

  it("keeps the store", () => {
    setSession(SERVER);
    expect(getSession()).toBe(SERVER);
  });

  it("the profile menu names the account and offers no other profile", () => {
    const local = menu();
    expect(local).toContain(pt.profile.createTitle);
    expect(local).toContain(pt.profile.integrations);
    expect(local).not.toContain("data-server-account");

    setSession(SERVER);
    const shared = menu();
    expect(shared).toContain(pt.server.account("ana@lab.org"));
    expect(shared).toContain(pt.profile.folder("ana-lab-org-0123456789"));
    expect(shared).toContain(pt.server.menuNote);
    expect(shared).toContain(pt.header.changeName);
    expect(shared).not.toContain(pt.profile.createTitle);
    expect(shared).not.toContain(pt.profile.switchProfile);
    expect(shared).not.toContain(pt.profile.integrations);
  });

  it("the sample dataset button is gone", () => {
    const button = () => render(<SampleDatasetButton task="classification" onApply={noop} />);
    expect(button()).toContain('data-sample-task="classification"');
    setSession(SERVER);
    expect(button()).toBe("");
  });

  it("the dataset field says which folders the server accepts", () => {
    expect(render(<ServerFoldersNote />)).toBe("");
    setSession(SERVER);
    const note = render(<ServerFoldersNote />);
    expect(note).toContain(pt.server.foldersTitle);
    expect(note).toContain("/srv/shared");
    expect(note).toContain(pt.server.folderOwn);
    expect(note).toContain(pt.server.folderShared);
    expect(note.indexOf("ana-lab-org")).toBeLessThan(note.indexOf("/srv/shared"));
  });

  it("the datasets button is gone with nothing behind it", () => {
    expect(bar()).toContain('data-tour="datasets"');
    setSession(SERVER);
    const shared = bar();
    expect(shared).not.toContain('data-tour="datasets"');
    expect(shared).toContain('data-tour="history"');
  });
});

describe("bootSession", () => {
  it("applies the session before the page is drawn", async () => {
    await bootSession(() => Promise.resolve(SERVER));
    expect(getSession()).toBe(SERVER);
  });

  it("does not hold the page for a slow server, and applies the answer when it comes", async () => {
    vi.useFakeTimers();
    try {
      let answer: (session: Session) => void = () => {};
      const slow = new Promise<Session>((resolve) => {
        answer = resolve;
      });
      const booted = bootSession(() => slow, 3000);
      await vi.advanceTimersByTimeAsync(3000);
      await booted;
      expect(getSession()).toBe(LOCAL_SESSION);
      answer(SERVER);
      await vi.advanceTimersByTimeAsync(0);
      expect(getSession()).toBe(SERVER);
    } finally {
      vi.useRealTimers();
    }
  });
});
