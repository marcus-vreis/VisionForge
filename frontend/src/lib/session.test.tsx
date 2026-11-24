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
import { useDestHint } from "./dest-hint";
import { DEFAULT_PROFILE_INFO } from "./profile";
import {
  LOCAL_SESSION,
  ROUTE,
  bootSession,
  datasetsOffered,
  getSession,
  gigabytes,
  openStrategies,
  parseSession,
  quotaFull,
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
  quota: null,
};

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };
const render = (element: React.ReactNode) =>
  renderToStaticMarkup(<LanguageContext.Provider value={lang}>{element}</LanguageContext.Provider>);
const noop = () => {};

afterEach(() => {
  setSession(LOCAL_SESSION);
  vi.unstubAllGlobals();
});

describe("the disk quota (ADR-225)", () => {
  const GIB = 1024 ** 3;

  it("is read from the session, and anything else is no quota", () => {
    const quota = { used_bytes: 10, limit_bytes: 20 };
    expect(parseSession({ ...SERVER, quota }).quota).toEqual(quota);
    for (const raw of [undefined, null, "x", {}, { used_bytes: 1 }, { used_bytes: 1, limit_bytes: 0 }]) {
      expect(parseSession({ ...SERVER, quota: raw }).quota).toBeNull();
    }
    expect(LOCAL_SESSION.quota ?? null).toBeNull();
  });

  it("is full at the ceiling, not before", () => {
    expect(quotaFull({ used_bytes: 19, limit_bytes: 20 })).toBe(false);
    expect(quotaFull({ used_bytes: 20, limit_bytes: 20 })).toBe(true);
    expect(quotaFull(null)).toBe(false);
    expect(quotaFull(undefined)).toBe(false);
  });

  it("is written in gigabytes, in the locale's notation", () => {
    expect(gigabytes(1.5 * GIB, "pt-BR")).toBe("1,5 GB");
    expect(gigabytes(1.5 * GIB, "en-US")).toBe("1.5 GB");
    expect(gigabytes(0, "pt-BR")).toBe("0,0 GB");
  });
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

  it("the profile menu offers W&B and Comet, not local MLflow, once the profile is editable", () => {
    // What a server sends since ADR-223: the profile opens, MLflow's retry stays off.
    setSession({
      ...SERVER,
      closed: [ROUTE.profileCreate, ROUTE.mlflowRetry],
    });
    const shared = menu();
    expect(shared).toContain(pt.profile.integrations);
    expect(shared).toContain(pt.wandb.label.replaceAll("&", "&amp;"));
    expect(shared).toContain(pt.comet.label);
    expect(shared).not.toContain("data-mlflow-switch");
    expect(shared).not.toContain(pt.profile.createTitle);

    setSession(LOCAL_SESSION);
    expect(menu()).toContain("data-mlflow-switch");
  });

  it("the profile menu says how full the person's folder is", () => {
    const GIB = 1024 ** 3;
    expect(menu()).not.toContain("data-quota");

    setSession({ ...SERVER, quota: { used_bytes: 1.5 * GIB, limit_bytes: 5 * GIB } });
    const roomy = menu();
    expect(roomy).toContain(pt.server.quota("1,5 GB", "5,0 GB"));
    expect(roomy).toContain('data-quota="ok"');
    expect(roomy).not.toContain(pt.server.quotaFull);

    setSession({ ...SERVER, quota: { used_bytes: 5.2 * GIB, limit_bytes: 5 * GIB } });
    const full = menu();
    expect(full).toContain('data-quota="full"');
    expect(full).toContain(pt.server.quotaFull);
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

  it("under a run's image folder the note speaks of images, not of the dataset", () => {
    setSession(SERVER);
    const note = render(<ServerFoldersNote title={pt.server.foldersTitleRun} />);
    expect(note).toContain(pt.server.foldersTitleRun);
    expect(note).not.toContain(pt.server.foldersTitle);
    expect(note).toContain("/srv/shared");
  });

  it("a derived dataset's name says where it lands: beside its source, or in the own folder", () => {
    const Hint = () => <span>{useDestHint("beside the original")}</span>;
    expect(render(<Hint />)).toContain("beside the original");
    setSession(SERVER);
    expect(render(<Hint />)).toContain(pt.server.destHint);
    expect(render(<Hint />)).not.toContain("beside the original");
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
