import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { LOCAL_SESSION, pageTitle, studioLine } from "./session";

const SERVER = { ...LOCAL_SESSION, server_mode: true, user: "ana@lab.org" };

/** On a server several people share (ADR-212) the page is not "local". */
describe("what the page calls itself", () => {
  it("names the tab after where it runs, in both languages", () => {
    expect(pageTitle(pt.app, LOCAL_SESSION)).toBe(pt.app.pageTitle);
    expect(pageTitle(en.app, LOCAL_SESSION)).toBe(en.app.pageTitle);
    for (const dict of [pt, en]) {
      const title = pageTitle(dict.app, SERVER);
      expect(title).toBe(dict.app.pageTitleServer);
      expect(title).not.toMatch(/local/i);
    }
  });

  it("says 'shared' under the logo on a server, 'local' elsewhere", () => {
    expect(studioLine(false)).toBe("local ai studio");
    expect(studioLine(true)).toBe("shared ai studio");
    // The header itself needs a browser: e2e/server-mode.e2e.ts reads it.
  });
});
