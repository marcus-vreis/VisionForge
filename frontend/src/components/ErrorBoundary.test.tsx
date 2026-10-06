/**
 * The test setup has no DOM, so a child that throws cannot be mounted here;
 * what is checked instead is the contract React drives the boundary through
 * (the static lifecycle methods and `render`) and the notice it falls back to,
 * rendered to markup in each language. The real thing was exercised in a
 * browser when the boundary was added.
 */
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { ContentBoundary, CrashNotice, ErrorBoundary } from "./ErrorBoundary";

const inLanguage = (lang: "pt" | "en", node: React.ReactNode) =>
  renderToStaticMarkup(
    <LanguageContext.Provider
      value={{
        lang,
        t: lang === "pt" ? pt : en,
        locale: lang === "pt" ? "pt-BR" : "en-US",
        setLang: () => {},
      }}
    >
      {node}
    </LanguageContext.Provider>,
  );

describe("ErrorBoundary", () => {
  const props = { fallback: "FALLBACK", children: "CHILD" };

  it("renders its children until one of them throws", () => {
    const boundary = new ErrorBoundary(props);
    expect(boundary.state.failed).toBe(false);
    expect(boundary.render()).toBe("CHILD");
  });

  it("renders the fallback once React reports a throw", () => {
    const boundary = new ErrorBoundary(props);
    // What React does when a child throws while rendering: the error goes to
    // getDerivedStateFromError and its result is merged into the state.
    boundary.state = {
      ...boundary.state,
      ...ErrorBoundary.getDerivedStateFromError(),
    };
    expect(boundary.state.failed).toBe(true);
    expect(boundary.render()).toBe("FALLBACK");
  });

  it("tries its children again when the reset key changes, not before", () => {
    const tripped = { failed: true, resetKey: "classification" };
    expect(
      ErrorBoundary.getDerivedStateFromProps(
        { ...props, resetKey: "classification" },
        tripped,
      ),
    ).toBeNull();
    expect(
      ErrorBoundary.getDerivedStateFromProps({ ...props, resetKey: "detection" }, tripped),
    ).toEqual({ failed: false, resetKey: "detection" });
  });

  it("starts under the reset key it was given", () => {
    expect(new ErrorBoundary({ ...props, resetKey: "anomaly" }).state).toEqual({
      failed: false,
      resetKey: "anomaly",
    });
  });
});

describe("CrashNotice", () => {
  it("says what happened and offers a reload, in Portuguese", () => {
    const html = inLanguage("pt", <CrashNotice onReload={() => {}} />);
    expect(html).toContain('role="alert"');
    expect(html).toContain(pt.errorBoundary.title);
    expect(html).toContain(pt.errorBoundary.body);
    expect(html).toMatch(new RegExp(`<button[^>]*>${pt.errorBoundary.reload}</button>`));
  });

  it("says what happened and offers a reload, in English", () => {
    const html = inLanguage("en", <CrashNotice onReload={() => {}} />);
    expect(html).toContain(en.errorBoundary.title);
    expect(html).toContain(en.errorBoundary.body);
    expect(html).toMatch(new RegExp(`<button[^>]*>${en.errorBoundary.reload}</button>`));
    // None of the Portuguese leaks into the English notice.
    expect(html).not.toContain(pt.errorBoundary.reload);
  });
});

describe("ContentBoundary", () => {
  it("is invisible while nothing throws", () => {
    const html = inLanguage(
      "pt",
      <ContentBoundary resetKey="classification">
        <p>form</p>
      </ContentBoundary>,
    );
    expect(html).toBe("<p>form</p>");
  });
});
