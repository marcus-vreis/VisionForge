import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { RunSummary } from "./RunSummary";

const render = (ready: boolean, attempted: boolean) =>
  renderToStaticMarkup(
    <RunSummary
      text="5 seeds · run it with the button below"
      ready={ready}
      reason="Needs at least 2 valid seeds to run."
      attempted={attempted}
      accent="#0f0"
    />,
  );

describe("RunSummary (ADR-059 amendment)", () => {
  it("says what the bottom button will run when the card is ready", () => {
    const html = render(true, true);
    expect(html).toContain("5 seeds");
    expect(html).not.toContain('role="alert"');
  });

  it("shows the reason, quietly, before anyone pressed the button", () => {
    const html = render(false, false);
    expect(html).toContain("Needs at least 2 valid seeds");
    expect(html).not.toContain('role="alert"');
  });

  it("turns the reason into an alert once the bottom button was pressed", () => {
    expect(render(false, true)).toContain('role="alert"');
  });
});
