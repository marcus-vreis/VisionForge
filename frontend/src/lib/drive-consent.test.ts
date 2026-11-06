import { describe, expect, it } from "vitest";
import type { TrainingEvent } from "../types/run";
import { consentAnswerer, driveConsentPrompt } from "./drive-consent";

const ask = (attempt: number, url = `https://accounts.google.com/a${attempt}`): TrainingEvent => ({
  event: "drive_consent_required",
  url,
  attempt,
  attempts: 3,
});
const remote = (stage: string): TrainingEvent => ({ event: "remote", stage });

describe("driveConsentPrompt", () => {
  it("is null before the job asks", () => {
    expect(driveConsentPrompt([])).toBeNull();
    expect(driveConsentPrompt([remote("allocated"), remote("drive_mount")])).toBeNull();
  });

  it("shows the latest page while it is unanswered", () => {
    const events = [remote("drive_mount"), ask(1), remote("drive_consent_sent"), ask(2)];
    expect(driveConsentPrompt(events)).toEqual({
      url: "https://accounts.google.com/a2",
      attempt: 2,
      attempts: 3,
    });
  });

  it("goes away once the reply is sent, the Drive mounts or the run ends", () => {
    expect(driveConsentPrompt([ask(1), remote("drive_consent_sent")])).toBeNull();
    expect(driveConsentPrompt([ask(1), remote("drive_mounted")])).toBeNull();
    expect(driveConsentPrompt([ask(1), { event: "end", total_epochs: 1 }])).toBeNull();
  });
});

describe("consentAnswerer", () => {
  it("answers when the person comes back from the consent page", () => {
    let sent = 0;
    const a = consentAnswerer(() => (sent += 1));
    a.opened();
    a.left();
    a.returned();
    expect(sent).toBe(1);
    // One answer per trip: a later focus alone says nothing.
    a.returned();
    expect(sent).toBe(1);
  });

  it("does not answer a focus that never left, nor one before the page opened", () => {
    let sent = 0;
    const a = consentAnswerer(() => (sent += 1));
    a.left();
    a.returned();
    a.opened();
    a.returned();
    expect(sent).toBe(0);
  });

  it("falls back to 'I already allowed' when the focus never comes back", () => {
    let sent = 0;
    const a = consentAnswerer(() => (sent += 1));
    a.opened();
    a.allowed();
    expect(sent).toBe(1);
  });
});
