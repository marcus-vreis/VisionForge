import { describe, expect, it, vi } from "vitest";

import {
  announce,
  canNotify,
  messageFor,
  requestPermission,
  resetTitle,
  titleFor,
} from "./run-notify";

function fakeDoc(hidden = true): Document {
  return { title: "VisionForge — Local AI Training", hidden } as Document;
}

function fakeNotification(permission: NotificationPermission, calls: unknown[] = []) {
  const ctor = function (title: string, opts?: NotificationOptions) {
    calls.push({ title, opts });
  } as unknown as typeof Notification;
  (ctor as { permission: NotificationPermission }).permission = permission;
  return ctor;
}

describe("titleFor", () => {
  it("puts the mark first, where the tab strip clips last", () => {
    expect(titleFor("completed", "coffee_v2")).toBe("✓ coffee_v2 — VisionForge");
    expect(titleFor("failed", "coffee_v2")).toBe("✗ coffee_v2 — VisionForge");
  });
});

describe("messageFor", () => {
  it("names the run and its outcome", () => {
    expect(messageFor("completed", "coffee_v2").title).toMatch(/concluído.*coffee_v2/);
    expect(messageFor("failed", "coffee_v2").title).toMatch(/falhou.*coffee_v2/);
  });

  it("prefers a real detail over the generic line", () => {
    expect(messageFor("completed", "x", "accuracy 0.87").body).toBe("accuracy 0.87");
  });
});

describe("requestPermission", () => {
  it("reports unsupported rather than throwing", async () => {
    expect(await requestPermission(undefined)).toBe("unsupported");
  });

  it("does not re-ask once the answer is sticky", async () => {
    const ctor = fakeNotification("denied");
    const ask = vi.fn();
    (ctor as unknown as { requestPermission: unknown }).requestPermission = ask;

    expect(await requestPermission(ctor)).toBe("denied");
    expect(ask).not.toHaveBeenCalled();
  });
});

describe("announce", () => {
  it("always sets the title, even with notifications denied", () => {
    // The title is the channel that carries the guarantee.
    const doc = fakeDoc();

    announce("completed", "run_a", undefined, {
      doc,
      notification: fakeNotification("denied"),
    });

    expect(doc.title).toBe("✓ run_a — VisionForge");
  });

  it("notifies when the page is hidden and permission is granted", () => {
    const calls: unknown[] = [];

    announce("completed", "run_a", "accuracy 0.87", {
      doc: fakeDoc(true),
      notification: fakeNotification("granted", calls),
      hidden: true,
    });

    expect(calls).toHaveLength(1);
  });

  it("stays silent when the researcher is already looking at the page", () => {
    // A toast for a window in front of you is noise.
    const calls: unknown[] = [];

    announce("completed", "run_a", undefined, {
      doc: fakeDoc(false),
      notification: fakeNotification("granted", calls),
      hidden: false,
    });

    expect(calls).toHaveLength(0);
  });

  it("survives a notification constructor that throws", () => {
    const ctor = function () {
      throw new Error("blocked by the OS");
    } as unknown as typeof Notification;
    (ctor as { permission: NotificationPermission }).permission = "granted";
    const doc = fakeDoc(true);

    expect(() =>
      announce("failed", "run_a", undefined, { doc, notification: ctor, hidden: true }),
    ).not.toThrow();
    expect(doc.title).toBe("✗ run_a — VisionForge");
  });
});

describe("canNotify / resetTitle", () => {
  it("is false without the API at all", () => {
    expect(canNotify(undefined)).toBe(false);
  });

  it("restores the plain title", () => {
    const doc = { title: "✓ x — VisionForge" } as Document;

    resetTitle(doc);

    expect(doc.title).toBe("VisionForge — Local AI Training");
  });
});
