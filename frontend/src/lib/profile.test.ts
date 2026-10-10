import { afterEach, describe, expect, it, vi } from "vitest";

import {
  DEFAULT_PROFILE_INFO,
  activeProfileSlug,
  canCreateProfile,
  clearProfile,
  hasNamedProfiles,
  isProfileStorageKey,
  isValidProfileSlug,
  jobProfileLabel,
  needsProfileChoice,
  normalizeProfileName,
  pickStoredProfile,
  profileAfterStorageChange,
  profileHeaders,
  profileIsTheName,
  queueShowsProfiles,
  readProfile,
  saveProfile,
  slugifyProfile,
  type ProfileInfo,
} from "./profile";

const ana: ProfileInfo = { slug: "ana", name: "Ana", is_default: false };
const bob: ProfileInfo = { slug: "bob", name: "Bob", is_default: false };

/** A localStorage that remembers, for the tests that need storage. */
function stubStorage(initial: Record<string, string> = {}) {
  const data = new Map(Object.entries(initial));
  vi.stubGlobal("localStorage", {
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
    removeItem: (k: string) => void data.delete(k),
  });
  return data;
}

afterEach(() => vi.unstubAllGlobals());

// The same table as tests/gui/test_profiles.py: the server is the authority and
// this only previews its folder, so the two must agree.
describe("slugifyProfile", () => {
  it.each([
    ["Ana", "ana"],
    ["João Silva", "joao-silva"],
    ["  Ana   Maria  ", "ana-maria"],
    ["ana_maria", "ana_maria"],
    ["Ünïcödé", "unicode"],
    ["---a---", "a"],
    ["a/b\\c", "a-b-c"],
    ["../x", "x"],
    ["Lab 2 (GPU)", "lab-2-gpu"],
    ["!!!", ""],
    ["李雷", ""],
    ["", ""],
  ])("%j -> %j", (name, slug) => {
    expect(slugifyProfile(name)).toBe(slug);
  });

  it("caps at 40 characters without a dangling dash", () => {
    expect(slugifyProfile("a".repeat(39) + " b")).toBe("a".repeat(39));
    expect(slugifyProfile("x".repeat(80))).toHaveLength(40);
  });
});

describe("isValidProfileSlug", () => {
  it.each(["ana", "a", "ana-2", "ana_2", "0", "x".repeat(40)])("accepts %j", (s) => {
    expect(isValidProfileSlug(s)).toBe(true);
  });

  it.each([
    "",
    "Ana",
    "../x",
    "..",
    ".",
    "a/b",
    "a\\b",
    "a.b",
    "a b",
    "-a",
    "_a",
    "x".repeat(41),
    "joão",
    "default",
    "con",
    "nul",
    "com0",
    "com1",
    "lpt0",
    "lpt9",
  ])("refuses %j", (s) => {
    expect(isValidProfileSlug(s)).toBe(false);
  });
});

describe("canCreateProfile", () => {
  it("is true only when the name leaves a usable folder", () => {
    expect(canCreateProfile("Ana Souza")).toBe(true);
    expect(canCreateProfile("!!!")).toBe(false);
    expect(canCreateProfile("")).toBe(false);
    expect(canCreateProfile("default")).toBe(false);
    expect(canCreateProfile("NUL")).toBe(false);
  });
});

describe("normalizeProfileName", () => {
  it("collapses spaces and caps at 60", () => {
    expect(normalizeProfileName("  Ana \t  Maria ")).toBe("Ana Maria");
    expect(normalizeProfileName("n".repeat(100))).toHaveLength(60);
  });
});

describe("profileHeaders", () => {
  it("names the profile for a named one", () => {
    expect(profileHeaders("ana")).toEqual({ "X-VF-Profile": "ana" });
  });

  it.each([null, undefined, "", "default", "../x", "Ana", "con"])(
    "sends nothing for %j",
    (slug) => {
      expect(profileHeaders(slug)).toEqual({});
    },
  );
});

describe("the stored choice", () => {
  it("is null, and harmless to write, where there is no storage", () => {
    // The tests run in node: there is no localStorage at all.
    expect(readProfile()).toBeNull();
    expect(() => saveProfile({ slug: "ana", name: "Ana" })).not.toThrow();
    expect(() => clearProfile()).not.toThrow();
    expect(activeProfileSlug()).toBe("default");
  });

  it("round-trips the slug and the display name", () => {
    stubStorage();

    saveProfile({ slug: "joao-silva", name: "João Silva" });

    expect(readProfile()).toEqual({ slug: "joao-silva", name: "João Silva" });
    expect(activeProfileSlug()).toBe("joao-silva");
  });

  it("falls back to the slug when no name was kept", () => {
    stubStorage({ "vf.profile": "ana" });

    expect(readProfile()).toEqual({ slug: "ana", name: "ana" });
  });

  it("is forgotten by clearProfile", () => {
    stubStorage({ "vf.profile": "ana", "vf.profile.name": "Ana" });

    clearProfile();

    expect(readProfile()).toBeNull();
    expect(activeProfileSlug()).toBe("default");
  });
});

describe("who is asked", () => {
  const listed = [DEFAULT_PROFILE_INFO, ana, bob];

  it("finds the stored profile in the list, with the server's current name", () => {
    const renamed = { ...ana, name: "Ana Souza" };

    expect(pickStoredProfile({ slug: "ana", name: "Ana" }, [DEFAULT_PROFILE_INFO, renamed])).toBe(
      renamed,
    );
  });

  it("does not count a profile whose folder is gone", () => {
    expect(pickStoredProfile({ slug: "gone", name: "Gone" }, listed)).toBeNull();
    expect(pickStoredProfile(null, listed)).toBeNull();
  });

  it("asks only where there are profiles to choose from and none was chosen", () => {
    expect(needsProfileChoice(listed, null)).toBe(true);
    expect(needsProfileChoice(listed, ana)).toBe(false);
    // A one-person install never sees the question.
    expect(needsProfileChoice([DEFAULT_PROFILE_INFO], null)).toBe(false);
    expect(needsProfileChoice([], null)).toBe(false);
  });

  it("tells a server with profiles from one without", () => {
    expect(hasNamedProfiles([DEFAULT_PROFILE_INFO])).toBe(false);
    expect(hasNamedProfiles(listed)).toBe(true);
  });
});

describe("the header", () => {
  it("lets the profile chip stand for the name chip when they say the same thing", () => {
    expect(profileIsTheName(ana, "Ana")).toBe(true);
    // A name past the chip's 32 characters is compared as it was stored.
    const long: ProfileInfo = { slug: "x", name: "N".repeat(50), is_default: false };
    expect(profileIsTheName(long, "N".repeat(32))).toBe(true);
  });

  it("keeps both when the names differ, for the default profile, or with no profile", () => {
    expect(profileIsTheName(ana, "Marcus")).toBe(false);
    expect(profileIsTheName(DEFAULT_PROFILE_INFO, "default")).toBe(false);
    expect(profileIsTheName(null, "Ana")).toBe(false);
    expect(profileIsTheName(ana, "")).toBe(false);
  });
});

// localStorage is shared by every tab, and the API client reads it on each call,
// so a switch made in one tab already moves the others' requests; these decide
// when the other tabs' screens must follow (the `storage` event).
describe("a profile switched in another tab", () => {
  it("cares about the profile keys, and about a wiped storage", () => {
    expect(isProfileStorageKey("vf.profile")).toBe(true);
    expect(isProfileStorageKey("vf.profile.name")).toBe(true);
    // localStorage.clear() reports a null key.
    expect(isProfileStorageKey(null)).toBe(true);
    expect(isProfileStorageKey("vf.welcome.name")).toBe(false);
    expect(isProfileStorageKey("vf.lang")).toBe(false);
  });

  it("moves this tab to the profile the other tab chose", () => {
    expect(profileAfterStorageChange(ana, { slug: "bob", name: "Bob" })).toEqual(bob);
    expect(profileAfterStorageChange(DEFAULT_PROFILE_INFO, { slug: "ana", name: "Ana" })).toEqual(
      ana,
    );
  });

  it("goes back to the default when the other tab chose it or cleared the choice", () => {
    expect(profileAfterStorageChange(ana, { slug: "default", name: "default" })).toEqual(
      DEFAULT_PROFILE_INFO,
    );
    expect(profileAfterStorageChange(ana, null)).toEqual(DEFAULT_PROFILE_INFO);
    // A value the client would not send is the default as far as the server goes.
    expect(profileAfterStorageChange(ana, { slug: "../x", name: "x" })).toEqual(
      DEFAULT_PROFILE_INFO,
    );
  });

  it("does nothing when the tab already shows that profile", () => {
    expect(profileAfterStorageChange(ana, { slug: "ana", name: "Ana" })).toBeNull();
    expect(profileAfterStorageChange(DEFAULT_PROFILE_INFO, null)).toBeNull();
    expect(
      profileAfterStorageChange(DEFAULT_PROFILE_INFO, { slug: "default", name: "default" }),
    ).toBeNull();
  });

  it("picks up a new display name for the same profile", () => {
    expect(profileAfterStorageChange(ana, { slug: "ana", name: "Ana Souza" })).toEqual({
      slug: "ana",
      name: "Ana Souza",
      is_default: false,
    });
  });

  it("waits while this tab is still choosing", () => {
    // No profile is settled yet: the welcome flow reads the storage itself.
    expect(profileAfterStorageChange(null, { slug: "ana", name: "Ana" })).toBeNull();
    expect(profileAfterStorageChange(null, null)).toBeNull();
  });
});

describe("the queue", () => {
  it("names whose job it is only once someone besides the default is in it", () => {
    expect(queueShowsProfiles([])).toBe(false);
    expect(queueShowsProfiles([{ profile: "default" }, {}])).toBe(false);
    expect(queueShowsProfiles([{ profile: "default" }, { profile: "ana" }])).toBe(true);
  });

  it("labels a job with the display name, and the default with the interface's word", () => {
    expect(jobProfileLabel({ profile: "ana", profile_name: "Ana Souza" }, "Padrão")).toBe(
      "Ana Souza",
    );
    expect(jobProfileLabel({ profile: "default", profile_name: "default" }, "Padrão")).toBe(
      "Padrão",
    );
    expect(jobProfileLabel({}, "Padrão")).toBe("Padrão");
    expect(jobProfileLabel({ profile: "ana" }, "Padrão")).toBe("ana");
  });
});
