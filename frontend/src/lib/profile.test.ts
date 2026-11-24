import { afterEach, describe, expect, it, vi } from "vitest";

import {
  DEFAULT_PROFILE_INFO,
  accountProfile,
  activeProfileSlug,
  canCreateProfile,
  clearProfile,
  defaultHasName,
  entryStep,
  isProfileStorageKey,
  isValidProfileSlug,
  legacyDefaultName,
  namedAfterAccount,
  offersProfileChoice,
  profileHasName,
  profileLabel,
  jobProfileLabel,
  normalizeProfileName,
  pickStoredProfile,
  profileAfterStorageChange,
  profileHeaders,
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

  it("never offers a choice with a single profile, and always with two or more", () => {
    const marcus = { ...DEFAULT_PROFILE_INFO, name: "Marcus" };
    expect(offersProfileChoice([marcus])).toBe(false);
    expect(offersProfileChoice([])).toBe(false);
    expect(offersProfileChoice([marcus, ana])).toBe(true);
    // An unnamed default beside a named profile (old data) still makes two.
    expect(offersProfileChoice([DEFAULT_PROFILE_INFO, ana])).toBe(true);
  });
});

describe("the entry", () => {
  const marcus = { ...DEFAULT_PROFILE_INFO, name: "Marcus" };

  it("asks only the name on the first run, when no profile has one yet", () => {
    expect(entryStep([DEFAULT_PROFILE_INFO], "")).toBe("name");
    // A list that could not be read and nothing saved here: the same first run.
    expect(entryStep([], "")).toBe("name");
  });

  it("goes straight in with exactly one profile, on every load", () => {
    expect(entryStep([marcus], "")).toBe("enter");
    expect(entryStep([marcus], "Marcus")).toBe("enter");
    // The name this browser kept is handed to the unnamed default (legacyDefaultName).
    expect(entryStep([DEFAULT_PROFILE_INFO], "Marcus")).toBe("enter");
  });

  it("always shows the choice with more than one profile, also on a reload", () => {
    expect(entryStep([marcus, ana], "Marcus")).toBe("choose");
    expect(entryStep([marcus, ana, bob], "")).toBe("choose");
    expect(entryStep([DEFAULT_PROFILE_INFO, ana], "")).toBe("choose");
  });

  it("tells a named profile from the default that still has none", () => {
    expect(profileHasName(ana)).toBe(true);
    expect(profileHasName(marcus)).toBe(true);
    expect(profileHasName(DEFAULT_PROFILE_INFO)).toBe(false);
    expect(profileHasName({ ...DEFAULT_PROFILE_INFO, name: "" })).toBe(false);
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

  it("labels a job with the display name, the default too, and an unnamed one as such", () => {
    expect(jobProfileLabel({ profile: "ana", profile_name: "Ana Souza" }, "Sem nome")).toBe(
      "Ana Souza",
    );
    expect(jobProfileLabel({ profile: "default", profile_name: "Marcus" }, "Sem nome")).toBe(
      "Marcus",
    );
    expect(jobProfileLabel({ profile: "default", profile_name: "default" }, "Sem nome")).toBe(
      "Sem nome",
    );
    expect(jobProfileLabel({}, "Sem nome")).toBe("Sem nome");
    expect(jobProfileLabel({ profile: "ana" }, "Sem nome")).toBe("ana");
  });
});

describe("the profile of an account on a shared server (ADR-223)", () => {
  const ana = { slug: "ana-lab-org-0123456789", name: "ana@lab.org", is_default: false };

  it("is the one profile the server lists, and only on a server", () => {
    expect(accountProfile(true, [ana])).toBe(ana);
    expect(accountProfile(false, [ana])).toBeNull();
    // A list that could not be read falls back to the local default: not an account.
    expect(accountProfile(true, [DEFAULT_PROFILE_INFO])).toBeNull();
    expect(accountProfile(true, [])).toBeNull();
    expect(accountProfile(true, [ana, { ...ana, slug: "bia" }])).toBeNull();
  });

  it("still carries the login until the person says their name", () => {
    expect(namedAfterAccount(ana, "ana@lab.org")).toBe(true);
    expect(namedAfterAccount(ana, " Ana@Lab.org ")).toBe(true);
    expect(namedAfterAccount({ ...ana, name: "Ana Lima" }, "ana@lab.org")).toBe(false);
    expect(namedAfterAccount(ana, null)).toBe(false);
  });
});

describe("the default profile's name (ADR-126)", () => {
  const marcus = { ...DEFAULT_PROFILE_INFO, name: "Marcus" };

  it("labels the default by the welcome's name once it has one", () => {
    expect(profileLabel(marcus, "Sem nome")).toBe("Marcus");
    expect(profileLabel(DEFAULT_PROFILE_INFO, "Sem nome")).toBe("Sem nome");
    expect(profileLabel(ana, "Sem nome")).toBe("Ana");
    expect(defaultHasName(marcus)).toBe(true);
    expect(defaultHasName(DEFAULT_PROFILE_INFO)).toBe(false);
    expect(defaultHasName(ana)).toBe(false);
  });

  it("hands a name kept only in the browser to an unnamed default, once", () => {
    // The browser greets "Marcus" and is on the default: the server learns it.
    expect(legacyDefaultName([DEFAULT_PROFILE_INFO, ana], null, "Marcus")).toBe("Marcus");
    expect(
      legacyDefaultName([DEFAULT_PROFILE_INFO], { slug: "default", name: "" }, "Marcus"),
    ).toBe("Marcus");
    // Already named, no name to give, or this browser is on another profile.
    expect(legacyDefaultName([marcus], null, "Marcus")).toBeNull();
    expect(legacyDefaultName([DEFAULT_PROFILE_INFO], null, "")).toBeNull();
    expect(legacyDefaultName([DEFAULT_PROFILE_INFO, ana], { slug: "ana", name: "Ana" }, "Ana")).toBeNull();
    // A list that could not be read says nothing about the default.
    expect(legacyDefaultName([], null, "Marcus")).toBeNull();
  });
});
