import { useEffect, useRef, useState } from "react";

import { ApiError, createProfile, fetchProfiles, updateProfile } from "../api/client";
import { useT } from "../i18n/useT";
import { greeting } from "../lib/greeting";
import {
  DEFAULT_PROFILE_INFO,
  PRONOUNS,
  canCreateProfile,
  clearProfile,
  entryStep,
  legacyDefaultName,
  normalizeProfileName,
  offersProfileChoice,
  pickStoredProfile,
  profileHasName,
  profileLabel,
  readProfile,
  saveProfile,
  slugifyProfile,
  type ProfileInfo,
  type Pronoun,
} from "../lib/profile";
import {
  normalizeUserName,
  readUserName,
  readUserPronoun,
  saveUserName,
  saveUserPronoun,
} from "../lib/user-name";

type Phase = "boot" | "hello" | "ask" | "who" | "create" | "exit" | "done";

interface WelcomeOverlayProps {
  /** Chamado quando o nome está definido (primeira visita ou visita seguinte). */
  onName: (name: string) => void;
  /** Chamado junto com `onName`, com o perfil em uso e se há mais de um perfil
   *  (o menu do perfil no header oferece "trocar" em vez de "novo"). */
  onProfile?: (profile: ProfileInfo, shared: boolean) => void;
  /** Força a introdução completa mesmo com nome salvo (usado pelo "trocar nome"). */
  forceAsk?: boolean;
  /** Vai direto a "Quem é você?" (usado pelo "trocar de perfil" do header). */
  forcePick?: boolean;
  /** Vai direto a "Novo perfil" (usado pelo "novo perfil" do header). */
  forceCreate?: boolean;
}

/** Introdução de primeira execução (ADR-090), com a escolha de perfil (ADR-114).
 *
 * A tela escurece e a saudação entra; depois dela, `entryStep` decide:
 * - primeira vez (nenhum perfil tem nome): "Qual é o seu nome?" com o pronome,
 *   e nada mais; o nome vai para o perfil que o servidor já tem (ADR-126);
 * - um perfil só: entra direto, a cada carga;
 * - mais de um: "Quem é você?" lista os perfis a cada carga da página, com o
 *   último usado marcado e focado, para Enter continuar. Escolher um perfil
 *   também diz quem você é.
 *
 * As integrações (MLflow, W&B, Comet) ficam no menu do perfil no header, não
 * aqui. A saudação segue o pronome.
 */
export function WelcomeOverlay({
  onName,
  onProfile,
  forceAsk = false,
  forcePick = false,
  forceCreate = false,
}: WelcomeOverlayProps) {
  const t = useT();
  const saved = forceAsk || forcePick || forceCreate ? "" : readUserName();
  const [phase, setPhase] = useState<Phase>("boot");
  const [name, setName] = useState(saved);
  // "Trocar nome" abre o campo já com o nome atual, para corrigir em vez de redigitar.
  const [draft, setDraft] = useState(() => (forceAsk ? readUserName() : ""));
  // A saudação de entrada usa o pronome deste navegador: a lista ainda não chegou.
  const [helloPronoun] = useState<Pronoun | null>(() => readUserPronoun());
  const [pronoun, setPronoun] = useState<Pronoun>(() => readUserPronoun() ?? "neutro");
  // O perfil marcado em "Quem é você?": o último escolhido, para um clique continuar.
  const [lastSlug, setLastSlug] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const firstOptionRef = useRef<HTMLButtonElement>(null);
  const createRef = useRef<HTMLInputElement>(null);
  const returning = saved.length > 0;

  // O que o servidor respondeu sobre perfis. Fica em refs porque os temporizadores
  // da introdução, criados uma vez na montagem, precisam do valor de agora.
  const listRef = useRef<ProfileInfo[]>([DEFAULT_PROFILE_INFO]);
  const chosenRef = useRef<ProfileInfo>(DEFAULT_PROFILE_INFO);
  const aliveRef = useRef(true);
  // Quando a lista não pôde ser lida, o que o navegador guardou não é apagado:
  // uma falha de rede não prova que o perfil deixou de existir.
  const listFailedRef = useRef(false);
  // Os temporizadores nascem na montagem e chamariam os callbacks daquela
  // renderização; a ref entrega sempre os de agora.
  const handlers = useRef({ onName, onProfile });
  useEffect(() => {
    handlers.current = { onName, onProfile };
  });
  const [profiles, setProfiles] = useState<ProfileInfo[]>([DEFAULT_PROFILE_INFO]);
  // Falso quando o servidor não conhece /api/profiles (um servidor antigo):
  // então nada de perfil é oferecido e o fluxo é o de antes.
  const [profilesOk, setProfilesOk] = useState(false);

  const [createDraft, setCreateDraft] = useState("");
  const [createError, setCreateError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  // De onde veio "criar perfil", para o "voltar" voltar para lá; "none" é o
  // menu do header, e voltar fecha a tela no perfil em uso.
  const createFrom = useRef<"who" | "none">(forceCreate ? "none" : "who");

  // Termina a introdução: some a tela e entrega nome e perfil ao App.
  const finish = (finalName: string) => {
    setName(finalName);
    setPhase("exit");
    window.setTimeout(() => {
      if (!aliveRef.current) return;
      setPhase("done");
      const chosen = chosenRef.current;
      // Estar num perfil criado já prova que existe outro além do do servidor.
      handlers.current.onProfile?.(
        chosen,
        offersProfileChoice(listRef.current) || !chosen.is_default,
      );
      handlers.current.onName(finalName);
    }, 850);
  };

  /** O nome de quem está no perfil em uso: o do perfil, ou o deste navegador. */
  const currentName = () => {
    const chosen = chosenRef.current;
    return profileHasName(chosen) ? normalizeUserName(chosen.name) : readUserName();
  };

  /** Um perfil só: entra nele direto, com o nome dele. */
  const enterOnly = () => {
    const own = currentName();
    if (!own) {
      setPhase("ask");
      return;
    }
    saveUserName(own);
    const chosen = chosenRef.current;
    if (chosen.pronoun) saveUserPronoun(chosen.pronoun);
    finish(own);
  };

  useEffect(() => {
    aliveRef.current = true;
    const timers: number[] = [];
    const at = (ms: number, fn: () => void) =>
      timers.push(window.setTimeout(fn, ms));

    const load = fetchProfiles()
      .then((list) => {
        listRef.current = list;
        setProfiles(list);
        setProfilesOk(true);
        return list;
      })
      .catch(() => {
        listFailedRef.current = true;
        listRef.current = [DEFAULT_PROFILE_INFO];
        return listRef.current;
      });

    /** O perfil guardado se ainda existe; senão o padrão (e esquece o órfão). */
    const settle = (list: ProfileInfo[]) => {
      const stored = readProfile();
      if (listFailedRef.current) {
        chosenRef.current = stored
          ? {
              slug: stored.slug,
              name: stored.name,
              is_default: stored.slug === DEFAULT_PROFILE_INFO.slug,
            }
          : DEFAULT_PROFILE_INFO;
        return chosenRef.current;
      }
      const legacy = legacyDefaultName(list, stored, readUserName());
      if (legacy) void nameTheDefault(legacy);
      const chosen = pickStoredProfile(stored, list);
      if (chosen) {
        chosenRef.current = chosen;
        // O nome pode ter mudado no servidor desde que foi guardado.
        if (chosen.name !== stored?.name) saveProfile(chosen);
        if (chosen.pronoun) setPronoun(chosen.pronoun);
      } else {
        chosenRef.current =
          listRef.current.find((p) => p.is_default) ?? DEFAULT_PROFILE_INFO;
        if (stored) clearProfile();
      }
      setLastSlug(chosen?.slug ?? null);
      return chosen;
    };

    /** O nome que só este navegador sabia vira o do padrão no servidor (ADR-126). */
    const nameTheDefault = async (legacy: string) => {
      try {
        const named = await updateProfile(DEFAULT_PROFILE_INFO.slug, { name: legacy });
        listRef.current = listRef.current.map((p) => (p.is_default ? named : p));
        if (aliveRef.current) setProfiles(listRef.current);
        if (chosenRef.current.is_default) chosenRef.current = named;
      } catch {
        /* sem servidor, o perfil segue "Sem nome" na lista até a próxima visita */
      }
    };

    /** Depois da saudação: o nome, a entrada direta ou a lista (`entryStep`). */
    const proceed = (list: ProfileInfo[]) => {
      if (!aliveRef.current) return;
      settle(list);
      // "Trocar nome" é sobre o perfil em uso: vai direto ao campo.
      if (forceAsk) {
        setPhase("ask");
        return;
      }
      const step = entryStep(list, readUserName());
      if (step === "choose") setPhase("who");
      else if (step === "name") setPhase("ask");
      else enterOnly();
    };

    if (forcePick || forceCreate) {
      at(
        120,
        () =>
          void load.then((list) => {
            if (!aliveRef.current) return;
            settle(list);
            setPhase(forceCreate ? "create" : "who");
          }),
      );
    } else {
      at(returning ? 60 : 160, () => setPhase("hello"));
      at(returning ? 2100 : 2300, () => void load.then(proceed));
    }
    return () => {
      aliveRef.current = false;
      timers.forEach(clearTimeout);
    };
    // Roda uma vez por montagem: o fluxo é uma sequência, não um efeito reativo.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // O foco vai para o que a tela nova pede, depois que ela termina de entrar.
  useEffect(() => {
    const target =
      phase === "ask"
        ? inputRef
        : phase === "who"
          ? firstOptionRef
          : phase === "create"
            ? createRef
            : null;
    if (!target) return;
    const id = window.setTimeout(() => target.current?.focus(), 450);
    return () => clearTimeout(id);
  }, [phase]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const value = normalizeUserName(draft);
    if (!value) {
      inputRef.current?.focus();
      return;
    }
    saveUserName(value);
    saveUserPronoun(pronoun);
    // O nome e o pronome são do perfil em uso, o padrão inclusive (ADR-126).
    if (profilesOk) {
      try {
        const saved = await updateProfile(chosenRef.current.slug, { name: value, pronoun });
        chosenRef.current = saved;
        listRef.current = listRef.current.map((p) => (p.slug === saved.slug ? saved : p));
      } catch {
        /* o nome fica neste navegador; o servidor aprende na próxima visita */
      }
    }
    finish(value);
  };

  const choose = (profile: ProfileInfo) => {
    chosenRef.current = profile;
    saveProfile({ slug: profile.slug, name: profile.name });
    if (profile.pronoun) {
      saveUserPronoun(profile.pronoun);
      setPronoun(profile.pronoun);
    }
    if (profileHasName(profile)) {
      // Escolher um perfil já diz quem você é: uma pergunta só, não duas.
      const own = normalizeUserName(profile.name);
      saveUserName(own);
      finish(own);
      return;
    }
    // Um perfil antigo sem nome: pergunta, com o nome deste navegador já no campo.
    setDraft(readUserName());
    setPhase("ask");
  };

  const openCreate = () => {
    createFrom.current = "who";
    setCreateDraft("");
    setCreateError(null);
    setPhase("create");
  };

  const leaveCreate = () => {
    if (createFrom.current === "who") {
      setPhase("who");
      return;
    }
    // Veio do menu do header: volta ao perfil em uso, sem mudar nada.
    finish(currentName());
  };

  const submitCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    const display = normalizeProfileName(createDraft);
    if (!canCreateProfile(display)) {
      setCreateError(t.profile.invalid);
      createRef.current?.focus();
      return;
    }
    setCreating(true);
    setCreateError(null);
    try {
      const created = await createProfile(display, pronoun);
      listRef.current = [...listRef.current, created];
      choose(created);
    } catch (err) {
      setCreateError(
        err instanceof ApiError && err.status === 409
          ? t.profile.exists
          : err instanceof ApiError && err.status === 422
            ? t.profile.invalid
            : t.profile.failed,
      );
      createRef.current?.focus();
    } finally {
      setCreating(false);
    }
  };

  // O overlay some do fluxo depois da saída para não capturar cliques.
  if (phase === "done") return null;

  const visible =
    phase === "hello" || phase === "ask" || phase === "who" || phase === "create";
  const helloOn = phase === "hello";
  const askOn = phase === "ask";
  const whoOn = phase === "who";
  const createOn = phase === "create";
  const canEnter = draft.trim().length > 0;
  const createSlug = slugifyProfile(normalizeProfileName(createDraft));
  const canCreate = canCreateProfile(createDraft) && !creating;

  const layer = (on: boolean): React.CSSProperties => ({
    position: "absolute",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    opacity: on ? 1 : 0,
    transform: `translateY(${
      on ? "0px" : phase === "exit" ? "-18px" : "26px"
    })`,
    filter: `blur(${on ? 0 : 5}px)`,
    pointerEvents: on ? "auto" : "none",
    transition:
      "opacity 850ms ease 120ms, transform 950ms cubic-bezier(.16,.84,.24,1) 120ms, filter 850ms ease",
  });

  const titleStyle: React.CSSProperties = {
    fontFamily: "var(--font-display)",
    fontSize: 38,
    fontWeight: 500,
    letterSpacing: "-0.02em",
    color: "var(--vf-text)",
    textAlign: "center",
  };

  const linkStyle: React.CSSProperties = {
    background: "none",
    border: "none",
    padding: "4px 8px",
    fontFamily: "var(--font-mono)",
    fontSize: 13,
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    color: "var(--vf-text-dim)",
    textDecoration: "underline",
    textUnderlineOffset: 4,
    cursor: "pointer",
  };

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 40,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: `rgb(var(--lt-surface, 4 5 7) / ${phase === "boot" ? 0 : 0.78})`,
        backdropFilter: `blur(${visible ? 10 : 0}px)`,
        WebkitBackdropFilter: `blur(${visible ? 10 : 0}px)`,
        opacity: phase === "exit" ? 0 : 1,
        transition:
          "background 700ms ease, backdrop-filter 700ms ease, opacity 650ms ease",
      }}
    >
      <div
        style={{
          position: "absolute",
          width: 520,
          height: 520,
          borderRadius: "50%",
          border: "1px solid var(--accent-soft)",
          pointerEvents: "none",
          opacity: visible ? 1 : 0,
          transition: "opacity 800ms ease",
          animation: "vfRing 4.6s ease-out infinite",
        }}
      />

      <div
        style={{
          position: "absolute",
          textAlign: "center",
          pointerEvents: "none",
          opacity: helloOn ? 1 : 0,
          transform: `translateY(${
            phase === "boot" ? "22px" : helloOn ? "0px" : "-30px"
          }) scale(${helloOn ? 1 : 0.97})`,
          filter: `blur(${helloOn ? 0 : 6}px)`,
          transition:
            "opacity 900ms ease, transform 1100ms cubic-bezier(.16,.84,.24,1), filter 900ms ease",
        }}
      >
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            letterSpacing: "0.3em",
            textTransform: "uppercase",
            color: "var(--vf-text-muted)",
            marginBottom: 18,
          }}
        >
          VisionForge
        </div>
        <div
          style={{
            fontFamily: "var(--font-display)",
            fontSize: 74,
            fontWeight: 600,
            letterSpacing: "-0.03em",
            lineHeight: 1,
            color: "var(--vf-text)",
          }}
        >
          {returning ? greeting(t, helloPronoun, name) : greeting(t, helloPronoun)}
        </div>
      </div>

      <div style={layer(askOn)} inert={!askOn}>
        <div style={titleStyle}>{t.welcome.askName}</div>

        <form
          onSubmit={(e) => void submit(e)}
          style={{
            marginTop: 38,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 26,
          }}
        >
          <div
            style={{
              position: "relative",
              width: "min(520px, 78vw)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <input
              ref={inputRef}
              type="text"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder={t.welcome.placeholder}
              autoComplete="off"
              spellCheck={false}
              maxLength={40}
              aria-label={t.welcome.nameLabel}
              style={{
                width: "100%",
                textAlign: "center",
                background: "transparent",
                border: "none",
                outline: "none",
                padding: "6px 4px 14px",
                fontFamily: "var(--font-display)",
                fontSize: 30,
                letterSpacing: "-0.01em",
                color: "var(--vf-text)",
                caretColor: "var(--accent-vf)",
              }}
            />
            <div
              style={{
                height: 1,
                width: askOn ? "100%" : "0%",
                background:
                  "linear-gradient(90deg, transparent, var(--accent-vf), transparent)",
                boxShadow: "0 0 14px var(--accent-glow)",
                transition: "width 900ms cubic-bezier(.16,.84,.24,1) 380ms",
              }}
            />
          </div>

          <PronounPicker value={pronoun} onChange={setPronoun} />

          <button
            type="submit"
            style={{
              padding: "12px 30px",
              background: canEnter ? "var(--accent-soft)" : "transparent",
              border: `1px solid ${
                canEnter ? "var(--accent-vf)" : "rgb(var(--lt-ink, 255 255 255) / 0.10)"
              }`,
              borderRadius: 12,
              color: canEnter ? "var(--vf-text)" : "var(--vf-text-muted)",
              fontFamily: "var(--font-mono)",
              fontSize: 12,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              cursor: "pointer",
              boxShadow: canEnter ? "inset 0 0 18px var(--accent-glow)" : "none",
              transition: "all 400ms ease",
            }}
          >
            {t.welcome.enter}
          </button>
        </form>
      </div>

      <div style={layer(whoOn)} inert={!whoOn}>
        <div style={titleStyle}>{t.profile.who}</div>

        <div
          role="group"
          aria-label={t.profile.listLabel}
          style={{
            marginTop: 34,
            display: "flex",
            flexDirection: "column",
            gap: 10,
            width: "min(380px, 78vw)",
            maxHeight: "38vh",
            overflowY: "auto",
          }}
        >
          {profiles.map((profile, index) => {
            const isLast = lastSlug === null ? index === 0 : profile.slug === lastSlug;
            return (
              <button
                key={profile.slug}
                ref={isLast ? firstOptionRef : undefined}
                type="button"
                className="vf-profile-option"
                aria-current={profile.slug === lastSlug ? "true" : undefined}
                onClick={() => choose(profile)}
                style={{
                  padding: "14px 18px",
                  background:
                    profile.slug === lastSlug
                      ? "var(--accent-soft)"
                      : "rgb(var(--lt-ink, 255 255 255) / 0.025)",
                  border: `1px solid ${
                    profile.slug === lastSlug ? "var(--accent-vf)" : "var(--vf-panel-stroke)"
                  }`,
                  borderRadius: 12,
                  color: "var(--vf-text)",
                  fontFamily: "var(--font-display)",
                  fontSize: 20,
                  letterSpacing: "-0.01em",
                  textAlign: "left",
                  cursor: "pointer",
                  // "Sem nome" descreve um estado, não é um nome: vai em itálico.
                  fontStyle: profileHasName(profile) ? "normal" : "italic",
                }}
              >
                {profileLabel(profile, t.profile.unnamed)}
              </button>
            );
          })}
        </div>

        <button
          type="button"
          onClick={openCreate}
          style={{ ...linkStyle, marginTop: 22 }}
        >
          {t.profile.create}
        </button>

        <div
          style={{
            marginTop: 22,
            maxWidth: "min(520px, 86vw)",
            textAlign: "center",
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            lineHeight: 1.6,
            color: "var(--vf-text-muted)",
          }}
        >
          {t.profile.notice}
        </div>
      </div>

      <div style={layer(createOn)} inert={!createOn}>
        <div style={titleStyle}>{t.profile.createTitle}</div>

        <form
          onSubmit={(e) => void submitCreate(e)}
          style={{
            marginTop: 38,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 20,
          }}
        >
          <div
            style={{
              width: "min(520px, 78vw)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <input
              ref={createRef}
              type="text"
              value={createDraft}
              onChange={(e) => {
                setCreateDraft(e.target.value);
                setCreateError(null);
              }}
              placeholder={t.profile.namePlaceholder}
              autoComplete="off"
              spellCheck={false}
              maxLength={60}
              aria-label={t.profile.nameLabel}
              style={{
                width: "100%",
                textAlign: "center",
                background: "transparent",
                border: "none",
                outline: "none",
                padding: "6px 4px 14px",
                fontFamily: "var(--font-display)",
                fontSize: 30,
                letterSpacing: "-0.01em",
                color: "var(--vf-text)",
                caretColor: "var(--accent-vf)",
              }}
            />
            <div
              style={{
                height: 1,
                width: "100%",
                background:
                  "linear-gradient(90deg, transparent, var(--accent-vf), transparent)",
                boxShadow: "0 0 14px var(--accent-glow)",
              }}
            />
            <div
              style={{
                marginTop: 12,
                minHeight: 18,
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                letterSpacing: "0.06em",
                color: "var(--vf-text-muted)",
              }}
            >
              {createDraft.trim()
                ? createSlug
                  ? t.profile.folder(createSlug)
                  : t.profile.folderNone
                : ""}
            </div>
            <div
              role="alert"
              style={{
                minHeight: 18,
                marginTop: 4,
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                color: "oklch(calc(0.82 - var(--lt-flip, 0) * 0.34) 0.14 22)",
              }}
            >
              {createError ?? ""}
            </div>
          </div>

          <PronounPicker value={pronoun} onChange={setPronoun} />

          <button
            type="submit"
            disabled={!canCreate}
            style={{
              padding: "12px 30px",
              background: canCreate ? "var(--accent-soft)" : "transparent",
              border: `1px solid ${
                canCreate ? "var(--accent-vf)" : "rgb(var(--lt-ink, 255 255 255) / 0.10)"
              }`,
              borderRadius: 12,
              color: canCreate ? "var(--vf-text)" : "var(--vf-text-muted)",
              fontFamily: "var(--font-mono)",
              fontSize: 12,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              cursor: canCreate ? "pointer" : "default",
              boxShadow: canCreate ? "inset 0 0 18px var(--accent-glow)" : "none",
              transition: "all 400ms ease",
            }}
          >
            {creating ? t.profile.creating : t.profile.submit}
          </button>
        </form>

        <button
          type="button"
          onClick={leaveCreate}
          style={{ ...linkStyle, marginTop: 18 }}
        >
          {t.profile.back}
        </button>
      </div>
    </div>
  );
}

/** "Bem-vindo", "Bem-vinda" ou "Boas-vindas": como a saudação trata a pessoa (ADR-126). */
function PronounPicker({
  value,
  onChange,
}: {
  value: Pronoun;
  onChange: (next: Pronoun) => void;
}) {
  const t = useT();
  return (
    <div
      role="radiogroup"
      aria-label={t.welcome.pronounLabel}
      style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10 }}
    >
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 12,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        {t.welcome.pronounLabel}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 8 }}>
        {PRONOUNS.map((p) => {
          const on = p === value;
          return (
            <button
              key={p}
              type="button"
              role="radio"
              aria-checked={on}
              onClick={() => onChange(p)}
              style={{
                padding: "8px 14px",
                background: on ? "var(--accent-soft)" : "transparent",
                border: `1px solid ${on ? "var(--accent-vf)" : "var(--vf-panel-stroke)"}`,
                borderRadius: 10,
                color: on ? "var(--vf-text)" : "var(--vf-text-dim)",
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                cursor: "pointer",
              }}
            >
              {t.welcome.pronounNames[p]}
            </button>
          );
        })}
      </div>
    </div>
  );
}
