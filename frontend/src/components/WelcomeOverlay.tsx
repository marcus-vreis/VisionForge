import { useEffect, useRef, useState } from "react";

import { ApiError, createProfile, fetchProfiles } from "../api/client";
import { useT } from "../i18n/useT";
import {
  DEFAULT_PROFILE_INFO,
  canCreateProfile,
  clearProfile,
  hasNamedProfiles,
  needsProfileChoice,
  normalizeProfileName,
  pickStoredProfile,
  readProfile,
  saveProfile,
  slugifyProfile,
  type ProfileInfo,
} from "../lib/profile";
import {
  normalizeUserName,
  readUserName,
  saveUserName,
} from "../lib/user-name";

type Phase = "boot" | "hello" | "ask" | "who" | "create" | "exit" | "done";

interface WelcomeOverlayProps {
  /** Chamado quando o nome está definido (primeira visita ou visita seguinte). */
  onName: (name: string) => void;
  /** Chamado junto com `onName`, com o perfil em uso e se o servidor tem perfis
   *  além do padrão (é isso que faz o chip de perfil aparecer no header). */
  onProfile?: (profile: ProfileInfo, shared: boolean) => void;
  /** Força a introdução completa mesmo com nome salvo (usado pelo "trocar nome"). */
  forceAsk?: boolean;
  /** Vai direto a "Quem é você?" (usado pelo chip de perfil do header). */
  forcePick?: boolean;
}

/** Introdução de primeira execução (ADR-090), com a escolha de perfil (ADR-114).
 *
 * Primeira visita: a tela escurece, "Bem-vindo" entra, sai, e "Qual é o seu
 * nome?" abre uma linha no meio para digitar. O nome é salvo e passa a
 * aparecer no header.
 *
 * Visitas seguintes: só o cumprimento "Bem-vindo, {Nome}" por ~2s e entra —
 * nunca pergunta de novo. Quem quiser trocar clica no chip do header.
 *
 * Num servidor com perfis além do padrão, "Quem é você?" lista os perfis e
 * ocupa o lugar do nome: escolher um perfil também diz quem você é. Uma
 * instalação sem perfis segue o fluxo de sempre, com um "criar perfil" opcional
 * sob o campo do nome.
 */
export function WelcomeOverlay({
  onName,
  onProfile,
  forceAsk = false,
  forcePick = false,
}: WelcomeOverlayProps) {
  const t = useT();
  const saved = forceAsk || forcePick ? "" : readUserName();
  const [phase, setPhase] = useState<Phase>("boot");
  const [name, setName] = useState(saved);
  const [draft, setDraft] = useState("");
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
  // De onde veio "criar perfil", para o "voltar" voltar para lá.
  const createFrom = useRef<"ask" | "who">("who");

  // Termina a introdução: some a tela e entrega nome e perfil ao App.
  const finish = (finalName: string) => {
    setName(finalName);
    setPhase("exit");
    window.setTimeout(() => {
      if (!aliveRef.current) return;
      setPhase("done");
      const chosen = chosenRef.current;
      handlers.current.onProfile?.(
        chosen,
        hasNamedProfiles(listRef.current) || !chosen.is_default,
      );
      handlers.current.onName(finalName);
    }, 850);
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
      const chosen = pickStoredProfile(stored, list);
      if (chosen) {
        chosenRef.current = chosen;
        // O nome pode ter mudado no servidor desde que foi guardado.
        if (chosen.name !== stored?.name) saveProfile(chosen);
      } else {
        chosenRef.current = DEFAULT_PROFILE_INFO;
        if (stored) clearProfile();
      }
      return chosen;
    };

    if (forcePick) {
      at(
        120,
        () =>
          void load.then(() => {
            if (aliveRef.current) setPhase("who");
          }),
      );
    } else if (returning) {
      at(60, () => setPhase("hello"));
      at(
        2100,
        () =>
          void load.then((list) => {
            if (!aliveRef.current) return;
            if (needsProfileChoice(list, settle(list))) {
              setPhase("who");
              return;
            }
            finish(saved);
          }),
      );
    } else {
      at(160, () => setPhase("hello"));
      at(
        2300,
        () =>
          void load.then((list) => {
            if (!aliveRef.current) return;
            setPhase(needsProfileChoice(list, settle(list)) ? "who" : "ask");
          }),
      );
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

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const value = normalizeUserName(draft);
    if (!value) {
      inputRef.current?.focus();
      return;
    }
    saveUserName(value);
    finish(value);
  };

  const choose = (profile: ProfileInfo) => {
    chosenRef.current = profile;
    saveProfile({ slug: profile.slug, name: profile.name });
    if (!profile.is_default) {
      // Escolher um perfil já diz quem você é: uma pergunta só, não duas.
      const own = normalizeUserName(profile.name);
      saveUserName(own);
      finish(own);
      return;
    }
    // "Sem perfil" não diz nome nenhum; quem já tem um salvo o mantém.
    const kept = readUserName();
    if (kept) finish(kept);
    else setPhase("ask");
  };

  const openCreate = (from: "ask" | "who") => {
    createFrom.current = from;
    setCreateDraft(from === "ask" ? draft : "");
    setCreateError(null);
    setPhase("create");
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
      const created = await createProfile(display);
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
    fontSize: 11,
    letterSpacing: "0.14em",
    textTransform: "uppercase",
    color: "var(--vf-text-muted)",
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
        background: `rgba(4,5,7,${phase === "boot" ? 0 : 0.78})`,
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
            fontSize: 11,
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
          {returning ? t.welcome.helloName(name) : t.welcome.hello}
        </div>
      </div>

      <div style={layer(askOn)} inert={!askOn}>
        <div style={titleStyle}>{t.welcome.askName}</div>

        <form
          onSubmit={submit}
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

          <button
            type="submit"
            style={{
              padding: "12px 30px",
              background: canEnter ? "var(--accent-soft)" : "transparent",
              border: `1px solid ${
                canEnter ? "var(--accent-vf)" : "rgba(255,255,255,0.10)"
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

        {profilesOk && (
          <button
            type="button"
            onClick={() => openCreate("ask")}
            style={{ ...linkStyle, marginTop: 22 }}
          >
            {t.profile.createLink}
          </button>
        )}
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
          {profiles.map((profile, index) => (
            <button
              key={profile.slug}
              ref={index === 0 ? firstOptionRef : undefined}
              type="button"
              className="vf-profile-option"
              onClick={() => choose(profile)}
              style={{
                padding: "13px 18px",
                background: "rgba(255,255,255,0.025)",
                border: "1px solid var(--vf-panel-stroke)",
                borderRadius: 12,
                color: "var(--vf-text)",
                fontFamily: "var(--font-display)",
                fontSize: 19,
                letterSpacing: "-0.01em",
                textAlign: "left",
                cursor: "pointer",
              }}
            >
              {profile.is_default ? t.profile.defaultName : profile.name}
            </button>
          ))}
        </div>

        <button
          type="button"
          onClick={() => openCreate("who")}
          style={{ ...linkStyle, marginTop: 22 }}
        >
          {t.profile.create}
        </button>

        <div
          style={{
            marginTop: 26,
            maxWidth: 380,
            textAlign: "center",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
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
                fontSize: 11,
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
                fontSize: 11,
                color: "oklch(0.82 0.14 22)",
              }}
            >
              {createError ?? ""}
            </div>
          </div>

          <button
            type="submit"
            disabled={!canCreate}
            style={{
              padding: "12px 30px",
              background: canCreate ? "var(--accent-soft)" : "transparent",
              border: `1px solid ${
                canCreate ? "var(--accent-vf)" : "rgba(255,255,255,0.10)"
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
          onClick={() => setPhase(createFrom.current)}
          style={{ ...linkStyle, marginTop: 18 }}
        >
          {t.profile.back}
        </button>
      </div>
    </div>
  );
}
