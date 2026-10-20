import { PRONOUNS, type Pronoun } from "./profile";

const KEY = "vf.welcome.name";
// The pronoun the greeting used last, for the moment before the profile list loads.
const PRONOUN_KEY = "vf.welcome.pronoun";

/** Nome do pesquisador, guardado localmente.
 *
 * É o nome que o header cumprimenta neste navegador. O nome do perfil em uso
 * fica no servidor (o do padrão também, desde o ADR-126); esta cópia local é o
 * que a introdução mostra antes de a lista de perfis chegar, e o que sobra se o
 * servidor não responder. Leitura e escrita são tolerantes a falha (modo
 * privado / storage bloqueado): sem nome salvo, a introdução roda de novo.
 */
export function readUserName(): string {
  try {
    return (localStorage.getItem(KEY) ?? "").trim();
  } catch {
    return "";
  }
}

export function saveUserName(name: string): void {
  try {
    localStorage.setItem(KEY, name);
  } catch {
    /* storage indisponível — o nome vale só para esta sessão */
  }
}

export function clearUserName(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* idem */
  }
}

/** Normaliza o que foi digitado: espaços colapsados e um teto de 32 chars,
 *  que é o que cabe no chip do header sem quebrar a linha. */
export function normalizeUserName(raw: string): string {
  return raw.trim().replace(/\s+/g, " ").slice(0, 32);
}

/** O pronome guardado neste navegador, ou null (ADR-126). */
export function readUserPronoun(): Pronoun | null {
  try {
    const raw = localStorage.getItem(PRONOUN_KEY);
    return PRONOUNS.find((p) => p === raw) ?? null;
  } catch {
    return null;
  }
}

export function saveUserPronoun(pronoun: Pronoun | null | undefined): void {
  try {
    if (pronoun) localStorage.setItem(PRONOUN_KEY, pronoun);
    else localStorage.removeItem(PRONOUN_KEY);
  } catch {
    /* storage indisponível — vale só para esta sessão */
  }
}
