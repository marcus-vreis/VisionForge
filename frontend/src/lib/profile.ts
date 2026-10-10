/** Perfis de um servidor compartilhado (ADR-114).
 *
 * Um perfil é uma pasta de saída no servidor, não uma conta: ele muda só onde os
 * treinos gravam e quais treinos o Histórico lista. O navegador guarda a escolha
 * no localStorage e a manda em todas as chamadas como `X-VF-Profile`.
 *
 * Este módulo é puro, fora a leitura e a escrita do localStorage, que toleram um
 * storage ausente ou bloqueado (a escolha vale só para a sessão).
 */

import { normalizeUserName } from "./user-name";

export const PROFILE_HEADER = "X-VF-Profile";
export const DEFAULT_PROFILE = "default";

const SLUG_KEY = "vf.profile";
const NAME_KEY = "vf.profile.name";

/** Mesmos limites do servidor (gui/api/profiles.py). */
export const MAX_SLUG_LENGTH = 40;
export const MAX_PROFILE_NAME = 60;

/** Nomes de dispositivo do Windows, que não podem ser nome de pasta lá. */
const WINDOWS_DEVICE = /^(?:con|prn|aux|nul|com[0-9]|lpt[0-9])$/;

const SLUG_PATTERN = new RegExp(`^[a-z0-9][a-z0-9_-]{0,${MAX_SLUG_LENGTH - 1}}$`);

/** Um perfil como `GET /api/profiles` o devolve. */
export interface ProfileInfo {
  slug: string;
  name: string;
  is_default: boolean;
}

/** O que o navegador lembra do perfil escolhido. */
export interface StoredProfile {
  slug: string;
  name: string;
}

/** O perfil padrão tal como a interface o trata quando o servidor não responde. */
export const DEFAULT_PROFILE_INFO: ProfileInfo = {
  slug: DEFAULT_PROFILE,
  name: DEFAULT_PROFILE,
  is_default: true,
};

/** Espaços colapsados e o teto do servidor, para o que foi digitado. */
export function normalizeProfileName(raw: string): string {
  return raw.trim().replace(/\s+/g, " ").slice(0, MAX_PROFILE_NAME);
}

/** O nome da pasta que o servidor derivaria de um nome; "" se nada sobra.
 *
 * Espelha `slugify` do servidor para mostrar a pasta antes de criar: acentos
 * viram a letra base, o resto fora de ASCII some, e o que não é `[a-z0-9_-]`
 * vira um `-`. O servidor é quem decide; isto só antecipa.
 */
export function slugifyProfile(name: string): string {
  const folded = name
    .normalize("NFKD")
    // eslint-disable-next-line no-control-regex
    .replace(/[^\x00-\x7f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^[-_]+|[-_]+$/g, "");
  return folded.slice(0, MAX_SLUG_LENGTH).replace(/[-_]+$/, "");
}

/** Se um slug é aceito pelo servidor como nome de perfil. */
export function isValidProfileSlug(slug: string): boolean {
  // "default" é o layout de sempre e os dispositivos do Windows não são pastas:
  // são os nomes que o servidor reserva.
  return SLUG_PATTERN.test(slug) && slug !== DEFAULT_PROFILE && !WINDOWS_DEVICE.test(slug);
}

/** Se um nome digitado pode virar um perfil novo (tem pasta utilizável). */
export function canCreateProfile(name: string): boolean {
  return isValidProfileSlug(slugifyProfile(normalizeProfileName(name)));
}

/** O cabeçalho que identifica o perfil numa chamada.
 *
 * Vazio para o perfil padrão e para qualquer valor que o servidor recusaria:
 * ausência do cabeçalho já significa "padrão", e um slug estragado no storage
 * não deve derrubar todas as chamadas com 400.
 */
export function profileHeaders(
  slug: string | null | undefined,
): Record<string, string> {
  if (!slug || slug === DEFAULT_PROFILE || !isValidProfileSlug(slug)) return {};
  return { [PROFILE_HEADER]: slug };
}

/** O perfil escolhido neste navegador, ou null se nunca houve escolha. */
export function readProfile(): StoredProfile | null {
  try {
    const slug = (localStorage.getItem(SLUG_KEY) ?? "").trim();
    if (!slug) return null;
    const name = (localStorage.getItem(NAME_KEY) ?? "").trim();
    return { slug, name: name || slug };
  } catch {
    return null;
  }
}

/** Se uma chave do `storage` event é a do perfil escolhido.
 *
 * `key` é null quando o storage inteiro foi apagado (`localStorage.clear()`),
 * o que também leva a escolha embora.
 */
export function isProfileStorageKey(key: string | null): boolean {
  return key === null || key === SLUG_KEY || key === NAME_KEY;
}

/** O perfil para onde esta aba deve ir depois que outra mexeu no storage.
 *
 * O localStorage é de todas as abas e o cliente da API o lê a cada chamada: o
 * que uma aba escolhe já vale para as próximas chamadas das outras. Isto diz
 * quando a tela delas precisa acompanhar. `stored` é `readProfile()` agora;
 * sem escolha guardada o cliente manda o padrão, e a tela deve dizer o mesmo.
 *
 * Devolve null quando não há o que mudar: a aba já mostra esse perfil (com esse
 * nome), ou ainda não decidiu o dela (a introdução lê o storage por conta
 * própria ao terminar).
 */
export function profileAfterStorageChange(
  current: ProfileInfo | null,
  stored: StoredProfile | null,
): ProfileInfo | null {
  if (current === null) return null;
  // A slug the client would not even send (profileHeaders) is the default too.
  const next: ProfileInfo =
    stored && isValidProfileSlug(stored.slug)
      ? { slug: stored.slug, name: stored.name, is_default: false }
      : DEFAULT_PROFILE_INFO;
  return next.slug === current.slug && next.name === current.name ? null : next;
}

/** O slug a mandar nas chamadas: o escolhido, ou o padrão. */
export function activeProfileSlug(): string {
  return readProfile()?.slug ?? DEFAULT_PROFILE;
}

export function saveProfile(profile: StoredProfile): void {
  try {
    localStorage.setItem(SLUG_KEY, profile.slug);
    localStorage.setItem(NAME_KEY, profile.name);
  } catch {
    /* storage indisponível — a escolha vale só para esta sessão */
  }
}

export function clearProfile(): void {
  try {
    localStorage.removeItem(SLUG_KEY);
    localStorage.removeItem(NAME_KEY);
  } catch {
    /* idem */
  }
}

/** Se o servidor tem algum perfil além do padrão. */
export function hasNamedProfiles(profiles: readonly ProfileInfo[]): boolean {
  return profiles.some((p) => !p.is_default);
}

/** O perfil guardado, se ainda existe no servidor (com o nome atual dele).
 *
 * Uma pasta apagada à mão deixa um slug órfão no navegador; ele não conta como
 * escolha, e quem tem perfis a escolher volta a ser perguntado.
 */
export function pickStoredProfile(
  stored: StoredProfile | null,
  profiles: readonly ProfileInfo[],
): ProfileInfo | null {
  if (!stored) return null;
  return profiles.find((p) => p.slug === stored.slug) ?? null;
}

/** Se a introdução precisa perguntar "quem é você?".
 *
 * Só num servidor com perfis além do padrão e só quando este navegador ainda não
 * escolheu um que exista; uma instalação de uma pessoa nunca vê a pergunta.
 */
export function needsProfileChoice(
  profiles: readonly ProfileInfo[],
  chosen: ProfileInfo | null,
): boolean {
  return hasNamedProfiles(profiles) && chosen === null;
}

/** Se o chip do perfil já diz quem está na tela, dispensando o do nome.
 *
 * Escolher um perfil nomeado grava o nome dele como o nome de quem usa (uma
 * pergunta só); dois chips com o mesmo nome lado a lado são só ruído.
 */
export function profileIsTheName(
  profile: ProfileInfo | null,
  userName: string,
): boolean {
  return (
    profile !== null &&
    !profile.is_default &&
    userName !== "" &&
    userName === normalizeUserName(profile.name)
  );
}

/** De quem é uma execução da fila, no texto da etiqueta.
 *
 * O padrão se chama como a interface o chama; um perfil, pelo nome de exibição
 * (e pelo slug se um servidor não o mandou).
 */
export function jobProfileLabel(
  job: { profile?: string; profile_name?: string },
  defaultLabel: string,
): string {
  const slug = job.profile ?? DEFAULT_PROFILE;
  if (slug === DEFAULT_PROFILE) return defaultLabel;
  return job.profile_name || slug;
}

/** Se a fila deve dizer de quem é cada execução.
 *
 * Numa instalação sem perfis toda execução é do padrão e a etiqueta seria
 * ruído; basta uma de outro perfil para ela passar a informar.
 */
export function queueShowsProfiles(jobs: readonly { profile?: string }[]): boolean {
  return jobs.some((j) => (j.profile ?? DEFAULT_PROFILE) !== DEFAULT_PROFILE);
}
