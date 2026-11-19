/** Perfis de um servidor compartilhado (ADR-114).
 *
 * Um perfil é uma pasta de saída no servidor, não uma conta: ele muda só onde os
 * treinos gravam e quais treinos o Histórico lista. O navegador guarda a escolha
 * no localStorage e a manda em todas as chamadas como `X-VF-Profile`.
 *
 * Este módulo é puro, fora a leitura e a escrita do localStorage, que toleram um
 * storage ausente ou bloqueado (a escolha vale só para a sessão).
 */

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

/** Como a saudação trata a pessoa: "Bem-vindo", "Bem-vinda", "Boas-vindas" (ADR-126). */
export type Pronoun = "ele" | "ela" | "neutro";
export const PRONOUNS: readonly Pronoun[] = ["ele", "ela", "neutro"];

/** Um perfil como `GET /api/profiles` o devolve. */
export interface ProfileInfo {
  slug: string;
  /** No padrão, o nome dado na introdução, ou "default" enquanto não houver. */
  name: string;
  is_default: boolean;
  pronoun?: Pronoun | null;
  settings?: {
    mlflow: { enabled: boolean };
    wandb?: { enabled: boolean; project: string; entity: string };
    comet?: { enabled: boolean; project: string; workspace: string };
  };
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

/** O perfil de quem entrou num servidor compartilhado, ou null fora dele.
 *
 * Num servidor o perfil é a conta que o proxy de login nomeou, e a lista traz
 * só ele (ADR-223): não há escolha guardada a respeitar.
 */
export function accountProfile(
  serverMode: boolean,
  profiles: readonly ProfileInfo[],
): ProfileInfo | null {
  const only = profiles.length === 1 ? profiles[0] : undefined;
  return serverMode && only && !only.is_default ? only : null;
}

/** Se o perfil de uma conta ainda leva o nome da conta.
 *
 * O servidor cria o perfil com o login (um e-mail, quase sempre): enquanto for
 * esse o nome, ninguém disse como quer ser chamado, e a entrada pergunta.
 */
export function namedAfterAccount(profile: ProfileInfo, user: string | null): boolean {
  if (!user) return false;
  return profile.name.trim().toLowerCase() === user.trim().toLowerCase();
}

/** Se a entrada mostra a lista de perfis: só quando há mais de um.
 *
 * Com um perfil só não há o que escolher; com dois ou mais a lista aparece a
 * cada carga da página, com o último usado já marcado.
 */
export function offersProfileChoice(profiles: readonly ProfileInfo[]): boolean {
  return profiles.length > 1;
}

/** O passo da entrada depois da saudação; ver `entryStep`. */
export type EntryStep = "name" | "enter" | "choose";

/** O que a entrada faz depois da saudação.
 *
 * - `"name"`: primeira vez, nenhum perfil tem nome ainda: pede nome e pronome.
 * - `"enter"`: um perfil só, com nome: entra direto, a cada carga.
 * - `"choose"`: mais de um perfil: mostra a lista, a cada carga.
 *
 * `savedName` é o nome que este navegador guardou: um padrão ainda sem nome no
 * servidor o recebe (`legacyDefaultName`), então quem o tem já não é primeira
 * vez. Uma lista vazia (falha de leitura) decide só por ele.
 */
export function entryStep(profiles: readonly ProfileInfo[], savedName: string): EntryStep {
  if (offersProfileChoice(profiles)) return "choose";
  const only = profiles[0];
  if (only && profileHasName(only)) return "enter";
  return savedName ? "enter" : "name";
}

/** Se o perfil padrão já recebeu um nome na introdução (ADR-126). */
export function defaultHasName(profile: ProfileInfo): boolean {
  return profile.is_default && profile.name !== "" && profile.name !== DEFAULT_PROFILE;
}

/** Se um perfil tem nome: todo perfil criado tem; o padrão, depois da introdução. */
export function profileHasName(profile: ProfileInfo): boolean {
  return !profile.is_default || defaultHasName(profile);
}

/** Como um perfil aparece: pelo nome, ou pelo rótulo de "sem nome" da interface.
 *
 * O padrão é um detalhe de armazenamento (ADR-114): na tela ele é um perfil
 * como os outros, chamado pelo nome que recebeu.
 */
export function profileLabel(profile: ProfileInfo, unnamedLabel: string): string {
  return profileHasName(profile) ? profile.name : unnamedLabel;
}

/** O nome que este navegador guardou para um padrão que o servidor ainda não conhece.
 *
 * Antes do ADR-126 o nome da introdução ficava só no navegador, e o padrão
 * aparecia como "Padrão (sem perfil)" assim que surgia outro perfil. Quem está
 * no padrão (ou não escolheu nenhum) e tem um nome salvo o entrega ao servidor
 * uma vez. Uma lista vazia (falha de leitura) não diz nada sobre o padrão.
 */
export function legacyDefaultName(
  profiles: readonly ProfileInfo[],
  stored: StoredProfile | null,
  savedName: string,
): string | null {
  const fallback = profiles.find((p) => p.is_default);
  if (!fallback || defaultHasName(fallback) || !savedName) return null;
  if (stored && stored.slug !== DEFAULT_PROFILE) return null;
  return savedName;
}

/** De quem é uma execução da fila, no texto da etiqueta.
 *
 * Pelo nome de exibição, o do padrão inclusive; o padrão que ainda não tem nome
 * leva o rótulo de "sem nome", e um perfil cujo nome o servidor não mandou, o slug.
 */
export function jobProfileLabel(
  job: { profile?: string; profile_name?: string },
  unnamedLabel: string,
): string {
  const slug = job.profile ?? DEFAULT_PROFILE;
  if (slug !== DEFAULT_PROFILE) return job.profile_name || slug;
  const name = job.profile_name ?? "";
  return name && name !== DEFAULT_PROFILE ? name : unnamedLabel;
}

/** Se a fila deve dizer de quem é cada execução.
 *
 * Numa instalação sem perfis toda execução é do padrão e a etiqueta seria
 * ruído; basta uma de outro perfil para ela passar a informar.
 */
export function queueShowsProfiles(jobs: readonly { profile?: string }[]): boolean {
  return jobs.some((j) => (j.profile ?? DEFAULT_PROFILE) !== DEFAULT_PROFILE);
}
