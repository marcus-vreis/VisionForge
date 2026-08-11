const KEY = "vf.welcome.name";

/** Nome do pesquisador, guardado localmente.
 *
 * Fica no localStorage e não no backend de propósito: é preferência de
 * interface de quem está na frente da máquina, não estado do servidor — uma
 * instalação local compartilhada não deve reescrever o nome de outro usuário
 * a cada acesso. Leitura e escrita são tolerantes a falha (modo privado /
 * storage bloqueado): sem nome salvo, a introdução simplesmente roda de novo.
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
