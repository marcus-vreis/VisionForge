import type { Dict } from "../i18n/pt";
import type { Pronoun } from "./profile";

/** "Bem-vindo", "Bem-vinda" or "Boas-vindas", with the name when there is one (ADR-126).
 *
 * No pronoun yet (a profile from before ADR-126, or a server that cannot be
 * reached) greets in the neutral form rather than guessing.
 */
export function greeting(t: Dict, pronoun: Pronoun | null | undefined, name?: string): string {
  const word = t.welcome.greet[pronoun ?? "neutro"];
  return name ? `${word}, ${name}` : word;
}
