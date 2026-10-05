/** O roteiro do guia de primeira execução (ADR-104).
 *
 * Cada passo aponta para um elemento marcado com `data-tour`. O alvo é
 * procurado no DOM na hora: se ele não estiver na tela — porque a tarefa ativa
 * não tem aquele campo, ou porque o painel ainda está carregando — o passo vira
 * um cartão centralizado em vez de sumir. Assim o roteiro é o mesmo para as
 * cinco tarefas sem precisar de uma versão por painel.
 */

import type { Dict } from "../i18n/pt";

export interface TourStep {
  /** Valor de `data-tour` do elemento destacado. Sem ele, o cartão centraliza. */
  anchor?: string;
  title: string;
  body: string;
}

/** Os sete passos, com os textos no idioma do dicionário recebido.
 *
 * Só as âncoras moram aqui; títulos e corpos estão em `tour` (src/i18n). */
export function tourSteps(t: Dict): TourStep[] {
  const s = t.tour;
  return [
    { anchor: "tabs", title: s.tabs.title, body: s.tabs.body },
    { anchor: "dataset", title: s.dataset.title, body: s.dataset.body },
    { title: s.parameters.title, body: s.parameters.body },
    { anchor: "device", title: s.device.title, body: s.device.body },
    { anchor: "train", title: s.train.title, body: s.train.body },
    { anchor: "history", title: s.history.title, body: s.history.body },
    { anchor: "datasets", title: s.datasets.title, body: s.datasets.body },
  ];
}

const KEY = "vf.tour.seen";

/** Se o guia já foi oferecido nesta máquina.
 *
 * Mesmo raciocínio do nome do pesquisador: é preferência de quem está na
 * frente da tela, não estado do servidor. Storage bloqueado (modo privado)
 * responde "não visto" — o guia é oferecido de novo, o que é melhor do que
 * quebrar a tela por causa de uma preferência.
 */
export function readTourSeen(): boolean {
  try {
    return localStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

export function markTourSeen(): void {
  try {
    localStorage.setItem(KEY, "1");
  } catch {
    /* storage indisponível — o guia volta a ser oferecido na próxima visita */
  }
}

export function clearTourSeen(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* idem */
  }
}

/** Largura fixa do cartão do guia; a geometria abaixo depende dela. */
export const CARD_WIDTH = 400;
/** Folga mínima entre o cartão e a borda da tela. */
const MARGIN = 20;
/** Espaço entre o cartão e o elemento destacado. */
const GAP = 16;

export interface Placement {
  left: number;
  top: number;
}

export interface Viewport {
  width: number;
  height: number;
}

/** Onde o cartão cabe: abaixo do alvo, acima dele, ou no centro da tela.
 *
 * Sem alvo (o convite e os passos que falam da interface inteira) ele
 * centraliza. Com alvo, a preferência é ficar logo abaixo — é onde o olho já
 * está depois de ler o destaque — e só sobe quando o rodapé não tem espaço.
 *
 * O resultado é sempre preso à tela: o alvo pode estar fora dela enquanto a
 * rolagem suave não terminou, e "acima do alvo" seria então uma posição que
 * ninguém vê.
 */
export function placeCard(
  rect: DOMRect | null,
  height: number,
  view: Viewport,
): Placement {
  if (!rect) {
    return {
      left: clamp((view.width - CARD_WIDTH) / 2, view.width),
      top: clamp((view.height - height) / 2, view.height),
    };
  }
  const below = rect.bottom + GAP;
  const above = rect.top - GAP - height;
  const preferred =
    below + height + MARGIN <= view.height
      ? below
      : above >= MARGIN
        ? above
        : (view.height - height) / 2;
  const centred = rect.left + rect.width / 2 - CARD_WIDTH / 2;
  return {
    left: clamp(centred, view.width - CARD_WIDTH - MARGIN),
    top: clamp(preferred, view.height - height - MARGIN),
  };
}

/** Entre a margem e o limite, sem inverter quando a tela é menor que a caixa. */
function clamp(value: number, limit: number): number {
  return Math.min(Math.max(MARGIN, value), Math.max(MARGIN, limit));
}
