/** O roteiro do guia de primeira execução (ADR-104).
 *
 * Cada passo aponta para um elemento marcado com `data-tour`. O alvo é
 * procurado no DOM na hora: se ele não estiver na tela — porque a tarefa ativa
 * não tem aquele campo, ou porque o painel ainda está carregando — o passo vira
 * um cartão centralizado em vez de sumir. Assim o roteiro é o mesmo para as
 * cinco tarefas sem precisar de uma versão por painel.
 */

export interface TourStep {
  /** Valor de `data-tour` do elemento destacado. Sem ele, o cartão centraliza. */
  anchor?: string;
  title: string;
  body: string;
}

export const TOUR_STEPS: TourStep[] = [
  {
    anchor: "tabs",
    title: "Escolha o tipo de treino",
    body:
      "Cada aba é uma tarefa completa: classificação, detecção, regressão, segmentação e anomalia. Trocar de aba troca o formulário inteiro, as métricas e a cor da interface — nada é compartilhado por acidente entre elas.",
  },
  {
    anchor: "dataset",
    title: "Aponte a pasta do dataset",
    body:
      "Escolha a pasta raiz e o VisionForge procura sozinho as subpastas de treino, validação e teste pelos nomes usuais. Se o seu dataset usa outros nomes, os seletores ao lado deixam você corrigir sem renomear nada no disco.",
  },
  {
    title: "Os parâmetros que importam ficam na frente",
    body:
      "Cada painel mostra primeiro o essencial — épocas, batch, taxa de aprendizado — e guarda o resto em “Avançado”, recolhido. Os valores que já vêm preenchidos foram medidos por tarefa, então começar sem mexer em nada é uma escolha válida. O “i” ao lado de cada rótulo explica o que aquele campo faz.",
  },
  {
    anchor: "device",
    title: "GPU ou CPU",
    body:
      "O VisionForge detecta o que existe na máquina e escolhe a GPU quando ela está disponível. Dá para forçar a CPU aqui: é mais lento, mas roda em qualquer lugar e serve para conferir se um erro é do código ou da placa.",
  },
  {
    anchor: "train",
    title: "Treinar",
    body:
      "O botão roda exatamente o que está selecionado — um treino simples, uma busca em grade, validação cruzada ou réplicas. Enquanto roda, uma tela mostra as curvas ao vivo e você pode minimizá-la ou cancelar sem perder o que já foi feito.",
  },
  {
    anchor: "history",
    title: "Tudo fica salvo",
    body:
      "Cada execução guarda em disco a configuração, as métricas de todas as épocas, os gráficos e os pesos. O histórico deixa você reabrir, comparar duas execuções lado a lado, continuar um treino interrompido e testar o modelo em imagens novas.",
  },
  {
    anchor: "datasets",
    title: "Seus datasets",
    body:
      "Aqui você inspeciona o que tem no disco, vê a distribuição das classes, filtra imagens ruins e prepara divisões novas. Vale abrir antes do primeiro treino: quase todo resultado estranho começa em um dataset desbalanceado.",
  },
];

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
