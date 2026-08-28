/** One line per hyperparameter, and which ones start collapsed.
 *
 * The request that produced this was "conheço apenas metade desses; tá bem
 * difícil de entender, muita coisa junta" — and, explicitly, *not* to remove
 * any. So nothing here hides a parameter: the advanced ones are one click away
 * and travel in the payload with the same value as always.
 *
 * The split is by **how often a value changes**, not by how important it is.
 * The optimizer matters enormously and is still advanced, because it gets
 * decided once and then left alone; epochs and learning rate are what move
 * between one experiment and the next.
 *
 * Living in a data module rather than scattered through JSX is what makes
 * "every field is explained" a test rather than a promise.
 */

export type ParamTier = "basic" | "advanced";

/** What each knob does, in terms of what changes if you move it. */
export const PARAM_HELP: Record<string, string> = {
  // ── básico ────────────────────────────────────────────────────────────────
  epochs: "Quantas vezes o modelo vê o dataset inteiro. Mais épocas aprendem mais, até começarem a decorar.",
  batch_size: "Quantas imagens por passo. Maior estabiliza o gradiente e ocupa mais VRAM; se estourar memória, reduza este primeiro.",
  learning_rate: "O tamanho do passo a cada ajuste. Alto demais diverge, baixo demais nunca chega.",
  seed: "Fixa o sorteio (pesos iniciais, ordem dos dados). Mesmo seed e mesmos dados devolvem o mesmo resultado.",

  // ── avançado: otimização ──────────────────────────────────────────────────
  optimizer: "O algoritmo que aplica o gradiente. adam converge rápido sem ajuste fino; sgd costuma generalizar melhor com tempo.",
  momentum: "Quanto o passo anterior influencia o atual. Suaviza a trajetória e ajuda a atravessar platôs.",
  weight_decay: "Puxa os pesos para perto de zero. Combate overfitting; alto demais impede o modelo de aprender.",
  learning_rate_final: "Fração do learning rate inicial ao término do treino.",
  lrf: "Fração do learning rate inicial ao término do treino.",

  // ── avançado: agendamento ─────────────────────────────────────────────────
  scheduler: "Como o learning rate cai ao longo do treino. Quase sempre ajuda deixar cair.",
  step_size: "De quantas em quantas épocas o learning rate é reduzido.",
  gamma: "Por quanto o learning rate é multiplicado a cada redução.",
  cos_lr: "Faz o learning rate cair numa curva de cosseno em vez de degraus.",
  warmup_epochs: "Épocas iniciais com learning rate crescendo devagar, para o modelo não desestabilizar no começo.",

  // ── avançado: parada e regularização ──────────────────────────────────────
  early_stopping_patience:
    "Épocas seguidas sem melhora antes de encerrar o treino. Deixe 0 (ou vazio) para rodar todas as épocas configuradas.",
  patience: "Épocas sem melhora antes de parar sozinho.",
  label_smoothing: "Suaviza os rótulos para o modelo não ficar excessivamente confiante.",
  dropout: "Desliga neurônios ao acaso durante o treino, forçando o modelo a não depender de poucos.",
  freeze: "Congela as primeiras N camadas. Útil em transfer learning com pouco dado.",

  // ── avançado: mecânica ────────────────────────────────────────────────────
  amp: "Precisão mista: usa 16 bits onde dá. Treina mais rápido e ocupa menos VRAM, com risco baixo de instabilidade.",
  mixed_precision:
    "Faz parte das contas em 16 bits. Acelera o treino e ocupa menos VRAM em GPUs recentes; em modelos sensíveis pode custar precisão numérica.",
  deterministic:
    "Faz o mesmo config com a mesma seed devolver exatamente os mesmos números. Ligado por padrão: medimos o custo e ele é nulo ou negativo em treinos curtos.",
  num_workers: "Processos que carregam as imagens. No Windows cada um recarrega o torch e as DLLs da CUDA, ~1 GB — valor alto demais não deixa o treino lento, ele impede o treino de começar (WinError 1455).",
  workers: "Processos que carregam as imagens. No Windows cada um recarrega o torch e as DLLs da CUDA, ~1 GB — valor alto demais não deixa o treino lento, ele impede o treino de começar (WinError 1455).",
  pin_memory: "Acelera a cópia das imagens para a GPU. Deixe ligado, exceto se faltar RAM.",
  image_size: "Resolução de treino. Maior enxerga mais detalhe e custa VRAM e tempo ao quadrado.",
  nbs: "Batch nominal para normalizar o weight decay quando o batch real é menor.",
  single_cls: "Trata todas as classes como uma só. Serve para medir só a localização das caixas.",
  rect: "Agrupa imagens de proporção parecida em vez de forçar quadrado. Mais rápido, menos uniforme.",
  multi_scale: "Varia a resolução entre passos, para o modelo aguentar objetos de tamanhos diferentes.",
  close_mosaic: "Desliga o mosaico nas últimas N épocas, para o modelo terminar treinando em imagens reais.",
  box: "Peso da perda de localização das caixas.",
  cls: "Peso da perda de classificação.",
  dfl: "Peso da perda de distribuição das bordas da caixa.",
};

/** Which tier each parameter belongs to. Anything absent counts as basic. */
export const PARAM_TIER: Record<string, ParamTier> = {
  epochs: "basic",
  batch_size: "basic",
  learning_rate: "basic",
  seed: "basic",

  optimizer: "advanced",
  momentum: "advanced",
  weight_decay: "advanced",
  learning_rate_final: "advanced",
  lrf: "advanced",
  scheduler: "advanced",
  step_size: "advanced",
  gamma: "advanced",
  cos_lr: "advanced",
  warmup_epochs: "advanced",
  early_stopping_patience: "advanced",
  patience: "advanced",
  label_smoothing: "advanced",
  dropout: "advanced",
  freeze: "advanced",
  amp: "advanced",
  deterministic: "advanced",
  mixed_precision: "advanced",
  num_workers: "advanced",
  workers: "advanced",
  pin_memory: "advanced",
  nbs: "advanced",
  single_cls: "advanced",
  rect: "advanced",
  multi_scale: "advanced",
  close_mosaic: "advanced",
  box: "advanced",
  cls: "advanced",
  dfl: "advanced",
};

/** Whether a parameter starts collapsed.
 *
 * An unclassified name counts as basic: a field nobody thought about must stay
 * visible rather than disappear by accident.
 */
export function isAdvanced(key: string): boolean {
  return PARAM_TIER[key] === "advanced";
}

/** The explanation for a parameter, or undefined if it has none yet. */
export function paramHelp(key: string): string | undefined {
  return PARAM_HELP[key];
}

/** Whether any advanced field differs from its default.
 *
 * The advanced section starts collapsed, but hiding a value the researcher
 * deliberately set — or that arrived with an imported YAML — would be worse
 * than the clutter the collapsing exists to remove. So a tuned form opens.
 */
export function hasNonDefaultAdvanced(
  form: Record<string, unknown>,
  defaults: Record<string, unknown>,
): boolean {
  return Object.keys(form).some((key) => {
    if (!isAdvanced(key)) return false;
    // A default we do not know is not evidence of a change. Nested objects
    // (the scheduler) carry no `default` at their own level in the schema, and
    // counting them as different made the section open every single time —
    // which is the same as not having a section.
    if (defaults[key] === undefined) return false;
    return JSON.stringify(form[key]) !== JSON.stringify(defaults[key]);
  });
}
