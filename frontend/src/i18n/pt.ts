/**
 * Portuguese — the source dictionary. Its shape *is* the type every other
 * language must match, so a key added here and forgotten in en.ts fails the
 * build rather than showing up blank.
 *
 * Texts that carry values are functions: plurals and interpolation are then
 * ordinary TypeScript, checked like any other call.
 *
 * Organised by where the text appears (one section per component or area),
 * with `common` for the words every screen repeats.
 */
export const pt = {
  common: {
    back: "Voltar",
    next: "Continuar",
    skip: "Pular",
    close: "Fechar",
    cancel: "Cancelar",
    save: "Salvar",
    loading: "Carregando…",
  },
  header: {
    guide: "guia",
    guideTitle: "Rever o guia da interface",
    changeName: "Trocar nome",
    welcome: "Bem-vindo,",
  },
  language: {
    label: "Idioma",
    switchTo: "Mudar para inglês",
  },
  // Messages the API client raises. It is not a component, so it picks the
  // dictionary itself (see api/client.ts).
  errors: {
    cannotConnect:
      "Não foi possível conectar ao servidor. Verifique se o backend está rodando.",
    validation: "Erros de validação no formulário.",
    markdownExport: (status: number) => `HTTP ${status} ao gerar markdown.`,
    deviceInfo: "Falha ao consultar dispositivos.",
  },
  app: {
    train: {
      simple: "▶ Treinar",
      cv: "▶ Rodar K-fold",
      sweep: "▶ Rodar sweep",
      replicates: "▶ Rodar réplicas",
    },
    error: "Erro",
    // Names a run in the tab title when the server sent no run id.
    unnamedRun: "treino",
  },
  bottomBar: {
    reopenTraining: "🔬 abrir treino",
    running: "Executando…",
    history: "history",
    datasets: "datasets",
    datasetsTitle: "Baixar um dataset para uma pasta local",
    queue: "fila",
    queueTitle: "Ver e reordenar os treinos que estão esperando a GPU",
  },
  deviceSelector: {
    using: "usando",
    loadingTitle: "Carregando dispositivos…",
    cudaUnavailableTitle: "CUDA indisponível — somente CPU",
    loadingShort: "carregando…",
    errorPrefix: (message: string) => `erro: ${message}`,
    cudaNotDetected: "cuda não detectado",
    cpuSubtitle: "Treinar usando processador",
  },
  welcome: {
    hello: "Bem-vindo",
    helloName: (name: string) => `Bem-vindo, ${name}`,
    askName: "Qual é o seu nome?",
    placeholder: "digite aqui",
    nameLabel: "Seu nome",
    enter: "Entrar ↵",
  },
  datasetsOverlay: {
    kicker: "// datasets",
    title: "Obter dados",
    description:
      "Baixe uma vez para uma pasta local; depois aponte o campo de dataset de qualquer task para ela.",
  },
  lightbox: {
    closeTitle: "Fechar (Esc)",
  },
  taskHero: {
    kicker: (short: string) => `// task / ${short}`,
    dataset: "Dataset",
    task: "Task",
  },
  experimentRunner: {
    inProgress: "Treino em andamento...",
    runId: "ID da execução:",
  },
  advancedFields: {
    advanced: "avançado",
    changed: "valores alterados",
  },
  modelAdvice: {
    measured: (architecture: string, optimizer: string, learningRate: number) =>
      `Para ${architecture}, o valor medido é ${optimizer} a ${learningRate}.`,
    apply: (optimizer: string, learningRate: number) =>
      `usar ${optimizer} · ${learningRate}`,
  },
  infoDot: {
    label: "Explicação",
  },
  workersField: {
    label: "Workers",
    setManually: "Definir o número manualmente",
    backToAuto: "Voltar a decidir pela memória livre da máquina",
    automatic: "automático",
    suggestedNow: (n: number) => `≈ ${n} agora`,
  },
};

export type Dict = typeof pt;
