// Help texts shared by two field names (the same knob under two spellings, one
// per task family). One constant each, so the copies cannot drift apart.
const workersHelp =
  "Processos que carregam as imagens em paralelo. No automático o VisionForge divide a memória livre da máquina pelo custo de um worker — no Windows cada um recarrega o torch e as DLLs da CUDA, ~1 GB, e um número alto demais não deixa o treino lento: impede o treino de começar (WinError 1455).";
const lrFinalHelp = "Fração do learning rate inicial ao término do treino.";

// What the collapse note says about the suggested setting (components/ModelAdvice). "Treina normal"
// is a claim, so it is only ever a number, and only for the model whose recovery was run on the
// same setup; everywhere else the setting is just suggested.
const suggestedRemedy = (
  recoveredAccuracy: number | null,
  optimizer: string,
  learningRate: number,
): string =>
  recoveredAccuracy === null
    ? `Sugerimos ${optimizer} a ${learningRate}.`
    : `Com ${optimizer} a ${learningRate}, a acurácia foi ${recoveredAccuracy.toFixed(2)} nas mesmas condições.`;

// The measured part of the collapse note. Every number was measured on classification, so on any
// other form the note says "Em classificação" and that this task was not measured, and it never
// carries the recovery over: "nas mesmas condições" would not be true there.
const measuredNote = (facts: CollapseFacts, whatHappened: string): string => {
  const { architecture, measuredOn, accuracy, recoveredAccuracy, optimizer, learningRate } = facts;
  const result = `(acurácia ${accuracy.toFixed(2)} em 4 classes)`;
  const measuredItself = architecture.toLowerCase() === measuredOn;
  if (facts.task !== "classification") {
    const who = measuredItself ? `o ${measuredOn}` : `o ${measuredOn}, da mesma família do ${architecture},`;
    const notMeasured = measuredItself
      ? "esta tarefa não foi medida"
      : `nem o ${architecture} nem esta tarefa foram medidos`;
    return `Em classificação, ${who} ${whatHappened} com Adam a 1e-3 ${result}; ${notMeasured}. ${suggestedRemedy(null, optimizer, learningRate)}`;
  }
  const finding = measuredItself
    ? `${architecture} ${whatHappened} com Adam a 1e-3 ${result}.`
    : `${architecture}: o ${measuredOn}, da mesma família, ${whatHappened} com Adam a 1e-3 ${result}; este modelo não foi medido.`;
  return `${finding} ${suggestedRemedy(measuredItself ? recoveredAccuracy : null, optimizer, learningRate)}`;
};

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
    skip: "Pular",
    close: "Fechar",
    cancel: "Cancelar",
  },
  // The five task families by name: the history tabs and the queue rows.
  taskNames: {
    classification: "Classificação",
    detection: "Detecção",
    regression: "Regressão",
    segmentation: "Segmentação",
    anomaly: "Anomalia",
  },
  // The button that downloads a run's model card, in the run detail and in the
  // results view.
  modelCard: {
    button: "↓ markdown",
    title: "Baixar model card (markdown) deste run",
  },
  // The confidence interval a metric's tooltip explains (the run detail and the
  // results view show the same one).
  metrics: {
    // The bootstrap draws 1000 resamples by default and is skipped below 2: plural only.
    ciTooltip: (percent: number, resamples: number, samples: number) =>
      `IC ${percent}% por bootstrap percentil: ` +
      `${resamples} reamostragens ${samples === 1 ? "da única imagem" : `das ${samples} imagens`} de teste. ` +
      `Mede o ruído de amostragem do split com este modelo fixo — não a ` +
      `variação entre treinos, que réplicas com várias seeds medem.`,
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
    // Each language by its own name, whichever is on: they name the switch's two
    // buttons for a screen reader.
    names: { pt: "Português", en: "English" },
  },
  // Messages the API client raises. It is not a component, so it picks the
  // dictionary itself (see api/client.ts).
  errors: {
    cannotConnect:
      "Não foi possível conectar ao servidor. Verifique se o backend está rodando.",
    validation: "Erros de validação no formulário.",
    markdownExport: (status: number) => `HTTP ${status} ao gerar markdown.`,
    deviceInfo: "Falha ao consultar dispositivos.",
    // The preprocessing and augmentation previews fail with the same sentence.
    previewFailed: "Falha ao gerar preview.",
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
    unnamedRun: "sem nome",
  },
  // The notice that stands in for the main content when it crashes
  // (components/ErrorBoundary.tsx).
  errorBoundary: {
    title: "Algo deu errado nesta tela",
    body: "Um erro inesperado interrompeu esta parte da interface. Recarregue a página para continuar.",
    reload: "Recarregar página",
  },
  bottomBar: {
    reopenTraining: "🔬 abrir treino",
    running: "Executando…",
    history: "history",
    datasets: "datasets",
    datasetsTitle: "Baixar um dataset para uma pasta local",
    queue: "fila",
    queueTitle:
      "Ver o treino em execução e os que esperam a GPU — dá para parar um ou remover outro",
  },
  deviceSelector: {
    using: "usando",
    loadingTitle: "Carregando dispositivos…",
    cudaUnavailableTitle: "CUDA indisponível — somente CPU",
    cudaTitle: (version: string, gpus: number) =>
      `CUDA ${version} · ${gpus} ${gpus === 1 ? "GPU" : "GPUs"}`,
    cudaSummary: (version: string, gpus: number) =>
      `cuda ${version} · ${gpus} ${gpus === 1 ? "gpu" : "gpus"}`,
    gpuName: (index: number, name: string) => `GPU ${index} · ${name}`,
    multiGpu: (gpus: number) => `Multi-GPU (${gpus} GPUs)`,
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
    inProgress: "Treino em andamento…",
    runId: "ID da execução:",
  },
  advancedFields: {
    advanced: "avançado",
    changed: "valores alterados",
  },
  modelAdvice: {
    // Not "measured": only a few models were run, and this is shown for any other.
    suggested: (architecture: string, optimizer: string, learningRate: number) =>
      `Para ${architecture}, sugerimos ${optimizer} a ${learningRate}.`,
    apply: (optimizer: string, learningRate: number) =>
      `usar ${optimizer} · ${learningRate}`,
    // What /api/model/defaults finds, worded here from the numbers it returns so
    // the note is in the language of the interface (its own `note` is not shown).
    // `optimizer` and `learningRate` are the suggestion, not the form's values.
    // Only what was measured is said (ADR-099/100): one model per family, on 4
    // classes. `measuredOn` is that model; when it is not `architecture`, the
    // note says so rather than letting a sibling borrow its number. The recovery
    // (`recoveredAccuracy`) is the measured model's alone: a sibling only gets the suggestion.
    collapseMeasured: (facts: CollapseFacts) => measuredNote(facts, "previu uma classe só"),
    // ViT did not collapse: 0.41 on 4 classes is above the 0.25 of a one-class prediction, so it
    // "learned little", not "learned nothing".
    failsToLearnMeasured: (facts: CollapseFacts) => measuredNote(facts, "aprendeu pouco"),
    // The family shares the remedy but was never run: no number at all.
    unmeasured: (architecture: string, optimizer: string, learningRate: number) =>
      `${architecture}: Adam a 1e-3 não foi medido para esta família; sugerimos ${optimizer} a ${learningRate}, o mesmo das famílias de atenção medidas.`,
    upscaling: (medianSide: number) =>
      `As imagens têm cerca de ${medianSide}px de lado; treinar acima disso amplia a imagem sem acrescentar detalhe.`,
  },
  infoDot: {
    label: "Explicação",
  },
  // The reveal toggle inside a credential field (components/controls/TextField.tsx).
  textField: {
    showSecret: "Mostrar o valor digitado",
    hideSecret: "Ocultar o valor digitado",
  },
  workersField: {
    label: "Workers",
    setManually: "Definir o número manualmente",
    backToAuto: "Voltar a decidir pela memória livre da máquina",
    automatic: "automático",
    suggestedNow: (n: number) => `≈ ${n} agora`,
  },
  // What each hyperparameter does, in terms of what changes if you move it. Keys
  // are the backend's field names (plus the dot-path of a field whose meaning
  // depends on where it sits). `satisfies` keeps the literal keys, so en.ts must
  // list exactly the same ones; the lookup widens them back to open strings.
  paramHelp: {
    // ── básico ────────────────────────────────────────────────────────────────
    epochs:
      "Quantas vezes o modelo vê o dataset inteiro. Mais épocas aprendem mais, até começarem a decorar.",
    batch_size:
      "Quantas imagens por passo. Maior estabiliza o gradiente e ocupa mais VRAM; se estourar memória, reduza este primeiro.",
    learning_rate:
      "O tamanho do passo a cada ajuste. Alto demais diverge, baixo demais nunca chega.",
    seed:
      "Fixa o sorteio (pesos iniciais, ordem dos dados). Mesmo seed e mesmos dados devolvem o mesmo resultado.",

    // ── avançado: otimização ──────────────────────────────────────────────────
    optimizer:
      "O algoritmo que aplica o gradiente. adam converge rápido sem ajuste fino; sgd costuma generalizar melhor com tempo.",
    momentum:
      "Quanto o passo anterior influencia o atual. Suaviza a trajetória e ajuda a atravessar platôs.",
    weight_decay:
      "Puxa os pesos para perto de zero. Combate overfitting; alto demais impede o modelo de aprender.",
    learning_rate_final: lrFinalHelp,
    lrf: lrFinalHelp,

    // ── avançado: agendamento ─────────────────────────────────────────────────
    scheduler: "Como o learning rate cai ao longo do treino. Quase sempre ajuda deixar cair.",
    step_size: "De quantas em quantas épocas o learning rate é reduzido.",
    gamma: "Por quanto o learning rate é multiplicado a cada redução.",
    cos_lr: "Faz o learning rate cair numa curva de cosseno em vez de linearmente.",
    warmup_epochs:
      "Épocas iniciais com learning rate crescendo devagar, para o modelo não desestabilizar no começo.",

    // ── avançado: parada e regularização ──────────────────────────────────────
    early_stopping_patience:
      "Épocas seguidas sem melhora antes de encerrar o treino. Deixe 0 (ou vazio) para rodar todas as épocas configuradas.",
    patience: "Épocas sem melhora antes de parar sozinho.",
    // O `patience` do scheduler (reduzir no platô) não é o do early stopping; o
    // ParamPanel procura o dot-path antes do nome da folha.
    "training.scheduler.patience": "Épocas sem melhora antes de reduzir o learning rate.",
    label_smoothing: "Suaviza os rótulos para o modelo não ficar excessivamente confiante.",
    dropout:
      "Desliga neurônios ao acaso durante o treino, forçando o modelo a não depender de poucos.",
    freeze: "Congela as primeiras N camadas. Útil em transfer learning com pouco dado.",

    // ── avançado: mecânica ────────────────────────────────────────────────────
    amp:
      "Precisão mista: usa 16 bits onde dá. Treina mais rápido e ocupa menos VRAM, com risco baixo de instabilidade.",
    mixed_precision:
      "Faz parte das contas em 16 bits. Acelera o treino e ocupa menos VRAM em GPUs recentes; em modelos sensíveis pode custar precisão numérica.",
    deterministic:
      "Faz o mesmo config com a mesma seed devolver exatamente os mesmos números. Ligado por padrão: medimos o custo e ele é nulo ou negativo em treinos curtos.",
    base_dir:
      "Pasta raiz do dataset. Dentro dela ficam as subpastas de treino, validação e teste — o VisionForge procura os nomes usuais (train/val/test, treino/validacao/teste) e preenche sozinho quando encontra. O conteúdo de cada subpasta depende da tarefa: uma pasta por classe na classificação, imagens e labels na detecção, imagens e máscaras na segmentação.",
    train_dir: "Subpasta usada para ajustar os pesos. É a única com que o modelo aprende.",
    val_dir:
      "Subpasta usada a cada época para medir o progresso e escolher o melhor checkpoint. Não entra no ajuste dos pesos.",
    test_dir:
      "Subpasta avaliada uma única vez, no fim. Serve para reportar o resultado sem que ele tenha influenciado nenhuma escolha.",
    coreset_ratio:
      "Quanto do \"normal\" o PatchCore guarda para comparar depois. Ele corta as imagens de treino em pedaços pequenos e mantém uma amostra deles, a mais variada possível; uma imagem nova é anômala quando algum pedaço dela não se parece com nada guardado. 1% é o valor do artigo original. Aumentar deixa o banco mais completo, mas o tempo de montagem cresce na mesma proporção: 10% leva dez vezes mais.",
    num_workers: workersHelp,
    workers: workersHelp,
    pin_memory: "Acelera a cópia das imagens para a GPU. Deixe ligado, exceto se faltar RAM.",
    image_size: "Resolução de treino. Maior enxerga mais detalhe e custa VRAM e tempo ao quadrado.",
    nbs: "Batch nominal para normalizar o weight decay quando o batch real é menor.",
    single_cls: "Trata todas as classes como uma só. Serve para medir só a localização das caixas.",
    rect:
      "Agrupa imagens de proporção parecida em vez de forçar quadrado. Mais rápido, menos uniforme.",
    multi_scale:
      "Varia a resolução entre passos, para o modelo aguentar objetos de tamanhos diferentes.",
    close_mosaic:
      "Desliga o mosaico nas últimas N épocas, para o modelo terminar treinando em imagens reais.",
    box: "Peso da perda de localização das caixas.",
    cls: "Peso da perda de classificação.",
    dfl: "Peso da perda de distribuição das bordas da caixa.",
  } satisfies Record<string, string>,
  // The hyperparameter form (components/ParamPanel.tsx). Sentences with inline
  // emphasis are written whole, with `code`, **strong** and __emphasis__ marks
  // that ParamPanel's <Rich> turns into elements, so a translation can reorder
  // the words around them.
  paramPanel: {
    sectionLabels: {
      model: "Modelo",
      training: "Treinamento",
      data: "Dataset",
      output: "Saída",
      classification: "Classificação",
      transforms: "Transformações",
    },
    // Dot-path keys win over leaf names: `model.name` is the architecture.
    fieldLabels: {
      "model.name": "Arquitetura",
      name: "Nome do experimento",
      task: "Tipo de tarefa",
      block: "Bloco",
      num_classes: "Nº de classes",
      pretrained: "Pesos pré-treinados",
      weights_path: "Checkpoint custom (.pth)",
      learning_rate: "Learning rate",
      epochs: "Épocas",
      batch_size: "Batch size",
      early_stopping_patience: "Early stop (paciência)",
      optimizer: "Otimizador",
      weight_decay: "Weight decay",
      seed: "Seed",
      deterministic: "Determinístico",
      mixed_precision: "Precisão mista (AMP)",
      kind: "Tipo",
      step_size: "Step size",
      gamma: "Gamma",
      patience: "Paciência",
      factor: "Fator",
      min_lr: "LR mínimo",
      base_dir: "Pasta base",
      train_dir: "Subpasta treino",
      val_dir: "Subpasta validação",
      test_dir: "Subpasta teste",
      num_workers: "Workers",
      pin_memory: "Pin memory",
      image_size: "Tamanho da imagem",
      horizontal_flip: "Flip horizontal",
      rotation_degrees: "Rotação (graus)",
      color_jitter: "Color jitter",
      normalize_mean: "Normalização (média)",
      normalize_std: "Normalização (std)",
      n_folds: "Nº de folds",
      stratified: "Stratified",
      shuffle: "Shuffle",
      fold_seed: "Fold seed",
      mode: "Modo",
      unfreeze_from_layer: "Descongelar a partir de",
      backbone_lr_multiplier: "LR do backbone (×)",
      model_names: "Arquiteturas",
      metric: "Métrica de ranking",
    },
    kickers: {
      strategy: "// estratégia de experimento",
      scheduler: "// learning-rate scheduler",
      crossValidation: "// k-fold cross-validation",
      transferLearning: "// transfer learning",
      model: "// modelo",
      training: "// treinamento",
      dataset: "// dataset",
      classes: "// classes",
      image: "// imagem",
      augmentation: "// data augmentation",
      comingSoon: "// em breve",
    },
    blocks: {
      simple: "Treino simples",
      crossValidation: "K-Fold (CV)",
      transferLearning: "Transfer learning",
      gridSearch: "Grid search",
      randomSearch: "Random search",
    },
    blockHints: {
      crossValidation:
        "Treina N modelos em N folds da pasta de treino. Validação por fold; `normalize_mean/std` são recalculados por fold para evitar data leakage. Não usa o split de teste — agregação em `cv_summary.json`.",
      transferLearning:
        "Feature extraction (só treina o head) ou fine-tuning (head + backbone parcial com LR menor). Útil em datasets pequenos, sem destruir as features pré-treinadas. Em feature extraction os pesos do backbone não se movem, mas as estatísticas de BatchNorm se recalibram no seu dataset — o backbone fica congelado, não idêntico.",
      gridSearch:
        'Treina **uma vez por combinação** do produto cartesiano do espaço definido abaixo. Cada chave é um dot-path (ex: `training.learning_rate`); o valor é uma lista. Cuidado: 3×3×2 já são 18 treinos. Para **comparar arquiteturas**, adicione valores ao campo "Arquitetura" (botão "+ valor ao grid") — um grid de um eixo só; compare os runs no histórico.',
      randomSearch:
        "Amostra `n_trials` configurações independentes do espaço abaixo. Cada parâmetro tem um tipo: `uniform`, `log_uniform` (LR e weight_decay) ou `choice` (listas discretas).",
    },
    weights: {
      label: "Checkpoint custom (.pth)",
      clearTitle: "Remover checkpoint custom (volta a usar pesos pretrained / random)",
      clear: "limpar",
      placeholder: "opcional — sobrescreve ImageNet",
      browse: "Escolher",
      cancelled: "Cancelado.",
      pickFailed: "Falha ao abrir o seletor.",
    },
    grid: {
      addValue: "+ valor ao grid",
      addAnother: "+ valor",
      removeValue: "Remover valor",
      axisTag: (values: number) => `grade · ${values} ${values === 1 ? "valor" : "valores"}`,
      banner:
        "**Grid search ativo.** Clique em `+ valor ao grid` nos hiperparâmetros de __Modelo__ e __Treinamento__ para varrer múltiplos valores.",
      trials: (n: number) =>
        `// ${n} trial${n === 1 ? "" : "s"}${n > 12 ? " ⚠️ alto" : ""}`,
    },
    randomSearch: {
      kicker: (trials: number) =>
        `// random search · ${trials} trial${trials === 1 ? "" : "s"}`,
      add: "+ adicionar",
      trialsLabel: "n_trials",
      seedLabel: "seed",
      empty: 'Espaço de busca vazio — clique em "+ adicionar".',
      example: "Exemplo: `training.learning_rate` = log_uniform(1e-5, 1e-2).",
      keyPlaceholder: "dot-path (ex: training.learning_rate)",
      choicesPlaceholder: "csv: resnet18, resnet50",
      low: "low",
      high: "high",
      removeRow: "Remover linha",
    },
    modelComparison: {
      kicker: (selected: number) =>
        `// comparação de modelos · ${selected} selecionado${selected === 1 ? "" : "s"}`,
      needTwo: "Selecione pelo menos 2 arquiteturas para iniciar a comparação.",
    },
    lockedClasses: {
      title: "Tarefa binária — fixado em 1",
      badge: "🔒 binário",
    },
    normalizePlaceholder: "ex: 0.485, 0.456, 0.406",
    importWarnings: (count: number, summary: string, extra: number) =>
      `YAML importado com ${count} ${count === 1 ? "aviso estrutural" : "avisos estruturais"}:\n${summary}${extra > 0 ? `\n…(+${extra} mais)` : ""}`,
    // The two lines that follow the list, each only when it applies (the counts come from
    // reviewImportedConfig): optional values left out use their defaults; required fields
    // left without a value are what the backend will refuse.
    importDropped: (count: number) =>
      count === 1
        ? "1 valor com tipo inválido foi ignorado; o campo usa o valor padrão."
        : `${count} valores com tipo inválido foram ignorados; os campos usam o valor padrão.`,
    importMissing: (count: number) =>
      count === 1
        ? "1 campo obrigatório está sem valor: preencha-o antes de treinar, senão o backend rejeita a configuração."
        : `${count} campos obrigatórios estão sem valor: preencha-os antes de treinar, senão o backend rejeita a configuração.`,
    unavailable: (task: string) => `${task} ainda não está disponível`,
    unavailableBody: "Esta tarefa será implementada em uma próxima fase do VisionForge.",
    loadingSchema: "carregando schema…",
    hiddenParams: (n: number) =>
      `${n} ${n === 1 ? "parâmetro oculto" : "parâmetros ocultos"} — ligue para ajustar`,
    fieldErrors: (n: number) => `${n} ${n === 1 ? "campo" : "campos"} com erro:`,
  },
  // The panel for one run (components/RunDetailPanel.tsx). Config keys, measure
  // names and file names (opset_version, max_diff, best_model.onnx) read the same
  // in every language; they are repeated in both dictionaries only because every
  // word on screen comes from here. Metric names (F1, AUC-ROC, mAP@50) stay in the
  // component: only the words around them are translated.
  runDetail: {
    back: "← histórico",
    loading: "carregando…",
    loadFailed: "Falha ao carregar detalhes.",
    browse: "📁 Escolher",
    copy: "copy",
    copyPath: "Copiar caminho",
    imageFolder: "Pasta de imagens",
    resume: {
      button: "▶ retomar",
      title: (done: number, total: number) =>
        `Continuar da época ${done} até ${total}, na mesma pasta`,
      titleNoTotal: "Continuar este run na mesma pasta",
      queuing: "Enfileirando a continuação…",
      running: "Continuando este run — acompanhe no painel de treino.",
      queued: "Na fila: começa quando o treino atual terminar.",
      failed: "Falha ao retomar.",
    },
    // The native folder picker the three forms below share.
    picker: {
      opening: "Abrindo seletor…",
      cancelled: "Cancelado.",
      picked: (path: string) => `Pasta: ${path}`,
      failed: "Falha ao escolher pasta.",
    },
    dataset: {
      title: "Dataset",
      name: "Nome",
      path: "Caminho",
      contents: "Conteúdo",
      files: (n: number | null | undefined, size: string) => `${n ?? "—"} ${n === 1 ? "arquivo" : "arquivos"} · ${size}`,
      fingerprint: "Fingerprint",
      noFingerprint: "sem fingerprint — run anterior a 26/07/2026",
    },
    location: {
      title: "Localização no disco",
      runFolder: "Pasta do run",
      checkpoint: "Checkpoint",
      deviceUsed: "Dispositivo usado",
      env: (key: string) => `env · ${key}`,
    },
    training: {
      title: "Configuração de treino",
      transferLearning: "transfer learning",
    },
    pipeline: {
      title: "Pipeline aplicado (preprocessing + augmentation)",
      preprocessing: "// pré-processamento (ordem)",
      augmentation: "// augmentation & normalize",
    },
    onnx: {
      title: "Exportar para ONNX",
      open: "↗ exportar onnx",
      // `file` follows this text, in <code>.
      hint: "Converte o checkpoint para ONNX, valida diff numérico contra PyTorch e mede latência de inferência. O arquivo é salvo ao lado do checkpoint como",
      file: "best_model.onnx",
      opsetVersion: "opset_version",
      benchmarkRuns: "benchmark_runs",
      dynamicAxes: "dynamic_axes",
      validate: "validate",
      benchmark: "benchmark",
      run: "▶ Rodar export",
      running: "Exportando…",
      exporting: "Exportando para ONNX…",
      saved: (path: string) => `ONNX salvo em ${path}`,
      failed: "Falha ao exportar ONNX.",
      stats: {
        fileSize: "file_size",
        maxDiff: "max_diff",
        onnxLatency: "onnx latency μ",
        onnxP95: "onnx p95",
        torchLatency: "torch latency μ",
        speedup: "speedup (torch/onnx)",
        runs: "n_runs",
      },
    },
    batch: {
      title: "Inferência em lote (CSV)",
      open: "+ inferência em lote",
      folderPlaceholder: "ex: C:/datasets/inbox",
      recursive: "recursive (subpastas)",
      run: "▶ Rodar inferência",
      running: "Processando…",
      hint: "Roda o checkpoint sobre uma pasta de imagens e escreve um CSV com uma linha por imagem e a saída do modelo para a tarefa (classe e probabilidades, valores previstos ou score e decisão de anomalia). Útil para processar batches de dados novos sem retreinar.",
      needFolder: "Informe a pasta de imagens para inferência.",
      starting: "Rodando inferência em lote…",
      done: (ok: number, csv: string) => `${ok} ${ok === 1 ? "imagem processada" : "imagens processadas"} · CSV em ${csv}`,
      doneWithFailures: (ok: number, failed: number, csv: string) =>
        `${ok} ok · ${failed} ${failed === 1 ? "falhou" : "falharam"} · CSV em ${csv}`,
      failed: "Falha na inferência em lote.",
      processed: "processadas",
      failedCount: "falharam",
      csv: "csv",
      failedFiles: (n: number) =>
        `${n} ${n === 1 ? "arquivo falhou" : "arquivos falharam"} (clique para ver até 5)`,
      more: (n: number) => `…+${n} mais`,
    },
    gradcam: {
      title: "Grad-CAM (explicabilidade)",
      open: "🔥 Grad-CAM",
      folderPlaceholder: "ex: C:/datasets/amostras",
      samples: "Nº de amostras (1–64)",
      run: "▶ Gerar Grad-CAM",
      running: "Gerando…",
      hint: "Gera mapas de calor Grad-CAM sobre imagens de exemplo, destacando as regiões que mais influenciaram a saída do modelo: a classe predita na classificação, o primeiro alvo na regressão e a classe 0 na segmentação. Útil para interpretar o que o modelo aprendeu.",
      needFolder: "Informe a pasta de imagens.",
      starting: "Gerando mapas Grad-CAM…",
      done: (count: number, layer: string) => `${count} ${count === 1 ? "mapa gerado" : "mapas gerados"} · camada ${layer}`,
      failed: "Falha ao gerar Grad-CAM.",
      // Caption under an overlay: the model's answer, and the true class when known.
      classNumber: (n: number) => `classe ${n}`,
      predictedIs: (label: string) => `predito: ${label}`,
      actual: "real",
      predicted: "predito",
    },
    metrics: {
      title: "Métricas",
      none: "Sem métricas registradas.",
      // Keys are the backend's metric names. F1, Recall, AUC-ROC and the mAP
      // family are not here: they read the same everywhere.
      labels: {
        accuracy: "Acurácia",
        precision: "Precisão",
        best_val_loss: "Melhor val loss",
        best_epoch: "Melhor epoch",
        total_epochs: "Epochs treinados",
      },
      // `name` is a metric label: "Acurácia (teste)", "F1 (teste)".
      onTestSet: (name: string) => `${name} (teste)`,
    },
    graphs: {
      title: "Gráficos (clique para expandir)",
    },
    tests: {
      title: "Testes neste modelo",
      open: "+ testar",
      empty: "Nenhum teste executado ainda neste modelo.",
      folder: "Pasta de teste",
      manifest: "Manifesto de teste (.csv)",
      folderPlaceholder: "ex: C:/datasets/coffee_v2/test",
      manifestPlaceholder: "ex: C:/datasets/idade/test.csv",
      // What the folder has to contain, by the task the run was trained on.
      folderHint: {
        detection: "Uma pasta no layout YOLO: imagens e os .txt de rótulo correspondentes.",
        segmentation:
          "Uma pasta com as subpastas de imagens e de máscaras, pareadas pelo nome do arquivo.",
        anomaly: "Uma pasta com a subpasta de imagens normais e as de defeito, como no treino.",
        regression:
          "O .csv com a coluna de imagem e a(s) coluna(s) alvo. Os caminhos das imagens são resolvidos a partir da subpasta de imagens ao lado do .csv, como no treino.",
        classification: "Uma pasta com uma subpasta por classe — a mesma convenção do treino.",
      },
      label: "Nome (opcional)",
      labelPlaceholder: "ex: holdout_2026",
      run: "▶ Rodar teste",
      running: "Avaliando…",
      needFolder: "Informe a pasta de teste ou o manifesto .csv.",
      starting: "Avaliando modelo no novo dataset…",
      recorded: (id: string) => `Teste registrado: ${id}`,
      failed: "Falha no teste.",
    },
    cv: {
      title: (ok: number, total: number, failed: number) =>
        `Cross-validation · ${ok}/${total} folds ok${failed > 0 ? ` · ${failed} ${failed === 1 ? "falhou" : "falharam"}` : ""}`,
      meanAccuracy: "Acurácia média ± std",
      meanF1: "F1 média ± std",
      fold: "Fold",
      train: "train",
      val: "val",
      valLoss: "val_loss",
      accuracy: "accuracy",
      status: "status",
    },
  },
  // The detection form (components/DetectionPanel.tsx). Ultralytics hyperparameter
  // names and their short labels (lrf, Cosine LR, Mosaic, imgsz…) read the same in
  // every language and are repeated in both dictionaries; the hints beside them and
  // the section titles are what get translated. What each field does comes from
  // `paramHelp`, not from here.
  detectionPanel: {
    namePlaceholder: "detection_001",
    model: {
      title: "Modelo · detecção",
      backend: "Backend",
      backendHint: "origem do modelo",
      architecture: "Arquitetura",
      architectureHint: "detector",
      pretrained: "Pesos pré-treinados",
      pretrainedHint: "COCO",
    },
    training: {
      title: "Treinamento",
      epochs: "Épocas",
      batchSize: "Batch size",
      batchSizeHint: "qualquer inteiro",
      learningRate: "Learning rate",
      learningRateHint: "lr0",
      seed: "Seed",
      patience: "Patience",
      patienceHint: "early stop",
      deterministic: "Determinístico",
      deterministicHint: "reprodutível",
      optimizer: "Optimizer",
      optimizerHint: "auto = Ultralytics escolhe",
      momentum: "Momentum",
      momentumHint: "SGD momentum / Adam β1",
      weightDecay: "Weight decay",
      weightDecayHint: "L2",
    },
    schedule: {
      title: "Schedule de LR & pesos de loss",
      lrf: "lrf",
      lrfHint: "LR final = lr0 × lrf",
      cosLr: "Cosine LR",
      cosLrHint: "schedule cosseno",
      warmupEpochs: "Warmup epochs",
      warmupMomentum: "Warmup momentum",
      warmupBiasLr: "Warmup bias LR",
      boxGain: "Box loss gain",
      boxHint: "box",
      clsGain: "Cls loss gain",
      clsHint: "cls",
      dflGain: "DFL loss gain",
      dflHint: "dfl",
    },
    mechanics: {
      title: "Regularização & mecânica",
      labelSmoothing: "Label smoothing",
      dropout: "Dropout",
      nbs: "Nominal batch (nbs)",
      freeze: "Freeze layers",
      freezeHint: "0 = nenhuma",
      closeMosaic: "Close mosaic",
      closeMosaicHint: "desliga mosaico nas últimas N épocas",
      amp: "AMP",
      ampHint: "precisão mista",
      singleCls: "Single class",
      singleClsHint: "trata tudo como 1 classe",
      rect: "Rect",
      rectHint: "batches retangulares",
      multiScale: "Multi-scale",
      multiScaleHint: "varia imgsz ±50%",
    },
    dataset: {
      title: "Dataset",
      source: "Fonte do dataset",
      folderOption: "Pasta YOLO",
      yamlOption: "data.yaml",
      folderHint: "gera o data.yaml a partir da pasta",
      yamlHint: "usa um data.yaml existente",
      baseDir: "Pasta base",
      baseDirPlaceholder: "…/dataset (images/train, images/val)",
      baseDirHint: "raiz YOLO",
      browseFolder: "📁 Escolher",
      yamlFile: "Arquivo data.yaml",
      yamlPlaceholder: "…/dataset/data.yaml",
      yamlFileHint: "Ultralytics / Roboflow",
      browseYaml: "📄 Escolher",
      imageSize: "Image size",
      imageSizeHint: "imgsz",
      numClasses: "Nº de classes",
      numClassesHint: "preenchido pelo dataset abaixo; editável",
      // `nc` is the data.yaml key that holds the class count.
      yamlNote:
        "O data.yaml define splits e nomes de classe. Confirme que `nc` bate com o nº de classes acima.",
      preprocessingNote:
        "Os filtros são aplicados uma vez e gravados numa cópia temporária do dataset, usada só durante o treino e apagada ao fim. Os rótulos vão junto, inalterados.",
    },
    augmentation: {
      title: "Data augmentation",
      toggle: "Augmentation",
      toggleHint: "desligada, os valores abaixo ficam guardados",
      hsvHue: "HSV — hue",
      hsvSaturation: "HSV — saturation",
      hsvValue: "HSV — value",
      degrees: "Degrees",
      translate: "Translate",
      scale: "Scale",
      shear: "Shear",
      perspective: "Perspective",
      flipUpDown: "Flip up-down",
      flipLeftRight: "Flip left-right",
      bgrSwap: "BGR swap",
      mosaic: "Mosaic",
      mixup: "Mixup",
      copyPaste: "Copy-paste",
      autoAugment: "Auto augment",
      randomErasing: "Random erasing",
      probability: "probabilidade",
      disabled: "Desativado",
    },
  },
  // The plot files the backend writes, by file name. The run panel
  // (RunDetailPanel) and the results sheet (ResultsView) show the same ones under
  // the same names.
  plots: {
    labels: {
      "loss.png": "Loss (train + val)",
      "accuracy.png": "Accuracy (train + val)",
      "confusion_matrix.png": "Matriz de confusão",
      "confusion_matrix_normalized.png": "Matriz de confusão (normalizada)",
      "roc_curve.png": "Curva ROC",
      "precision_recall_curve.png": "Curva Precision-Recall",
      // Detection (Ultralytics / torchvision).
      "results.png": "Curvas de treino (loss + mAP)",
      "BoxPR_curve.png": "Curva Precision-Recall (box)",
      "BoxF1_curve.png": "Curva F1 (box)",
      // Test-set diagnostics per task.
      "auroc.png": "AUROC por época",
      "BoxP_curve.png": "Curva Precision (box)",
      "BoxR_curve.png": "Curva Recall (box)",
      "val_batch0_pred.jpg": "Predições na validação",
      "pred_vs_true.png": "Predito vs real",
      "residuals.png": "Distribuição dos resíduos",
      "iou_per_class.png": "IoU por classe",
      "score_histogram.png": "Escores: normal vs defeito",
    },
  },
  // The results sheet that opens after a run (components/ResultsView.tsx). Metric
  // names, the code-style table columns (train_size, val_loss…) and the `// `
  // section prefix read the same in every language.
  resultsView: {
    title: "// resultados",
    // Keys are the backend's metric names.
    metricLabels: {
      best_val_loss: "Best Val Loss",
      best_epoch: "Best Epoch",
      total_epochs: "Total Epochs",
      test_accuracy: "Accuracy",
      test_f1: "F1 Score",
      test_precision: "Precision",
      test_recall: "Recall",
      test_auc_roc: "AUC-ROC",
      map50: "mAP@50",
      map50_95: "mAP@50-95",
      precision: "Precision (box)",
      recall: "Recall (box)",
      box_loss: "Box loss (val)",
    },
    graphsTitle: "// gráficos · clique para expandir",
    // A report whose shape the view does not know is shown as JSON under this.
    reportTitle: "// report",
    // Columns the report tables share.
    cols: {
      rank: "Rank",
      architecture: "Arquitetura",
      time: "tempo (s)",
      status: "status",
      seed: "seed",
      overrides: "overrides",
    },
    // A trial's or fold's outcome cell.
    outcome: {
      ok: "ok",
      failed: (error: string) => `falhou · ${error}`,
    },
    // K-fold on the classification task.
    cv: {
      title: (ok: number, total: number, failed: number) =>
        `// k-fold cross-validation · ${ok}/${total} folds ok${failed > 0 ? ` · ${failed} ${failed === 1 ? "falhou" : "falharam"}` : ""}`,
      accuracyMeanStd: "Acurácia (média ± std)",
      f1MeanStd: "F1 (média ± std)",
      fold: "Fold",
      trainSize: "train_size",
      valSize: "val_size",
      valLoss: "val_loss",
      accuracy: "accuracy",
    },
    // K-fold on the other tasks.
    taskCv: {
      title: (ok: number, total: number, metric: string) =>
        `// k-fold · ${ok}/${total} folds ok · destaque ${metric}`,
      meanStd: "média ± desvio sobre os folds",
      fold: "fold",
      trainVal: "treino/val",
    },
    replicates: {
      title: (ok: number, total: number, metric: string) =>
        `// réplicas multi-seed · ${ok}/${total} seeds ok · destaque ${metric}`,
      citable: "🎯 resultado citável",
      // After the headline value: the interval it carries, and the sample size.
      headlineMeta: (hasCi: boolean, n: number) => `${hasCi ? "IC 95% · " : ""}n=${n}`,
      metric: "métrica",
      n: "n",
      mean: "média",
      std: "desvio",
      min: "min",
      max: "max",
      ci: "IC 95%",
    },
    // Model comparison on the other tasks.
    comparison: {
      title: (ok: number, total: number, failed: number, metric: string) =>
        `// comparação de arquiteturas · ${ok}/${total} ok${failed > 0 ? ` · ${failed} ${failed === 1 ? "falhou" : "falharam"}` : ""} · ranking por ${metric}`,
    },
    sweep: {
      title: (mode: string, ok: number, total: number, metric: string) =>
        `// sweep ${mode} · ${ok}/${total} trials ok · ranking por ${metric}`,
      // Followed by the metric's value.
      best: (metric: string) => `👑 melhor trial · ${metric}=`,
    },
    // Model comparison on classification.
    modelComparison: {
      title: (ok: number, total: number, failed: number) =>
        `// comparação de modelos · ${ok}/${total} ok${failed > 0 ? ` · ${failed} ${failed === 1 ? "falhou" : "falharam"}` : ""}`,
      accuracy: "Accuracy",
      aucRoc: "AUC-ROC",
      // The path goes in backticks, for <Rich>.
      footer: "Top-3 acima. O ranking completo está em `outputs/reports/<experiment>/ranking.csv`.",
    },
    gridSearch: {
      title: (ok: number, total: number, index: string) =>
        `// grid search · ${ok}/${total} trials ok · 👑 melhor trial #${index}`,
      overrides: "// overrides do trial vencedor",
      footer:
        "Tabela completa em `outputs/reports/<experiment>/grid_search_summary.csv` · config vencedora em `best_config.yaml`.",
    },
  },
  // The run-history sheet (components/HistoryOverlay.tsx). The status and the task
  // pill on each card are the backend's own values, shown as the server wrote them.
  history: {
    eyebrow: "// training history",
    title: "Treinamentos recentes",
    select: "✓ Selecionar",
    cancelSelect: "Cancelar seleção",
    compare: (n: number) => `↔ Comparar ${n}`,
    deleteSelected: (n: number) => `🗑 Excluir ${n}`,
    deleteSelectedTitle: (n: number) =>
      `Excluir ${n} ${n === 1 ? "run" : "runs"} permanentemente`,
    loading: "carregando histórico…",
    errorTitle: "Erro",
    loadFailed: "Erro ao carregar histórico.",
    emptyTitle: "Nenhum treinamento ainda",
    emptyHint: "Execute o primeiro experimento para vê-lo aqui.",
    selectModeTip: "Modo seleção — marque os runs que quer excluir ou comparar.",
    selectedTip: (n: number) =>
      `${n} ${n === 1 ? "selecionado" : "selecionados"} — 🗑 exclui; ↔ compara a partir de 2.`,
    searchPlaceholder: "🔍 buscar por nome, arquitetura ou run_id…",
    clearSearch: "Limpar busca",
    noMatch: "Nenhum run combina com o filtro atual.",
    // The task tabs; a custom task keeps its own key.
    allTab: "Todos",
    // The refinement rows inside a tab.
    allChip: "todos",
    filterType: "tipo",
    filterBlock: "bloco",
    filterStatus: "status",
    sortLabel: "ordenar",
    sort: {
      recent: "mais recente",
      oldest: "mais antigo",
      epochs: "mais épocas",
    },
    card: {
      deleteTitle: "Excluir este run permanentemente",
      preprocessing: (n: number) => `⚗ ${n} filtro${n === 1 ? "" : "s"}`,
      preprocessingTitle: (n: number) =>
        `${n} ${n === 1 ? "filtro de pré-processamento aplicado" : "filtros de pré-processamento aplicados"} ao treino`,
      resumeTitle: (done: number, total: number) =>
        `Parou na época ${done} de ${total} — dá para continuar`,
      resumeTitleNoTotal: "Parou antes do fim — dá para continuar",
      blockTitle: (block: string) => `Bloco de experimento: ${block}`,
      epochs: (n: number) => `${n} epoch${n !== 1 ? "s" : ""}`,
      // Where a finished run shows `→ end date`, a running one shows `· this`.
      inProgress: "em andamento",
    },
    // The confirmation modal.
    confirm: {
      title: (n: number) => `// excluir ${n === 1 ? "run" : `${n} runs`} permanentemente`,
      body: (n: number) =>
        `${n === 1 ? "A pasta do run" : "As pastas dos runs"}, checkpoints e todos os plots/relatórios serão removidos do disco. Esta ação é irreversível.`,
      // Runs go one at a time; the ones that failed are listed under this line.
      failed: (failed: number, total: number) =>
        `${failed} de ${total} ${failed === 1 ? "não pôde ser excluído" : "não puderam ser excluídos"}:`,
      unknownError: "erro desconhecido",
      deleting: "Excluindo…",
      submit: (n: number) => `🗑 Excluir${n > 1 ? ` ${n}` : ""}`,
    },
  },
  // What the segmentation, regression and anomaly forms (components/
  // SegmentationPanel.tsx, RegressionPanel.tsx, AnomalyPanel.tsx) have in common:
  // the training block, the transfer-learning block and the dataset basics. Model
  // names, metric names and config keys (ignore_index, backbone) read the same in
  // every language and are repeated in both dictionaries.
  taskPanel: {
    // The experiment-strategy selector; "Treino simples" and "K-Fold (CV)" are
    // `paramPanel.blocks`.
    sweep: "Sweep",
    replicates: "Réplicas",
    pretrained: "Pesos pré-treinados",
    backbone: "Backbone",
    imageNet: "ImageNet",
    // The hint under "pretrained weights" in the segmentation and regression forms.
    transferHint: "backbone pré-treinado (torchvision)",
    training: {
      title: "Treinamento",
      epochs: "Épocas",
      batchSize: "Batch size",
      batchSizeHint: "qualquer inteiro",
      learningRate: "Learning rate",
      loss: "Loss",
      seed: "Seed",
      optimizer: "Otimizador",
      earlyStop: "Early stop",
      earlyStopHint: "paciência",
      deterministic: "Determinístico",
      deterministicHint: "reprodutível",
    },
    transfer: {
      title: "Transfer learning",
      mode: "Modo",
      full: "Completo",
      featureExtraction: "Feature extr.",
      fineTuning: "Fine-tuning",
      backboneLr: "Backbone LR ×",
      backboneLrHint: "LR do backbone = LR × isto",
    },
    dataset: {
      baseDir: "Pasta base",
      baseDirHint: "raiz do dataset",
      browse: "📁 Escolher",
      imageSize: "Image size",
      trainSplit: "Split de treino",
      valSplit: "Split de validação",
      testSplit: "Split de teste",
      optional: "opcional",
      imagesSubdir: "Subpasta de imagens",
    },
  },
  // The segmentation form (components/SegmentationPanel.tsx).
  segmentationPanel: {
    namePlaceholder: "segmentation_001",
    // The metric the sweep and the replicates rank by.
    pixelAcc: "Pixel acc.",
    // `ignore_index` is the config key; `maxClass` is the highest real class id.
    ignoreIndexCollision: (index: number, maxClass: number) =>
      `ignore_index (${index}) colide com um id de classe real (0…${maxClass}). Use um valor fora desse intervalo (ex. 255 ou -1).`,
    model: {
      title: "Modelo · segmentação",
      architecture: "Arquitetura",
      architectureHint: "dense head",
      numClasses: "Nº de classes",
      numClassesHint: "inclui fundo",
      pretrainedHint: "backbone ImageNet",
    },
    lossHint: "por pixel (CE) ou sobreposição (Dice)",
    dataset: {
      title: "Dataset (imagens + máscaras)",
      baseDirPlaceholder: "…/dataset (train/{images,masks}, val/…)",
      imagesSubdirHint: "por split",
      masksSubdir: "Subpasta de máscaras",
      masksSubdirHint: "PNG · id por pixel",
      ignoreIndex: "ignore_index",
      ignoreIndexHint: "pixels void",
    },
  },
  // The regression form (components/RegressionPanel.tsx).
  regressionPanel: {
    namePlaceholder: "regression_001",
    model: {
      title: "Modelo · regressão",
      backboneHint: "CNN ou transformer",
      numTargets: "Nº de alvos",
      numTargetsHint: "derivado das colunas-alvo",
    },
    lossHint: "critério",
    dataset: {
      title: "Dataset (CSV manifest)",
      baseDirPlaceholder: "…/dataset (train.csv, val.csv, images/)",
      imagesDirHint: "relativa à base",
      imageColumn: "Coluna da imagem",
      imageColumnHint: "cabeçalho do CSV",
      targetColumns: "Colunas-alvo",
      targetColumnsPlaceholder: "target  ou  x,y,z",
      targetColumnsHint: "separadas por vírgula",
      trainCsv: "CSV de treino",
      valCsv: "CSV de validação",
      testCsv: "CSV de teste",
    },
  },
  // The anomaly-detection form (components/AnomalyPanel.tsx).
  anomalyPanel: {
    namePlaceholder: "anomaly_001",
    // The image-level F1 the sweep and the replicates can rank by.
    imageF1: "F1 (imagem)",
    model: {
      title: "Modelo · anomalia",
      method: "Método",
      methodHint: "abordagem",
      backboneHint: "extrator congelado",
      coresetRatio: "Coreset ratio",
      coresetRatioHint: "subamostra do banco",
      pretrainedBackbone: "Backbone pré-treinado",
      latentDim: "Latent dim",
      latentDimHint: "gargalo do autoencoder",
    },
    epochsHint: "ignoradas no PatchCore",
    threshold: "Threshold %ile",
    thresholdHint: "corte sobre scores normais",
    dataset: {
      title: "Dataset (MVTec · normal-only no treino)",
      baseDirPlaceholder: "…/categoria (train/good, test/good, test/<defeito>)",
      normalDir: "Subpasta normal",
      normalDirHint: "label 0 (ex. good)",
    },
  },
  // The one-shot dataset download (components/DatasetDownloadCard.tsx). Provider
  // names, the Kaggle token format and the example dataset ids stay as they are
  // in both languages; the sentences that carry them are whole strings.
  datasetDownload: {
    title: "Baixar dataset (online → pasta local)",
    cancel: "cancelar",
    open: "+ baixar dataset",
    provider: "Provedor",
    torchvision: {
      dataset: "Dataset",
      datasetHint: "built-in",
      limit: "Limite por classe",
      limitPlaceholder: "vazio = tudo",
      limitHint: "opcional",
    },
    roboflow: {
      dataset: "Dataset ou URL",
      datasetPlaceholder: "workspace/projeto — ou cole a URL",
      version: "Versão",
      apiKey: "API key",
      apiKeyHint: "app.roboflow.com → Settings → API Keys",
      format: "Formato",
      formatPlaceholder: "folder",
      formatHint: "folder p/ classificação",
    },
    kaggle: {
      dataset: "Dataset (owner/slug)",
      datasetPlaceholder: "zalando-research/fashionmnist",
      token: "API token",
      tokenPlaceholder: "KGAT_…",
      tokenHint: "kaggle.com → Settings → API → Create New Token",
    },
    huggingface: {
      dataset: "Dataset (id do HF Hub)",
      datasetPlaceholder: "owner/dataset",
      token: "Token",
      tokenHint: "opcional — só para datasets privados",
    },
    outDir: "Pasta de saída",
    outDirPlaceholder: "…/datasets/baixado",
    outDirHint: "onde gravar",
    browse: "📁 Escolher",
    download: "▶ Baixar",
    downloading: "Baixando…",
    needDatasetAndFolder: "Informe o dataset e a pasta de saída.",
    downloadingWait: "Baixando… (pode demorar)",
    done: (images: number, dir: string) => `${images} ${images === 1 ? "imagem" : "imagens"} em ${dir}`,
    failed: "Falha no download.",
    classes: (n: number) => ` · ${n} ${n === 1 ? "classe" : "classes"}`,
  },
  // A provider key you type once (components/CredentialField.tsx). The key
  // itself is never shown: a saved one appears masked.
  credentialField: {
    typeFirst: "Digite a chave antes de salvar.",
    saved: "Salva — não precisa digitar de novo.",
    saveFailed: "Falha ao salvar.",
    removed: "Chave removida deste computador.",
    removeFailed: "Falha ao remover.",
    savedPlaceholder: (masked: string) => `salva: ${masked}`,
    savedHint: "deixe em branco para usar a salva",
    replace: "↻ Substituir",
    save: "💾 Salvar",
    forget: "Esquecer",
  },
  // The dataset folder input with the split pickers (components/DatasetPicker.tsx).
  datasetPicker: {
    analyzing: "Analisando subpastas…",
    detectFailed: "Falha ao detectar splits do dataset.",
    opening: "Abrindo seletor nativo do sistema…",
    cancelled: "Seleção cancelada.",
    picked: (path: string) => `Pasta selecionada: ${path}`,
    pickFailed: "Falha ao abrir o seletor de pastas.",
    baseDir: "Pasta base do dataset",
    baseDirPlaceholder: "ex: C:/datasets/coffee  ou  /home/user/data",
    baseDirHint: "Pasta raiz que contém treino, validação e teste.",
    browseTitle: "Abrir seletor nativo do sistema (retorna o caminho absoluto)",
    browse: "📁 Escolher pasta",
    trainSubdir: "Subpasta treino",
    valSubdir: "Subpasta validação",
    testSubdir: "Subpasta teste",
    // A subfolder typed by hand that the auto-detection did not list.
    manual: (value: string) => `${value} (manual)`,
  },
  // The pre-training overview of a classification dataset (components/DatasetStats.tsx).
  // Its split names, "missing", the imbalance flag and the sample strip title are
  // also used by the detection and task overviews below.
  datasetStats: {
    splits: {
      train: "treino",
      val: "validação",
      test: "teste",
    },
    missing: "ausente",
    imbalanced: "⚠ desbalanceado",
    samples: (split: string) => `// amostras (split: ${split}) — sanity-check de labels`,
    analyzing: "Analisando dataset…",
    noClasses: "Nenhuma classe encontrada.",
    classMap: "mapeamento (ImageFolder):",
    appliedBinaryTitle: "Detectado e aplicado ao config: task=binary, num_classes=1",
    appliedMulticlassTitle: (classes: number) =>
      `Detectado e aplicado ao config: task=multiclass, num_classes=${classes}`,
    appliedBinary: "binary aplicado",
    appliedMulticlass: (classes: number) => `multiclass · ${classes} aplicado`,
    distribution: "// distribuição do dataset",
    sampleAlt: (className: string) => `${className} sample`,
    noImages: "sem imagens",
  },
  // The YOLO dataset overview (components/DetectionDatasetStats.tsx).
  detectionDatasetStats: {
    noSplits: "Nenhum split YOLO encontrado (images/<split>).",
    classMap: "mapeamento (YOLO):",
    appliedTitle: (classes: number) => `Detectado e aplicado ao config: num_classes=${classes}`,
    applied: (classes: number) => `${classes} ${classes === 1 ? "classe aplicada" : "classes aplicadas"}`,
    exampleAlt: (className: string, n: number) => `${className} — exemplo ${n}`,
    distribution: "// distribuição de anotações (instâncias)",
    layoutTitle: "Layout YOLO detectado neste split",
    images: (n: number) => `${n} img`,
    boxes: (n: number) => `${n} ${n === 1 ? "caixa" : "caixas"}`,
    unlabeled: (n: number) => `${n} sem label`,
  },
  // The segmentation, anomaly and regression dataset overviews
  // (components/TaskDatasetStats.tsx). Counts and column names come from the
  // server; `ignore_index` and NEAREST are identifiers.
  taskDatasetStats: {
    // Shown only above 32 distinct ids, so the count is never 1: plural only.
    interpolated: (ids: number) =>
      `⚠ ${ids} ids distintos na amostra — as máscaras parecem interpoladas (anti-aliasing). Use máscaras com um id de classe por pixel (resample NEAREST).`,
    maskIds: "ids nas máscaras (amostra):",
    voidTitle: "provável ignore_index (void)",
    applyClasses: (n: number) => `🎯 aplicar ${n} ${n === 1 ? "classe" : "classes"}`,
    pairing: "// pareamento imagem ↔ máscara",
    pairs: (n: number) => `${n} ${n === 1 ? "par" : "pares"}`,
    pairCounts: (images: number, masks: number) => `${images} img · ${masks} másc`,
    unpaired: (images: number, masks: number) =>
      `⚠ ${images} img sem máscara · ${masks} másc sem img`,
    anomalyKicker: "// distribuição normal vs. anômalo",
    trainNormal: "treino (normal)",
    testNormal: "teste · normal",
    testAnomalous: "teste · anômalo",
    images: (n: number) => `${n} img`,
    missingTestDir: "pasta de teste ausente",
    regressionKicker: "// manifest & distribuição dos alvos",
    rows: (n: number) => `${n} ${n === 1 ? "linha" : "linhas"}`,
    missingColumns: (columns: string) => `⚠ colunas ausentes: ${columns}`,
    missingImages: (missing: number, checked: number) =>
      `⚠ ${missing}/${checked} ${missing === 1 ? "imagem não encontrada" : "imagens não encontradas"}`,
    // `μ` is the mean, `[min, max]` the range, `n` the number of values.
    target: (column: string, mean: string, min: string, max: string, n: number) =>
      `${column}: μ ${mean} · [${min}, ${max}] · n=${n}`,
  },
  // The preprocessing pipeline builder (components/PreprocessingPanel.tsx). The
  // filter names (Gaussian blur, Unsharp mask…) are the same in every language.
  preprocessing: {
    needBaseDir: "Defina uma pasta base antes de gerar preview.",
    kicker: "// pré-processamento (filtros)",
    active: (n: number) => `${n} filtro${n === 1 ? "" : "s"} ativo${n === 1 ? "" : "s"} no treino`,
    activeTitle:
      "Estes filtros serão aplicados durante o treino, antes de augmentation e normalização",
    clearTitle: "Remover todos os filtros do pipeline",
    clear: "limpar",
    addFilter: "+ adicionar filtro",
    generating: "Gerando…",
    preview: "▶ Ver preview",
    empty: 'Pipeline vazio — clique em "+ adicionar filtro".',
    // Marks: **antes** is emphasised.
    emptyNote:
      "O pipeline configurado aqui roda **antes** de augmentation e normalização, em todos os splits (treino / val / teste).",
    original: "Original",
    final: "Final",
    moveUp: "Mover para cima",
    moveDown: "Mover para baixo",
    remove: "Remover",
  },
  // The augmentation preview strip (components/AugmentPreview.tsx).
  augmentPreview: {
    needBaseDir: "Defina a pasta base do dataset primeiro.",
    generating: "Gerando…",
    button: "🎲 preview de augmentation",
    active: "ativos:",
    noneActive: "nenhum aumento ativo — apenas resize",
    original: "original",
    variant: (n: number) => `variante ${n}`,
  },
  // Normalization and augmentation of the standalone task panels
  // (components/TransformsSection.tsx). The field names are `paramPanel.fieldLabels`.
  transforms: {
    image: "Imagem",
    normalizeHint: "R, G, B — aplicada a treino, validação e teste",
    augmentation: "Data augmentation",
    augmentToggle: "Augmentation",
    augmentHint: "só no treino; desligada, os valores abaixo ficam guardados",
    flipHint: "treino",
    rotationHint: "0 = desliga",
  },
  // The on/off button of every switch (components/controls/Toggle.tsx). The
  // button sets them in capitals.
  toggle: {
    on: "ATIVADO",
    off: "DESATIVADO",
  },
  // The five built-in tasks (types/tasks.ts): the tab name and the sentence under
  // the hero title. A researcher-defined task brings its own from the server.
  tasks: {
    classification: {
      label: "Classificação",
      description: "Categorize imagens em rótulos discretos",
    },
    detection: {
      label: "Detecção de Objeto",
      description: "Localize e classifique objetos com bounding boxes",
    },
    regression: {
      label: "Regressão",
      description: "Estime valores contínuos a partir das entradas",
    },
    segmentation: {
      label: "Segmentação",
      description: "Máscaras a nível de pixel para cada categoria",
    },
    anomaly: {
      label: "Anomalia",
      description: "Detecte defeitos treinando só com imagens normais",
    },
  },
  // The sub-line of each entry in the model dropdowns (lib/regression-models.ts,
  // lib/segmentation-models.ts, lib/anomaly-models.ts), keyed by model value. The
  // model names themselves are identifiers and stay in the code.
  regressionModels: {
    resnet18: "light · 11.7M",
    resnet34: "21.8M",
    resnet50: "imagenet · 25.6M",
    resnet101: "deep · 44.5M",
    efficientnet_b1: "eficiente · 7.8M",
    efficientnet_b7: "grande · 66M",
    vgg16: "clássico · 138M",
    vgg19: "144M",
    alexnet: "baseline · 61M",
    vit_b_16: "transformer · 86M",
    swin_t: "transformer · 28M",
    convnext_tiny: "moderno · 28M",
  },
  segmentationModels: {
    deeplabv3_resnet50: "imagenet · 42M",
    deeplabv3_resnet101: "deep · 61M",
    deeplabv3_mobilenet_v3_large: "leve · 11M",
    fcn_resnet50: "32.9M",
    fcn_resnet101: "51.9M",
    lraspp_mobilenet_v3_large: "mobile · 3.2M",
    unet: "clássico · 31M",
  },
  anomalyModels: {
    autoencoder: "reconstrução · treinável",
    patchcore: "memory bank · backbone",
    // The feature extractor PatchCore reads; ResNet-34 and ResNet-50 carry no note.
    backbones: {
      resnet18: "light",
      wide_resnet50_2: "patchcore padrão",
    },
  },
  // How a queued job is named (lib/queue-format.ts). The server sends the task and
  // the strategy as identifiers; a custom task shows as its own key.
  queueFormat: {
    strategies: {
      simple: "treino simples",
      kfold: "K-fold",
      transferLearning: "transfer learning",
      gridSearch: "grid search",
      randomSearch: "random search",
      sweep: "sweep",
      replicates: "réplicas",
      comparison: "comparação",
      replicatedComparison: "comparação replicada",
    },
  },
  // Why a candidate value of a grid-search axis is refused (lib/grid-axis.ts).
  gridAxis: {
    notAnOption: "fora das opções",
    invalidValue: "valor inválido",
    mustBeInteger: "precisa ser inteiro",
    mustBeAbove: (min: number) => `precisa ser > ${min}`,
    mustBeAtLeast: (min: number) => `precisa ser ≥ ${min}`,
    mustBePowerOfTwo: "precisa ser potência de 2",
  },
  // The tab title and the notification that announce a finished run
  // (lib/run-notify.ts). One line each: OS toasts truncate the rest.
  runNotify: {
    completedTitle: (label: string) => `Treino concluído — ${label}`,
    completedBody: "Abra o VisionForge para ver os resultados.",
    failedTitle: (label: string) => `Treino falhou — ${label}`,
    failedBody: "Abra o VisionForge para ver o erro.",
  },
  // The YAML import (lib/yaml-config.ts). `reason` of cannotRead/invalidFile
  // is either the browser's or the parser's own message (it stays as they
  // wrote it) or `notMapping`.
  yamlConfig: {
    cannotRead: (reason: string) => `Não foi possível ler o arquivo YAML: ${reason}`,
    invalidFile: (reason: string) => `Arquivo YAML inválido: ${reason}`,
    notMapping:
      "o conteúdo deve ser um mapeamento (chave: valor), não um valor solto nem uma lista.",
    expectedObject: "Esperado um objeto.",
    expectedArray: "Esperada uma lista.",
    requiredMissing: "Campo obrigatório ausente.",
    mustBeOneOf: (options: string) => `Deve ser um destes: ${options}.`,
    expectedBoolean: "Esperado um booleano.",
    expectedInteger: "Esperado um inteiro.",
    expectedNumber: "Esperado um número.",
    expectedString: "Esperado um texto.",
  },
  // Whether two runs saw the same data (lib/dataset-identity.ts); only the
  // "cannot tell" answers carry a reason.
  datasetIdentity: {
    noFingerprint: "um dos runs não tem fingerprint (anterior a 26/07/2026)",
    differentMethod: "os dois runs usaram método diferente",
  },
  // The tab description of a researcher-defined task that declared none
  // (lib/custom-tasks.ts).
  customTasks: {
    fallbackDescription: "Tarefa definida pelo pesquisador",
  },
  // Why an explicit seed list is refused (lib/replicates-form.ts).
  replicatesForm: {
    needTwoSeeds: "informe pelo menos 2 seeds",
    duplicateSeeds: "seeds duplicadas",
  },
  // The run hook's messages and the breadcrumb that names a field in a validation
  // error, "Treinamento › Learning Rate" (hooks/useExperiment.ts). Filter names in
  // the breadcrumb are technical and stay in the code.
  experiment: {
    resultFetchFailed: "Falha ao buscar resultados do experimento.",
    failedNoDetail: "O experimento falhou sem mensagem detalhada.",
    connectionLost: "Conexão com o servidor perdida durante o polling.",
    validationFailed: (n: number) =>
      n === 1
        ? "1 campo com erro de validação. Confira o destaque no formulário."
        : `${n} campos com erro de validação. Confira os destaques no formulário.`,
    alreadyRunning: "Já existe um experimento em execução. Aguarde terminar.",
    unexpected: (message: string) => `Erro inesperado: ${message}`,
    unknown: "Erro desconhecido ao iniciar o experimento.",
    // Only the path segments the form has no label for: the form's own names
    // (`paramPanel.sectionLabels`/`fieldLabels`) word the rest of a path, so an
    // error calls a field what the field is called.
    sections: {
      preprocessing: "Pré-processamento",
      steps: "Filtro",
      scheduler: "Scheduler",
      device: "Dispositivo",
    },
  },
  // The card every task panel opens with (components/ExperimentHeader.tsx): the
  // experiment name, the YAML round trip and the strategy selector. The strategy
  // names themselves are `paramPanel.blocks.simple`, `taskPanel.sweep` and
  // `taskPanel.replicates`.
  experimentHeader: {
    title: "Experimento",
    nameLabel: "Nome do experimento",
    nameHint: "usado na pasta de saída e no histórico",
    exportTitle: "Exportar configuração atual como arquivo .yaml",
    exportYaml: "↓ Exportar YAML",
    importTitle: "Importar configuração a partir de um arquivo .yaml",
    importYaml: "↑ Importar YAML",
    imported: (file: string) => `✓ ${file} importado`,
    strategyTitle: "Estratégia de experimento",
    mode: "Modo",
    // What the selected strategy does, under the selector.
    hints: {
      simple: "um treino com a config abaixo (botão Treinar)",
      cv: "K folds sobre o treino → métricas fold a fold + média ± desvio",
      sweep: "grid / random / optuna sobre a config abaixo",
      replicates: "mesma config, N seeds → média ± IC 95%",
    },
  },
  // The K-fold launcher (components/CvCard.tsx).
  cvCard: {
    title: "Cross-validation (K-fold)",
    description:
      "Divide as linhas de treino em K folds: cada fold treina um modelo novo em K-1 partes e avalia na parte restante (nunca augmentada). O split de teste não é usado.",
    folds: "Nº de folds",
    shuffle: "Shuffle",
    shuffleHint: "antes do split",
    foldSeed: "Seed do split",
    // K is at least 2 (the field's minimum), so "folds" is plural only.
    run: (folds: number) => `⛓ Rodar CV · ${folds} folds`,
  },
  // The multi-seed launcher (components/ReplicatesCard.tsx).
  replicatesCard: {
    title: "Réplicas multi-seed · rigor estatístico",
    description:
      "Treina a mesma config N vezes sob seeds diferentes e agrega cada métrica em média ± IC 95% (t de Student). Um run único é uma amostra de uma distribuição — réplicas tornam o número defensável.",
    seeds: "Seeds",
    automatic: "Automáticas",
    explicit: "Explícitas",
    automaticHint: "consecutivas a partir do training.seed",
    explicitHint: "lista exata, reproduzível",
    count: "Nº de réplicas",
    seedList: "Seeds (vírgula)",
    seedCount: (n: number) => `${n} seeds`,
    headlineMetric: "Métrica destaque",
    run: (seeds: number) => `🎲 Rodar réplicas · ${seeds} seed${seeds === 1 ? "" : "s"}`,
  },
  // The hyperparameter sweep editor (components/SweepCard.tsx). Grid, Random and
  // Optuna are algorithm names; dot-paths and the model names in the placeholders
  // read the same in every language.
  sweepCard: {
    title: "Sweep de hiperparâmetros · modo avançado",
    // `paths` is the task's suggested dot-paths, joined with commas.
    description: (paths: string) =>
      `Varre hiperparâmetros por dot-path (ex.: ${paths}) e ranqueia pela métrica. Grid = produto cartesiano; Random = amostras; Optuna = busca TPE adaptativa.`,
    strategy: "Estratégia",
    presetTitle: "preset · arquiteturas → eixo model.name",
    compareArchitectures: (n: number) => `⇒ comparar ${n} arquitetura${n === 1 ? "" : "s"}`,
    parameter: "Parâmetro (dot-path)",
    values: "Valores (vírgula)",
    distribution: "Distribuição",
    kinds: {
      uniform: "Uniforme",
      logUniform: "Log-uniforme",
      choice: "Escolha",
    },
    options: "Opções (vírgula)",
    low: "low",
    high: "high",
    removeParameter: "Remover parâmetro",
    addParameter: "+ adicionar parâmetro",
    rankingMetric: "Métrica de ranking",
    trials: "Nº de trials",
    seed: "Seed",
    run: (trials: number) => `⛓ Rodar sweep · ${trials} trial${trials === 1 ? "" : "s"}`,
  },
  // The generic form of a researcher-defined task (components/SchemaForm.tsx). The
  // fields' own titles and descriptions come from the task's schema and are shown
  // as written; only the card headings and the list hint are ours.
  schemaForm: {
    // Keys are the names of the blocks every custom task inherits.
    sections: {
      training: "Treinamento",
      data: "Dataset",
      transforms: "Aumentos & normalização",
      preprocessing: "Pré-processamento (filtros)",
      scheduler: "Learning-rate scheduler",
      output: "Saída",
      model: "Modelo",
    },
    taskParameters: "Parâmetros da tarefa",
    listHint: "valores separados por vírgula",
  },
  // The panel a researcher-defined task gets (components/CustomTaskPanel.tsx).
  customTaskPanel: {
    // Followed by the task key in bold, a colon and the server's reason.
    schemaLoadFailed: "Não foi possível carregar o schema de",
    // The path goes in backticks, for <Rich>.
    schemaLoadHint:
      "Verifique o arquivo em `user_tasks/` — um erro de import é registrado no log do servidor e a tarefa fica sem formulário.",
    loadingForm: (task: string) => `Carregando o formulário de ${task}…`,
  },
  // Hide or delete a researcher-defined task (components/CustomTaskManageCard.tsx).
  // Deleting removes the file the researcher wrote, so it asks for the key to be typed.
  customTaskManage: {
    heading: "// gerenciar esta task",
    close: "fechar",
    options: "⚙ opções",
    hideTitle: "Some com a aba; o arquivo continua em user_tasks/",
    hide: "👁 Ocultar aba",
    hideNote: "reversível — o arquivo fica",
    hideFailed: "Falha ao ocultar.",
    delete: "🗑 Excluir do disco",
    deleteNote: "apaga o .py que você escreveu — sem desfazer",
    // The task's label goes in bold between these two, then its key in bold after them.
    confirmBefore: "Isto remove o arquivo de",
    confirmAfter: "do disco. Para confirmar, digite a chave da task:",
    typeKeyTitle: "Digite a chave exata para habilitar",
    confirmLabel: "Chave da tarefa para confirmar",
    deleting: "Excluindo…",
    deleteForever: "Excluir definitivamente",
    deleteFailed: "Falha ao excluir.",
  },
  // The run queue sheet (components/QueueOverlay.tsx). Task and strategy names are
  // `queueFormat`.
  queueOverlay: {
    readFailed: "Não foi possível ler a fila.",
    cancelFailed: "Não foi possível cancelar esse treino.",
    kicker: "// fila",
    title: "Treinos na fila",
    description:
      "Uma GPU, um treino por vez. Submeta quantos quiser — eles rodam em ordem de envio, sozinhos.",
    emptyTitle: "A GPU está livre e nada está esperando.",
    emptyHint: "Envie um treino e, se enviar outro em seguida, ele aparece aqui.",
    position: (n: number) => `${n}º`,
    running: "em execução",
    waiting: (waited: string) => `esperando ${waited}`,
    removeTitle: "Remover da fila (não afeta treinos já iniciados)",
    stopTitle: "Interromper este treino (o trabalho já feito é mantido)",
    // The running row of a kind of run the server cannot stop mid-way (lib/run-control.ts).
    stopUnavailable:
      "Esta execução não pode ser interrompida no meio: ela segue até o fim.",
    stop: "■ parar",
    remove: "🗑 remover",
  },
  // The live training sheet (components/TrainingOverlay.tsx). A failed run's error
  // text comes from the server and is shown as it wrote it; so is a phase label
  // other than the three below; the epoch and trial lines of the log are the
  // trainer's own vocabulary and read the same in every language. Grid and random
  // search are `paramPanel.blocks`.
  trainingOverlay: {
    blocks: {
      modelComparison: "Comparação de modelos",
      crossValidation: "K-Fold CV",
      replicates: "Réplicas multi-seed",
    },
    // The three phases PatchCore reports, which the server labels in Portuguese
    // (lib/training-phase.ts maps them).
    phases: {
      extractingFeatures: "extraindo features",
      buildingBank: "montando o banco",
      scoring: "pontuando",
    },
    // The first two lines of the log.
    initializing: (runId: string) => `> inicializando runtime · ${runId}`,
    loadingDataset: "> carregando dataset…",
    trainingFailed: "training failed",
    trainingComplete: "training complete",
    queued: (task: string) => `na fila · ${task}`,
    training: (task: string) => `training · ${task}`,
    starting: "iniciando…",
    // `position` is the run's place in line and `queued` how many are waiting; both
    // may be missing.
    queuedNote: (position: number | undefined, queued: number | undefined) =>
      `${position ? `aguardando a GPU — ${position}º na fila` : "aguardando a GPU"}${
        typeof queued === "number" && queued > 1 ? ` · ${queued} submissões esperando` : ""
      }. O treino começa sozinho quando chegar a vez.`,
    errorDetail: "Detalhe do erro",
    unknownError: "Erro desconhecido — verifique os logs do servidor.",
    // The same, as it reads in the log line.
    logUnknownError: "erro desconhecido",
    queueBanner: (block: string, runs: number | undefined) =>
      `⛓ bloco de treinos · ${block}${runs && runs > 1 ? ` · ${runs} runs` : ""}`,
    queueBannerBody:
      'Este bloco executa múltiplos treinos sequencialmente. A barra de progresso reflete o trial corrente; o resultado agregado aparece em "Ver resultados" ao final.',
    pipeline: (n: number) => `⚗ pipeline ativo · ${n} filtro${n === 1 ? "" : "s"}`,
    minimize: "Minimizar",
    viewResults: "↗ Ver resultados",
    // The stop control of a running job. The server stops a run at the top of its next
    // epoch, so it finishes the one in progress and keeps what it has saved; a search
    // also skips the trials not yet started (lib/run-control.ts).
    stop: "■ Parar",
    stopTitle: "Parar o treino ao fim da época em andamento",
    stopUnavailable:
      "Esta execução não pode ser interrompida no meio: ela segue até o fim.",
    stopConfirmEpoch:
      "Parar este treino? Ele termina a época em andamento e mantém o melhor checkpoint e o histórico até aqui. Se parar antes da última época, dá para retomar pelo Histórico.",
    stopConfirmTrial:
      "Parar esta busca? O treino em andamento termina a época atual e os que ainda não começaram são pulados. Os trials já concluídos ficam salvos; uma busca interrompida não pode ser retomada.",
    stopConfirmYes: "Parar o treino",
    stopConfirmNo: "Continuar treinando",
    stopSending: "Parando…",
    stopRequested:
      "Parada pedida: o treino termina a época em andamento e então para.",
    stopFailed: "Não foi possível parar o treino.",
    // The server found no such job: it ended between the click and the request.
    stopAlreadyEnded: "O treino já tinha terminado quando o pedido chegou.",
    // An anomaly run that has not yet shown whether it trains by epochs (PatchCore does not).
    stopUnconfirmed:
      "O botão Parar libera quando o treino mostrar que roda por épocas.",
    // Header, once a stopped run has ended; the status the server reports for it is still "completed".
    stopped: "Treino interrompido",
    // Before its first epoch there is no checkpoint to keep, so none is claimed.
    stoppedLog: (epoch: number | null, total: number | null) =>
      epoch === 0
        ? "interrompido antes da primeira época"
        : `interrompido${epoch !== null && total !== null ? ` na época ${epoch}/${total}` : ""} · melhor checkpoint mantido`,
    stopTooLate:
      "> a parada chegou na última época: o treino terminou normalmente",
  },
  // Side-by-side comparison of two or more runs (components/CompareRunsPanel.tsx).
  compareRuns: {
    back: "← histórico",
    // A comparison needs at least two runs, so "runs" is plural only.
    comparing: (n: number) => `Comparando ${n} runs`,
    loading: "carregando runs…",
    loadFailed: "Falha ao carregar runs.",
    // Whether the runs saw the same data; the reason of an unverifiable pair is
    // `datasetIdentity`.
    verdict: {
      same: "mesmos dados",
      different: "dados diferentes",
      unknown: "não verificável",
    },
    metric: "Métrica",
    device: "Dispositivo",
    field: "Campo",
    // Keys are the backend's metric names.
    metrics: {
      best_val_loss: "Melhor val loss",
      best_epoch: "Melhor epoch",
      total_epochs: "Total de epochs",
      test_accuracy: "Acurácia (teste)",
      test_f1: "F1 (teste)",
      test_precision: "Precisão (teste)",
      test_recall: "Recall (teste)",
      test_auc_roc: "AUC-ROC (teste)",
    },
    configDiffTitle: "// diff de configuração (células destacadas = diferentes da 1ª run)",
    // The rows of the configuration table.
    config: {
      architecture: "Arquitetura",
      numClasses: "Num classes",
      pretrained: "Pretrained",
      task: "Task",
      learningRate: "Learning rate",
      optimizer: "Optimizer",
      batchSize: "Batch size",
      epochsMax: "Epochs (max)",
      weightDecay: "Weight decay",
      seed: "Seed",
      mixedPrecision: "Mixed precision",
      scheduler: "Scheduler",
      imageSize: "Image size",
      horizontalFlip: "Horizontal flip",
      rotation: "Rotation (°)",
      colorJitter: "Color jitter",
      preprocessing: "Preprocessing",
    },
    preprocessingTitle: "// pipelines de pré-processamento",
    noPreprocessing: "sem pré-processamento",
    valLossChart: "Val loss × epoch",
    valAccuracyChart: "Val accuracy × epoch",
  },
  // The first-run guide (components/GuidedTour.tsx). The steps are `tour`.
  guidedTour: {
    dialogLabel: "Guia do VisionForge",
    closeLabel: "Fechar o guia",
    eyebrow: "Primeira vez por aqui",
    inviteTitle: "Quer uma volta rápida?",
    inviteBody:
      "Sete paradas curtas pelos pontos principais: onde escolher a tarefa, como apontar o dataset, o que já vem decidido para você e onde os resultados ficam guardados. Dá para sair a qualquer momento — e o guia continua disponível no cabeçalho depois.",
    notNow: "Agora não",
    seeGuide: "Ver o guia →",
    next: "Continuar →",
    finish: "Concluir",
  },
  // The seven stops of the guide (lib/tour.ts `tourSteps`).
  tour: {
    tabs: {
      title: "Escolha o tipo de treino",
      body: "Cada aba é uma tarefa completa: classificação, detecção, regressão, segmentação e anomalia. Trocar de aba troca o formulário inteiro, as métricas e a cor da interface — nada é compartilhado por acidente entre elas.",
    },
    dataset: {
      title: "Aponte a pasta do dataset",
      body: "Escolha a pasta raiz e o VisionForge procura sozinho as subpastas de treino, validação e teste pelos nomes usuais. Se o seu dataset usa outros nomes, os seletores ao lado deixam você corrigir sem renomear nada no disco.",
    },
    parameters: {
      title: "Os parâmetros que importam ficam na frente",
      body: "Cada painel mostra primeiro o essencial — épocas, batch, taxa de aprendizado — e guarda o resto em “Avançado”, recolhido. Os valores que já vêm preenchidos foram medidos por tarefa, então começar sem mexer em nada é uma escolha válida. O “i” ao lado de cada rótulo explica o que aquele campo faz.",
    },
    device: {
      title: "GPU ou CPU",
      body: "O VisionForge detecta o que existe na máquina e escolhe a GPU quando ela está disponível. Dá para forçar a CPU aqui: é mais lento, mas roda em qualquer lugar e serve para conferir se um erro é do código ou da placa.",
    },
    train: {
      title: "Treinar",
      body: "O botão roda exatamente o que está selecionado — um treino simples, uma busca em grade, validação cruzada ou réplicas. Enquanto roda, uma tela mostra o progresso e as métricas de cada época; você pode minimizá-la e voltar a ela depois. Nos treinos simples e nas buscas ela também tem o botão Parar: o treino termina a época em andamento e guarda o que já foi feito.",
    },
    history: {
      title: "Tudo fica salvo",
      body: "Cada execução guarda em disco a configuração, as métricas de todas as épocas, os gráficos e os pesos. O histórico deixa você reabrir, comparar duas execuções lado a lado, continuar um treino interrompido e testar o modelo em imagens novas.",
    },
    datasets: {
      title: "Seus datasets",
      body: "Aqui você baixa um dataset (torchvision, Roboflow, Kaggle ou Hugging Face) para uma pasta local e aponta qualquer tarefa para ela. Com a pasta definida, o painel da tarefa mostra a distribuição das classes — vale conferir antes do primeiro treino: quase todo resultado estranho começa em um dataset desbalanceado.",
    },
  },
};

export type Dict = typeof pt;

/** The form a ModelAdvice note appears on. The measured numbers are classification's. */
export type ModelAdviceTask = "classification" | "regression" | "segmentation";

/** What the collapse note is worded from: the response's evidence and where it is shown. */
export interface CollapseFacts {
  architecture: string;
  /** The one model of the family that was actually run. */
  measuredOn: string;
  accuracy: number;
  /** What `measuredOn` reached at the suggested setting; null where that was not run. */
  recoveredAccuracy: number | null;
  optimizer: string;
  learningRate: number;
  task: ModelAdviceTask;
}
