// Help texts shared by two field names (the same knob under two spellings, one
// per task family). One constant each, so the copies cannot drift apart.
const workersHelp =
  "Processos que carregam as imagens em paralelo. No automático o VisionForge divide a memória livre da máquina pelo custo de um worker — no Windows cada um recarrega o torch e as DLLs da CUDA, ~1 GB, e um número alto demais não deixa o treino lento: impede o treino de começar (WinError 1455).";
const lrFinalHelp = "Fração do learning rate inicial ao término do treino.";

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
    cudaTitle: (version: string, gpus: number) => `CUDA ${version} · ${gpus} GPU(s)`,
    cudaSummary: (version: string, gpus: number) => `cuda ${version} · ${gpus} gpu(s)`,
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
    cos_lr: "Faz o learning rate cair numa curva de cosseno em vez de cair em linha reta.",
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
      weights_path: "Caminho dos pesos",
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
      base_dir: "Diretório base",
      train_dir: "Subdir treino",
      val_dir: "Subdir validação",
      test_dir: "Subdir teste",
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
      axisTag: (values: number) => `grade · ${values} valores`,
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
    exportYaml: "↓ Exportar YAML",
    exportTitle: "Exportar configuração atual como arquivo .yaml",
    importYaml: "↑ Importar YAML",
    importTitle: "Importar configuração a partir de um arquivo .yaml",
    importWarnings: (count: number, summary: string, extra: number) =>
      `YAML importado com ${count} aviso(s) estrutural(is):\n${summary}${extra > 0 ? `\n…(+${extra} mais)` : ""}\n\nCorrija antes de treinar — o backend rejeita por validação Pydantic.`,
    unavailable: (task: string) => `${task} ainda não está disponível`,
    unavailableBody: "Esta tarefa será implementada em uma próxima fase do VisionForge.",
    loadingSchema: "carregando schema…",
    hiddenParams: (n: number) => `${n} parâmetros ocultos — ligue para ajustar`,
    fieldErrors: (n: number) => `${n} campo(s) com erro:`,
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
    cancel: "cancelar",
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
    modelCard: {
      button: "↓ markdown",
      title: "Baixar model card (markdown) deste run",
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
      files: (n: number | null | undefined, size: string) => `${n ?? "—"} arquivos · ${size}`,
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
      hint: "Roda o checkpoint sobre uma pasta de imagens e escreve um CSV com uma linha por imagem (probabilidades + classe predita). Útil para classificar batches de dados novos sem retreinar.",
      needFolder: "Informe a pasta de imagens para inferência.",
      starting: "Rodando inferência em lote…",
      done: (ok: number, csv: string) => `${ok} imagens processadas · CSV em ${csv}`,
      doneWithFailures: (ok: number, failed: number, csv: string) =>
        `${ok} ok · ${failed} falharam · CSV em ${csv}`,
      failed: "Falha na inferência em lote.",
      processed: "processadas",
      failedCount: "falharam",
      csv: "csv",
      failedFiles: (n: number) => `${n} arquivos falharam (clique para ver até 5)`,
      more: (n: number) => `…+${n} mais`,
    },
    gradcam: {
      title: "Grad-CAM (explicabilidade)",
      open: "🔥 Grad-CAM",
      folderPlaceholder: "ex: C:/datasets/amostras",
      samples: "Nº de amostras (1–64)",
      run: "▶ Gerar Grad-CAM",
      running: "Gerando…",
      hint: "Gera mapas de calor Grad-CAM sobre imagens de exemplo, destacando as regiões que mais influenciaram a classe predita pelo checkpoint. Útil para interpretar o que o modelo aprendeu.",
      needFolder: "Informe a pasta de imagens.",
      starting: "Gerando mapas Grad-CAM…",
      done: (count: number, layer: string) => `${count} mapa(s) gerado(s) · camada ${layer}`,
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
      ciTooltip: (percent: number, resamples: number, samples: number) =>
        `IC ${percent}% por bootstrap percentil: ` +
        `${resamples} reamostragens das ${samples} imagens de ` +
        `teste. Mede o ruído de amostragem do split com este modelo ` +
        `fixo — não a variação entre treinos.`,
    },
    // The plot files the backend writes, by file name.
    graphs: {
      title: "Gráficos (clique para expandir)",
      labels: {
        "loss.png": "Loss (train + val)",
        "accuracy.png": "Accuracy (train + val)",
        "confusion_matrix.png": "Matriz de confusão",
        "confusion_matrix_normalized.png": "Matriz de confusão (normalizada)",
        "roc_curve.png": "Curva ROC",
        "precision_recall_curve.png": "Curva Precision-Recall",
        // Detection (Ultralytics / torchvision).
        "results.png": "Resultados (loss + mAP)",
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
          "O .csv com a coluna de imagem e a(s) coluna(s) alvo. As imagens seguem a mesma pasta do treino.",
        classification: "Uma pasta com uma subpasta por classe — a mesma convenção do treino.",
      },
      label: "Rótulo (opcional)",
      labelPlaceholder: "ex: holdout_2026",
      run: "▶ Rodar teste",
      running: "Avaliando…",
      needFolder: "Informe o diretório base do dataset de teste.",
      starting: "Avaliando modelo no novo dataset…",
      recorded: (id: string) => `Teste registrado: ${id}`,
      failed: "Falha no teste.",
    },
    cv: {
      title: (ok: number, total: number, failed: number) =>
        `Cross-validation · ${ok}/${total} folds ok${failed > 0 ? ` · ${failed} falharam` : ""}`,
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
};

export type Dict = typeof pt;
