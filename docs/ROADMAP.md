# Roadmap

VisionForge is below `1.0.0`: the configuration schema and the HTTP API may still change between minor releases, while `schema_version` migrations keep exported YAML files working. This document summarizes what is already on `development` for the next release, what is planned next, and what defines `1.0.0`, without target dates.

- **What already shipped:** [`CHANGELOG.md`](../CHANGELOG.md)
- **Architectural and statistical decisions (ADRs):** [`docs/dev/DECISIONS.md`](dev/DECISIONS.md)
- **Active backlog and open issues:** [`docs/dev/BACKLOG.md`](dev/BACKLOG.md) and [GitHub Issues](https://github.com/marcus-vreis/VisionForge/issues)
- **Superseded pre-0.2 plan (historical):** [`docs/archive/ROADMAP.md`](archive/ROADMAP.md)

## On `development` (next release)

- **Classification replicates and CLI strategies:** multi-seed replicates in the Classification tab, and `strategy:` + `config:` YAML envelopes (`replicates`, `replicated_comparison`, `sweep`, `cv`) in `visionforge run`.
- **History rerun:** load any run's configuration back into the form or repeat it across 3 new seeds directly from History.
- **Dataset explorer and safe derived datasets:** the **Explore** tab in **Datasets** (class/split counts, thumbnail gallery with bounding boxes, background SHA-256 scan) and sibling-folder operations to incorporate images, merge datasets, resplit with the test set locked by default, and convert between YOLO and COCO.
- **Code tab across all five tasks:** live inspection of the actual package functions (`inspect`) for classification, regression, segmentation, anomaly detection and `torchvision` object detection, highlighting the active configuration's branch.
- **Opt-in tracking and publication:** local MLflow, Weights & Biases, Hugging Face Hub, Zenodo draft, private GitHub repository report and DVC, each with a local preview before anything is sent, plus Google Drive datasets for Colab classification runs.
- **Paper figures and privacy defaults:** 300 dpi PNG and vector (`pdf`, `svg`) figure output, `replicates_strip` plots, and third-party SDK telemetry disabled by default.

## Planned next

- **Paired comparison in the GUI** (`replicated-comparison`) across all five task families, with confidence intervals matched to the paired test (Student's *t* or Hodges-Lehmann).
- **License boundary for detection backends:** move `ultralytics` (AGPL-3.0) to an optional `detection` extra while keeping `torchvision` detectors available in the MIT base install, accompanied by `THIRD_PARTY_NOTICES.md`.
- **Guaranteed offline mode:** an explicit offline switch that blocks non-loopback network sockets during training, verified by automated tests.
- **Apple Silicon GPU (`mps`):** MPS device selection on macOS once validated on Apple Silicon hardware.
- **Independent bundle reproduction:** `visionforge reproduce <bundle>` to re-train a seed from a verified experiment bundle and compare the result against its recorded confidence interval.

## Criteria for `1.0.0`

1. **Frozen 1.x contracts:** stable configuration schema (with tested forward migration from all `0.x` versions), stable `run.json` provenance schema, and a versioned stability guarantee for `user_tasks/` and `user_models/`.
2. **Full strategy parity:** single run, multi-seed replicates, paired variant comparison, hyperparameter sweeps and K-fold cross-validation (where statistically valid) available in both the GUI and `visionforge run` across the five task families.
3. **Verifiable empirical record:** public `run.json` artifacts and reproducible summaries for the multi-seed variance study and the companion software paper.

---

# Português (pt-BR)

O VisionForge está abaixo da `1.0.0`: o schema de configuração e a API HTTP ainda podem mudar entre versões menores, enquanto a migração por `schema_version` mantém funcionando os arquivos YAML já exportados. Este documento resume o que já está na branch `development` para a próxima release, o que vem a seguir e os critérios para a `1.0.0`, sem prazos prometidos.

- **O que já saiu em cada versão:** [`CHANGELOG.md`](../CHANGELOG.md)
- **Decisões de arquitetura e estatística (ADRs):** [`docs/dev/DECISIONS.md`](dev/DECISIONS.md)
- **Backlog ativo e issues abertas:** [`docs/dev/BACKLOG.md`](dev/BACKLOG.md) e [Issues no GitHub](https://github.com/marcus-vreis/VisionForge/issues)
- **Plano inicial pré-0.2 (histórico):** [`docs/archive/ROADMAP.md`](archive/ROADMAP.md)

## Em `development` (próxima release)

- **Réplicas na classificação e estratégias no CLI:** réplicas multi-semente na aba Classificação e envelopes YAML `strategy:` + `config:` (`replicates`, `replicated_comparison`, `sweep`, `cv`) no `visionforge run`.
- **Repetir pelo Histórico:** carregar a configuração de uma execução no formulário ou repeti-la com 3 novas sementes direto do Histórico.
- **Explorador e operações seguras de dataset:** aba **Explorar** em **Datasets** (contagens por classe e divisão, galeria de miniaturas com caixas e varredura SHA-256 em segundo plano) e criação de datasets derivados em pasta irmã (incorporar imagens, juntar datasets, refazer divisão com teste travado por padrão e converter entre YOLO e COCO).
- **Aba Código nas cinco tarefas:** leitura das funções reais do pacote instalado (`inspect`) para classificação, regressão, segmentação, anomalia e detecção `torchvision`, marcando o ramo ativo da configuração.
- **Acompanhamento e publicação opt-in:** MLflow local, Weights & Biases, Hugging Face Hub, rascunho no Zenodo, relatório em repositório privado do GitHub e DVC, sempre com prévia local antes do envio, além de datasets no Google Drive para treinos no Colab.
- **Figuras vetoriais e privacidade por padrão:** saída em PNG a 300 dpi e vetorial (`pdf`, `svg`), gráfico `replicates_strip` e telemetria de SDKs de terceiros desligada por padrão.

## Próximos passos

- **Comparação pareada na GUI** (`replicated-comparison`) nas cinco famílias de tarefa, com intervalo de confiança alinhado ao teste pareado (*t* de Student ou Hodges-Lehmann).
- **Fronteira de licença nos backends de detecção:** tornar `ultralytics` (AGPL-3.0) um extra opcional `detection`, mantendo os detectores `torchvision` na instalação base MIT, junto com o `THIRD_PARTY_NOTICES.md`.
- **Modo offline garantido:** opção explícita que bloqueia sockets de rede fora de loopback durante o treino, provada por teste automatizado.
- **GPU Apple Silicon (`mps`):** suporte ao dispositivo MPS no macOS após validação em hardware Apple Silicon.
- **Reprodução independente de bundle:** `visionforge reproduce <bundle>` para retreinar uma semente a partir de um pacote de experimento verificado e comparar o resultado com o intervalo de confiança registrado.

## Critérios para a `1.0.0`

1. **Contratos 1.x estáveis:** schema de configuração estável (com migração testada a partir de todas as versões `0.x`), contrato `run.json` estável e garantia explícita de estabilidade para `user_tasks/` e `user_models/`.
2. **Paridade completa de estratégias:** treino simples, réplicas multi-semente, comparação pareada entre variantes, buscas de hiperparâmetros e validação cruzada K-fold (onde estatisticamente válida) disponíveis tanto na GUI quanto em `visionforge run` nas cinco tarefas.
3. **Registro empírico verificável:** artefatos `run.json` públicos e sumários regeneráveis para o estudo de variância entre sementes e o artigo do software.
