# VisionForge — Documento de Contexto do Projeto

> **Autor:** Marcus Reis  
> **Versão atual:** `0.0.1`  
> **Python:** `>=3.13`  
> **Paradigma:** Modular / Plugin-based · Pesquisa Acadêmica  
> **Stack principal:** PyTorch · torchvision · scikit-learn · FastAPI · React · ONNX

---

## 1. O que é o VisionForge

VisionForge é uma plataforma local de treinamento e experimentação de modelos de visão computacional baseada em PyTorch. Seu objetivo central é **automatizar e padronizar o ciclo completo de experimentos de classificação de imagens**, desde a configuração de hiperparâmetros até a exportação do modelo treinado para produção.

O projeto nasce da necessidade prática de pesquisa: rodar dezenas de experimentos com diferentes arquiteturas, estratégias de treinamento e configurações — de forma reproduzível, rastreável e sem reescrever código a cada nova hipótese.

O sistema é operado **localmente**, aproveitando a GPU da máquina do pesquisador via CUDA, e expõe uma interface visual no browser via React + FastAPI para que o ciclo experimental seja acessível sem exigir linha de comando para cada operação.

---

## 2. Estado atual do projeto (v0.0.1)

O projeto está na **Fase 1 — Fundação**. O que existe e funciona hoje:

### ✅ Implementado e testado

#### `src/visionforge/utils/config.py`
O sistema de configuração é o módulo mais completo do projeto. Utiliza **Pydantic v2** para validar todos os parâmetros do experimento antes da execução. Possui cinco modelos encadeados:

- `ModelConfig` — arquitetura (`resnet18/34/50/101`, `efficientnet_b1/b7`, `vgg16/19`, `alexnet`), `num_classes`, `pretrained`
- `TrainingConfig` — `learning_rate` (gt=0), `epochs`, `batch_size` (validado como potência de 2), `early_stopping_patience`, `optimizer` (`adam/sgd/adamw`), `weight_decay`
- `DataConfig` — `base_dir` (validado para existir em disco), `train/val/test_dir`, `image_size`, `num_workers`, `pin_memory`
- `OutputConfig` — caminhos de saída com defaults em `outputs/{models,graphics,logs,reports}/`
- `ExperimentConfig` — top-level, com `model_validator` que garante coerência entre `task` e `num_classes` (binary → 1, multiclass → ≥2)

A função `load_config(path)` lê um YAML, valida o conteúdo e retorna um `ExperimentConfig` completamente tipado.

#### `src/visionforge/utils/logger.py`
Logger baseado em **loguru** com dois sinks configurados via `setup_logger()`:
- **Terminal:** colorizado, nível configurável (default `DEBUG`)
- **Arquivo:** `outputs/logs/visionforge.log`, rotação em 10 MB, retenção de 7 dias, compressão zip

#### `src/visionforge/utils/cuda.py`
CUDA/GPU verification module. Uses `torch.cuda` to detect available hardware. Exposes:
- `CUDAInfo` — immutable dataclass (`frozen=True`) with fields: `available`, `device_count`, `current_device`, `device_name`, `cuda_version`, `devices: tuple[GPUDevice, ...]`
- `GPUDevice` — per-GPU detail (`index`, `name`, `total_memory_mb`, `compute_capability`) consumed by the GUI device selector
- `check_cuda()` — returns `CUDAInfo` without ever raising (safe fallback to `available=False`)
- `log_cuda_status()` — logs a one-line diagnostic via loguru (`INFO` with GPU, `WARNING` without)

#### `src/visionforge/utils/config.DeviceConfig`
`DeviceConfig` (in `config.py`) declares which compute device the experiment should use:
- `kind`: `"cpu"`, `"cuda"` or `"multi_cuda"` (DataParallel)
- `gpu_ids`: optional explicit list of GPU indices

`Trainer.resolve_device()` materialises the choice at runtime, falling back to CPU (with a warning) when CUDA is requested but unavailable. The resolved device label is persisted to `run.json` under `device_used` so the user can verify *which device actually ran* — not just what was requested.

#### `src/visionforge/__main__.py`
Entry point do pacote. Inicializa o logger e exibe mensagem de boot. Acessível via `python -m visionforge` ou comando `visionforge` após instalação.

#### `configs/baseline.yaml`
Configuração de referência: ResNet50, classificação binária, dataset `USK-Coffee_Binary`, LR `0.0001`, 100 epochs, batch 16, early stopping com paciência 10.

### 🔲 Estrutura criada, aguardando implementação

Os seguintes módulos têm apenas `__init__.py` vazio — são o próximo passo de desenvolvimento:

```
src/visionforge/
├── blocks/    # ✅ ExperimentBlock ABC + ClassificationBlock + BlockRegistry
├── core/      # ✅ Trainer, Evaluator, DataModule, MetricsPlotter
├── gui/       # ✅ FastAPI server + React SPA (MVP)
└── models/    # ✅ ModelFactory
```

---

## 3. Arquitetura

```
CI/CD (GitHub Actions)
    ├── pre-commit: ruff (lint+format) · codespell · mypy · pytest
    ├── CI (ci.yml): spell check · lint · type check · pytest · coverage · SonarCloud
    └── CD (cd.yml): build wheel + sdist via uv · attach to GitHub Release (on tag v*.*.*)
         │
         ▼
React SPA (browser local) ← servido como arquivos estáticos
    └── ConfigForm (schema-driven) · ExperimentRunner · ResultsView
         │ HTTP (fetch)
         ▼
FastAPI (mesmo processo Python)
    └── /api/schema · /api/experiment/* · /api/artifacts/*
         │ import direto
         ▼
Plugin Blocks (ExperimentBlock ABC)
    └── BlockRegistry (auto-descoberta em blocks/)
    └── Trainer · DataModule · Evaluator · ModelFactory
         │
         ▼
Storage (outputs/)
    └── models/ · graphics/ · logs/ · reports/
```

### Blocos de experimento planejados

| Bloco | Função |
|---|---|
| **GridSearchBlock** | Varredura exaustiva de hiperparâmetros via YAML |
| **RandomSearchBlock** | Amostragem aleatória com `n_trials` configurável |
| **CrossValidationBlock** | K-Fold e Stratified K-Fold com métricas por fold |
| **TransferLearningBlock** | Feature extraction vs fine-tuning, LR diferencial |
| **ModelComparisonBlock** | Ranking de N modelos por F1, AUC, tempo |
| **BatchPredictionBlock** | Inferência em lote, saída CSV |
| **ExportONNXBlock** | Export + validação + benchmark de latência |

Cada bloco implementará uma interface `ExperimentBlock` com `setup()`, `run()` e `report()`.

---

## 4. Pipeline de qualidade (detalhamento real)

### pre-commit (`.pre-commit-config.yaml`)
Roda localmente a cada commit:
- **ruff** (`v0.15.6`) — lint com `--fix` + formatação
- **codespell** (`v2.4.2`) — correção de typos com `--write-changes`
- **mypy** (`v1.19.1`) — type checking com dependências extras (`loguru`, `pydantic>=2.0`, `types-PyYAML`)
- **pytest** — testes completos (`always_run: true`, mas pulado no CI com `SKIP: pytest`)

> **Nota:** No CI, o pytest do pre-commit é pulado (`SKIP: pytest`) porque existe um job dedicado `tests` com configuração mais completa (coverage, XML para SonarCloud, threshold de 70%).

### CI workflow (`ci.yml`) — jobs em sequência

```
pre-commit ──┐
             ├── main (lint · format · mypy · version check)
pre-commit-  │        │
check-hooks ─┘        ▼
                    tests (pytest --cov --cov-fail-under=70)
                        │
                   ┌────┴────┐
                coverage   sonarqube
                (artifact)  (só em PR)
```

- **`pre-commit`** — roda todos os hooks exceto pytest
- **`pre-commit-check-hooks-versions`** — detecta hooks desatualizados com `pre-commit autoupdate` (emite warning, não bloqueia)
- **`main`** — spell check, ruff check, ruff format --check, mypy, verificação de version bump em PRs
- **`tests`** — pytest com coverage XML + HTML, threshold mínimo 70%, upload dos artefatos
- **`coverage`** — republica o report HTML como artefato de 30 dias
- **`sonarqube`** — SonarCloud scan com o coverage.xml gerado (só em pull_request)

**Instalação no CI:** usa `uv` para setup do venv + torch/torchvision da CPU wheel (`--index-url https://download.pytorch.org/whl/cpu`).

### CD workflow (`cd.yml`)
Dispara em push de tags `v*.*.*` ou release publicado:
- Build de wheel + sdist via `uv build`
- Upload como artefato (retenção 90 dias)
- Attach automático ao GitHub Release

### SonarCloud (`sonar-project.properties`)
- `projectKey`: `marcus-vreis_VisionForge`
- `organization`: `marcus-vreis`
- Analisa `src/` (`.py`), testa com `tests/`
- Consome `coverage.xml` gerado pelo pytest

### Ferramentas de desenvolvimento (`pyproject.toml`)
- **ruff** — substitui black + flake8 + isort (line-length 88, target py313)
- **mypy** — `warn_return_any=true`, exclui `outputs/` e `configs/`
- **codespell** — ignora abreviações de ML (`nd`)
- **bump-my-version** — versionamento semântico com commit + tag automáticos
- **pytest** — `testpaths=tests`, coverage em `src/`, relatório `term-missing`
- **coverage** — omite `__init__.py`, exclui `pragma: no cover`

---

## 5. Testes existentes

### `tests/smoke_test.py`
Verifica que os módulos implementados são importáveis e expõem a superfície pública esperada (`ExperimentConfig`, `load_config`, `logger`, `setup_logger`, `main`).

### `tests/utils/test_config.py`
Suite completa para o sistema de configuração. Cobre:
- Carregamento de config válida e defaults do `OutputConfig`
- `FileNotFoundError` para arquivo inexistente
- Modelo inválido (`resnet999`) e `num_classes=0`
- `learning_rate` negativo, `epochs=0`
- `batch_size` não potência de 2 (+ todos os tamanhos válidos de 1 a 128)
- Optimizer inválido (`rmsprop`)
- `base_dir` inexistente
- Inconsistência `binary/multiclass` vs `num_classes`

### `tests/utils/test_logger.py`
Suite completa para o logger. Cobre:
- Setup sem exceção, criação do arquivo, escrita de mensagens
- Todos os níveis capturados no arquivo (DEBUG mesmo com terminal em INFO)
- Double setup sem duplicar mensagens no terminal
- Criação do diretório de log se não existir
- Fallback para `_DEFAULT_LOG_DIR` quando `log_dir=None`

### `tests/utils/test_cuda.py`
Suite completa para verificação CUDA. Todos os testes usam mock de `torch.cuda` para funcionar sem GPU. Cobre:
- Dataclass `CUDAInfo` — defaults corretos, instanciação completa, imutabilidade (`frozen`)
- `check_cuda()` — CUDA disponível (single GPU), indisponível, múltiplas GPUs, `RuntimeError` no torch, `ImportError` se torch ausente
- `log_cuda_status()` — emissão de `INFO` com CUDA e `WARNING` sem CUDA (captura via sink loguru)
- Exports do módulo — `__all__` com exatamente 3 nomes públicos

### `tests/conftest.py`
Fixture `project_root` com escopo de sessão — diretório temporário para testes de integração futuros.

---

## 6. Cuidados e pontos de atenção

### 6.1 `base_dir` validado em tempo de config — cuidado no CI
O `DataConfig` valida que `base_dir` existe em disco. Isso é ótimo em produção, mas qualquer teste que instancie `DataConfig` com um caminho real vai quebrar em máquinas sem o dataset. Os testes atuais contornam isso com `tmp_path`. Futuros testes de integração precisam manter esse padrão rigorosamente.

### 6.2 O schema YAML está evoluindo — congele cedo
O `baseline.yaml` e os modelos Pydantic ainda podem mudar enquanto o projeto está em `0.0.1`. Cada mudança de campo obrigatório quebra configs salvas. Antes de acumular experimentos reais, defina a versão do schema e adicione um campo `schema_version` ao YAML — isso vai permitir migrações futuras sem perder rastreabilidade.

### 6.3 Reprodutibilidade — seed ausente
O `ExperimentConfig` ainda **não tem campo de seed**. Em pesquisa acadêmica isso é crítico: dois runs com a mesma config podem produzir resultados diferentes. Adicionar `seed: int = 42` ao `TrainingConfig` deve ser feito antes de qualquer experimento sério — e o seed precisa ser aplicado em PyTorch, NumPy e Python random no início de cada run.

### 6.4 Threshold de coverage (70%) vai pressionar à medida que o código crescer
Hoje o código real é pequeno e os testes são densos — 70% é fácil de manter. Quando `core/`, `blocks/` e `gui/` forem implementados, manter esse número vai exigir disciplina. Considere subir para 80% quando o `Trainer` e os `Blocks` estiverem prontos.

### 6.5 `pin_memory: true` com `num_workers=0` é inútil
No `baseline.yaml`, `pin_memory: true` e `num_workers: 4` fazem sentido juntos. Nos testes, `num_workers=0` é usado corretamente para CI. Considere adicionar um `model_validator` que emita um aviso quando `pin_memory=True` e `num_workers=0` — combinação que não gera erro mas não traz benefício nenhum.

### 6.6 SonarCloud só roda em PR
O job `sonarqube` tem `if: github.event_name == 'pull_request'`. Push direto em `main` ou `develop` não dispara análise estática. Avaliar se faz sentido rodar também em push para `develop`.

### 6.7 Gerenciamento de memória GPU (futuro — quando blocks forem implementados)
Quando o `ModelComparisonBlock` e o `GridSearchBlock` forem implementados, haverá risco de vazamento de VRAM entre trials. Garantir `torch.cuda.empty_cache()` e deleção explícita de objetos de modelo entre execuções.

### 6.8 Data leakage no K-Fold (futuro — quando CrossValidationBlock for implementado)
A normalização (média/desvio padrão) deve ser calculada **apenas sobre o fold de treino** e aplicada ao fold de validação. Calcular sobre o dataset completo antes da divisão é data leakage e invalida os resultados em contexto acadêmico.

---

## 7. O que o VisionForge deve se tornar

### 7.1 Próximos passos imediatos (Fase 1 → 2)

1. **Adicionar `seed` ao `TrainingConfig`** — antes de qualquer experimento real
2. **Implementar `ModelFactory`** em `src/visionforge/models/` — suporta as arquiteturas já listadas no `Literal` do config
3. **Implementar `DataModule`** em `src/visionforge/core/` — transforms, augmentation, loaders respeitando splits
4. **Implementar `Trainer`** em `src/visionforge/core/` — com early stopping, checkpoint, histórico JSON por run
5. **Definir ABC `ExperimentBlock`** em `src/visionforge/blocks/` — contrato `setup/run/report`
6. **Implementar `BlockRegistry`** — auto-descoberta de blocos na pasta `blocks/`

### 7.2 Médio prazo — expansão científica

- **Bayesian Optimization Block** (Optuna) — substituir grid/random search por otimização bayesiana, reduzindo drasticamente o número de trials
- **Explainability Block** — Grad-CAM, LIME e SHAP integrados nativamente para interpretação de decisões (essencial em pesquisa)
- **Calibration Block** — Platt Scaling, Temperature Scaling para análise de confiança das predições
- **Dataset Analysis Block** — estatísticas, detecção de desbalanceamento, distribuição de classes antes do treino
- **Augmentation Search Block** — AutoAugment / RandAugment automatizado

### 7.3 Médio prazo — infraestrutura

- **`schema_version` no YAML** com sistema de migração para configs antigas
- **MLflow ou W&B** — integração opcional de rastreamento além do JSON local
- **Suporte a HuggingFace Datasets** — carregar benchmarks acadêmicos diretamente
- **Relatórios LaTeX** — exportar tabelas de resultados em `.tex` para inserção direta em artigos

### 7.4 Visão de longo prazo

O VisionForge tem potencial para se tornar um **laboratório pessoal de benchmark** — onde o pesquisador mantém um histórico completo, rastreável e reproduzível de todos os experimentos realizados ao longo de uma linha de pesquisa.

Em contexto acadêmico, isso significa responder com precisão: *"qual foi exatamente a configuração que produziu esse resultado na Tabela 3?"* — e reproduzi-la com um único comando YAML.

A arquitetura modular e o sistema de configuração com validação forte já sustentam essa visão. O próximo gargalo crítico é o **contrato do `ExperimentBlock`**: uma vez que ele esteja estável e testado, adicionar experimentos será uma operação de baixo risco e alta rastreabilidade.

---

## 8. Estrutura de arquivos atual

```
visionforge/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Code quality: pre-commit · lint · mypy · tests · sonarqube
│       └── cd.yml              # Build & Release: uv build · attach to GitHub Release
├── configs/
│   ├── baseline.yaml           # Config de referência: ResNet50 binary (USK-Coffee)
│   └── synthetic_test.yaml     # Config leve para testes rápidos
├── frontend/                   # React source (dev-time only, não entra no wheel)
│   ├── src/
│   │   ├── api/client.ts       # Wrappers tipados para endpoints da API
│   │   ├── components/         # ConfigForm, ResultsView, ExperimentRunner
│   │   ├── hooks/              # useExperiment (lifecycle do experimento)
│   │   └── types/              # TypeScript types para schema e run.json
│   ├── vite.config.ts          # Build → src/visionforge/gui/static/
│   └── package.json
├── src/
│   └── visionforge/
│       ├── __init__.py
│       ├── __main__.py         # Entry point: subcommands "run" (CLI) e "gui" (web)
│       ├── blocks/
│       │   ├── base.py         # ✅ ExperimentBlock ABC
│       │   ├── classification.py # ✅ ClassificationBlock (train + evaluate + plots)
│       │   └── registry.py     # ✅ BlockRegistry (auto-descoberta)
│       ├── core/
│       │   ├── data.py         # ✅ DataModule (ImageFolder + transforms)
│       │   ├── evaluator.py    # ✅ Evaluator (Accuracy, F1, AUC-ROC, confusion matrix)
│       │   ├── plotter.py      # ✅ MetricsPlotter (loss curve, confusion matrix PNG)
│       │   └── trainer.py      # ✅ Trainer (early stopping, checkpoint, run.json)
│       ├── gui/
│       │   ├── server.py       # ✅ FastAPI app + SPA fallback
│       │   ├── api/
│       │   │   ├── routes.py   # ✅ REST endpoints (/api/schema, /api/experiment/*)
│       │   │   └── schemas.py  # ✅ RunStatus, RunResponse, RunResult
│       │   └── static/         # ✅ React SPA pré-compilado (gerado por npm run build)
│       ├── models/
│       │   └── factory.py      # ✅ ModelFactory (ResNet, EfficientNet, VGG, AlexNet)
│       └── utils/
│           ├── config.py       # ✅ ExperimentConfig + load_config (Pydantic v2)
│           ├── cuda.py         # ✅ CUDAInfo + check_cuda + log_cuda_status
│           └── logger.py       # ✅ setup_logger (loguru, terminal + arquivo)
├── tests/                      # 126 testes passando, 82% coverage
├── .pre-commit-config.yaml
├── pyproject.toml
└── sonar-project.properties
```

---

## 9. Code writing rules

These rules apply to all source code and tests:

- Every public class and function gets a single-line description.
- `Args:` and `Returns:` only when the signature or return value is non-obvious.
- `Raises:` only when the caller needs to handle a specific exception.
- Pydantic model fields document themselves via name and `Field()` — no `Args:` block on model classes.
- Comments explain *why*, not *what*. Skip the comment if the code is clear.
- No unnecessary capitalization or filler phrases.
- A docstring that reads like AI wrote it needs trimming.

### Documentation

Document decisions, not options. When a choice is made (framework, library, pattern), write down the decision and its reason. If the decision changes, update the doc. Documentation and code must always agree.

---

## 10. GUI

The GUI uses **React + shadcn/ui** (frontend) served by **FastAPI** (backend) in the same Python process. Training runs via `asyncio.to_thread()` so the API stays responsive while PyTorch uses the GPU — no separate worker process, no IPC.

- **Launch:** `python -m visionforge gui` starts FastAPI + serves the pre-built React SPA. Users never need Node.js.
- **Config form:** Auto-generated from `ExperimentConfig.model_json_schema()`. Pydantic `Literal` fields become Select dropdowns, numbers become Input fields, booleans become Switch toggles, nested models become Card sections.
- **Training:** `POST /api/experiment/run` validates the config via Pydantic and runs the experiment in a background thread. The frontend polls `GET /api/experiment/status` and consumes a live SSE stream at `/api/experiment/events`.
- **Results:** Metrics grid + plot images served from `outputs/` via `GET /api/artifacts/{path}`.
- **Dev mode:** `cd frontend && npm run dev` starts Vite with hot-reload, proxying `/api` to the FastAPI server.

### 10.1 Device selection (real, not cosmetic)

`GET /api/device/info` returns the live list of compute devices (`cpu_name`, `cuda_available`, `cuda_version`, and one `GPUInfo` per detected GPU). The header / bottom-bar `DeviceSelector` component is driven entirely by this endpoint:

- The dropdown only offers options that *actually exist* (CPU, each individual GPU, "Multi-GPU" only when ≥ 2 GPUs).
- The chosen `{kind, gpu_ids}` is injected into the experiment config on submit (`device: ...`) so the backend honours it.
- `Trainer.resolve_device()` records what was used (e.g. `"cuda:0 (NVIDIA RTX 4090)"` or `"cpu (fallback: CUDA unavailable)"`) into `run.json` under `device_used`, so the user can verify which device actually ran.

### 10.2 Dataset folder picker

Browsers cannot return absolute filesystem paths via `showDirectoryPicker` (sandbox restriction). To deliver a *real* path, `POST /api/dataset/pick` opens a native `tkinter` directory dialog on the server. Because VisionForge always runs locally, this is safe: the user sees the OS-native picker and the absolute path is sent back to the frontend.

### 10.3 Plots

`MetricsPlotter` emits, when defined, the following per-run PNGs into the run directory:

- `loss.png` — train + val loss curves
- `accuracy.png` — train + val accuracy curves
- `confusion_matrix.png` — raw counts heatmap
- `confusion_matrix_normalized.png` — row-normalized (per-class recall)
- `roc_curve.png` — ROC (binary or one-vs-rest multiclass; skipped when only one class is present)
- `precision_recall_curve.png` — precision-recall curve (same conditions)

### 10.4 Run history & per-model test runs

- `GET /api/runs` — paginated summaries.
- `GET /api/runs/{run_id}` — full `RunDetail` with config, metrics, history, all artifact paths, device used, and accumulated tests.
- `POST /api/runs/{run_id}/test` — reloads the saved checkpoint, runs `Evaluator` on a new dataset path, generates a confusion matrix + ROC for that test, and appends the result to `run.json` under `tests[]`. Each saved model thus accumulates its own per-dataset test history.

In the frontend, clicking a `RunCard` in `HistoryOverlay` opens `RunDetailPanel`, which:
- shows the absolute `run_dir` and checkpoint path (with copy-to-clipboard buttons),
- displays every metric and every generated plot (click any plot → full-screen `Lightbox`),
- exposes a "+ testar" form that calls `/api/runs/{run_id}/test` and lists every prior test with its dataset path, metrics, and per-test plots.