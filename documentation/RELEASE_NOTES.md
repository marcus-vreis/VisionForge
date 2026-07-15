# VisionForge — Novidades do último mês (2026-06-14 → 2026-07-12)

> Documento de apresentação. Cada item aponta o ADR e o commit que o entregou;
> tudo abaixo está na `main` com CI verde. Roteiro de demonstração ao final.
> Como rodar passo a passo: `README.md` (seções *Installation* e *Quickstart*)
> e `docs/QUICKSTART.md`.

## Resumo executivo

Em um mês o VisionForge saiu de "cinco tarefas funcionais com painéis
inconsistentes" para uma plataforma **padronizada, estatisticamente rigorosa,
pronta para adoção externa e extensível pelo próprio pesquisador** — a nova
superfície de SDK permite definir uma família de tarefa inteira em um único
arquivo Python e ganhar GUI, monitor ao vivo, proveniência, sweeps e réplicas
de graça.

## 1. Rigor estatístico — números defensáveis

- **Réplicas multi-seed (ADR-056,** `34f1e4b`**)** — treine a mesma config N
  vezes sob seeds diferentes e reporte `métrica = média ± IC 95%` (t de
  Student) em vez de um ponto único. Réplicas **nunca são ranqueadas** — são
  amostras de uma distribuição, não competidoras. Disponível em todas as
  tarefas via seletor de estratégia.
- **Proveniência de hardware (ADR-057,** `34f1e4b`**)** — todo `run.json`
  agora grava CUDA, cuDNN e modelo de GPU além de Python/torch/numpy/seed/
  config completa: o resultado carrega o ambiente exato que o produziu.

## 2. Padronização total dos painéis (ADR-059, bricks A–F — completo)

O painel de classificação foi ratificado como contrato canônico e os quatro
painéis standalone foram trazidos a ele (`1850a16` → `b79c840`):

- **Cabeçalho canônico** (`99f7957`): nome do experimento + botões exportar/
  importar YAML lado a lado + seletor de estratégia abaixo, **na mesma caixa**
  — idêntico em todas as tarefas. Import de YAML valida contra o schema vivo
  e reconstrói o formulário com conversores de ida-e-volta testados
  (`formFromPayload(buildPayload(form)) == form`).
- **Estratégia como seletor de primeira classe** (`a64c06d`): `Treino simples |
  K-Fold (onde existe) | Sweep | Réplicas` — controle segmentado que
  transforma o formulário, em vez de cards empilhados misturados com
  hiperparâmetros.
- **Ordem canônica de seções** (`1850a16`): Nome → Estratégia → Modelo →
  Treinamento → Dataset (+stats) → Pré-processamento → Augmentação (+preview).
  Augmentação silenciosa exposta (antes rodava sem o usuário ver).
- **"Comparar arquiteturas" deixou de ser um card separado** (`b79c840`):
  virou um preset de grid de um eixo sobre `model.name` — comparação de
  verdade acontece no histórico.
- **Dataset stats em todas as tarefas** (`4a48bf0` + `b24aa02`): contagens por
  classe/split, pareamento imagem-máscara, distribuição de alvos — antes de
  gastar GPU.

## 3. Paridade entre tarefas

- **K-fold CV para regressão** (ADR-050, `69ffdfe`) **e segmentação**
  (`73df2a6`) — métricas fold a fold + média ± desvio, com transforms sem
  vazamento entre folds. Classificação já tinha; paridade completa.
- **TensorBoard em tudo** (ADR-054, `d9a31da`) — escalares por época em
  `<run_dir>/tensorboard/` para as cinco tarefas, incluindo anomalia e
  detecção torchvision.

## 4. Download de datasets integrado (ADR-055, `d2ae95c`)

Card "Baixar dataset" em qualquer painel: torchvision (CIFAR10 etc., com
`limit` por classe), Roboflow, Kaggle e Hugging Face — materializado no layout
exato que a tarefa consome. Zero fricção para demonstrar a plataforma.

## 5. SDK de tarefas custom — a grande novidade (ADR-058, bricks 1–5)

O pesquisador define **uma família de tarefa inteira em um arquivo Python**:

```bash
visionforge new-task contagem_celulas   # o arquivo gerado JÁ TREINA (dados sintéticos)
```

- **Brick 1** (`3eccc8a`) — `BaseTaskConfig` (compõe os blocos compartilhados:
  training/data/output/device), `TaskSpec` com 4 hooks (`build_model`,
  `build_loaders`, `compute_loss`, `compute_metrics`), `@register_task` (nome,
  **cor da aba**, descrição, métricas com direção `higher`/`lower`) e
  descoberta automática em `user_tasks/`.
- **Brick 2** (`eebc2bc`) — `GenericTaskEngine`: o loop genérico cuida de
  seed (+ modo determinístico), device, AMP, early stopping, melhor
  checkpoint pela métrica primária (ciente da direção), eventos SSE,
  TensorBoard, curva da métrica em PNG e `run.json` versionado. Nível 2:
  `run(cfg, ctx)` para treinos que não são em épocas (GANs, EM).
- **Brick 3** (`45ca6b8`) — API: `GET /api/tasks` (built-ins + customs, com
  scan a cada chamada — task nova aparece **sem reiniciar o servidor**),
  `GET /api/custom/{key}/schema` (formulário validado nasce do Config),
  `POST /api/custom/{key}/run` (mesmo monitor SSE das tarefas nativas).
- **Brick 4** (`97be0c1`) — `CustomTaskRunner`: **sweeps e réplicas de graça**
  (`/api/custom/{key}/sweep`, `/api/custom/{key}/replicates`) usando os mesmos
  orquestradores das tarefas nativas; o espaço de busca valida contra o Config
  da própria tarefa (dá para varrer o campo que o pesquisador declarou).
  Decisão documentada: sem endpoint de compare para customs — comparação é um
  sweep de um eixo.
- **Brick 5** (`5770a04`) — `visionforge new-task` (scaffolder CLI), exemplo
  funcional `user_tasks/example_counting/` (CNN conta pontos em imagens
  sintéticas — treina em segundos na CPU, sem dataset em disco) e
  `user_tasks/README.md` (guia PT + EN). O CI treina o template gerado e o
  exemplo de verdade — template quebrado = build vermelho. `TaskSpec` é
  genérico (PEP 695): hooks tipados com o próprio Config, sem `type: ignore`.

Aprendizado técnico registrado (ADR-030): datasets definidos no arquivo da
tarefa exigem `num_workers=0` — workers spawn não re-importam módulos
carregados por caminho. Documentado no template, no exemplo e no README.

## 6. Pronto para adoção externa (Fase C — completa)

- **README em inglês com screenshots** (`0873300`, `68873f2`) — pitch,
  tabela de tarefas, instalação com `visionforge doctor`.
- **CITATION.cff + licença MIT** (`0873300`) — o GitHub renderiza "Cite this
  repository".
- **Pacote PyPI real** (`65bee61`) — a SPA compilada embarca no wheel
  (usuário final não precisa de Node) + job de **Trusted Publishing** (OIDC,
  sem tokens) no `cd.yml` disparado por tags `v*`.
- **QUICKSTART** (`523318e`) — do clone a um resultado defensável (réplicas
  com IC) em ~10 minutos, usando o download de dataset embutido.

## 7. Higiene de engenharia

- `main` promovida e sincronizada com `development`; 15 branches obsoletas
  removidas; cada fatia entra com CI verde (fluxo branch → development → main).
- Correções de compatibilidade de teste entre versões do Starlette
  (`6e9af6e`, `acd907e`).

---

## Como rodar (passo a passo)

```bash
# 1. instalar
git clone https://github.com/marcus-vreis/VisionForge.git
cd VisionForge
uv venv
.venv\Scripts\activate                      # Linux/macOS: source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. torch para o SEU hardware (o doctor imprime a linha exata)
visionforge doctor
# ex.: uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. construir a interface uma vez (depois é servida pelo Python)
cd frontend && npm install && npm run build && cd ..

# 4. abrir
visionforge gui                             # http://127.0.0.1:8000
```

Guia detalhado com dataset de exemplo: `docs/QUICKSTART.md`.

## Roteiro de demonstração sugerido (15 min)

1. **Abertura** — `visionforge gui`; mostrar as cinco abas e o cabeçalho
   idêntico em todas (nome + YAML + estratégia).
2. **Dataset sem fricção** — card "Baixar dataset" → torchvision CIFAR10 com
   `limit=300`; stats por classe aparecem antes de treinar.
3. **Treino ao vivo** — Classificação, 3 épocas; monitor SSE; ao final:
   métricas, matriz de confusão, model card; abrir `outputs/models/<run>/run.json`
   e apontar o bloco `environment` (CUDA/cuDNN/GPU) — proveniência.
4. **Número defensável** — estratégia **Réplicas** (3 seeds em regressão ou na
   própria classificação): `média ± IC 95%` — o slide de "como reportar
   resultado".
5. **YAML round-trip** — Exportar YAML → `visionforge run <arquivo>.yaml` no
   terminal — GUI e CLI são o mesmo experimento.
6. **O gran finale — SDK**:
   ```bash
   visionforge new-task demo_apresentacao
   ```
   Mostrar o arquivo gerado (Config + 4 hooks comentados). Em outro terminal:
   ```bash
   curl http://127.0.0.1:8000/api/tasks              # a task nova já está na lista
   curl http://127.0.0.1:8000/api/custom/demo_apresentacao/schema
   ```
   E o exemplo que acompanha o repositório treina de verdade:
   `user_tasks/example_counting/` (contagem de pontos, segundos na CPU).
7. **Fechar** — CITATION.cff, MIT, wheel PyPI pronto: "instala com pip,
   cita com um clique".

## O que vem a seguir (honesto, para a seção de roadmap)

- **ADR-058 brick 6** — as tarefas custom como **abas reais na GUI**
  (formulário gerado do schema, resultados/histórico genéricos). A API já
  está 100%; falta o front.
- **Fase D — saídas de paper**: IC bootstrap, testes de significância
  pareados, fingerprint de dataset, export LaTeX.
- **Publicação no PyPI** — passo manual do mantenedor: registrar o Trusted
  Publisher em pypi.org e criar a tag `v0.1.0`.
