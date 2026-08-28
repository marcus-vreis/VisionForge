# Changelog

All notable changes to VisionForge are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0, the config schema and the HTTP API may change
between minor releases. Configs carry a `schema_version` and are migrated on
load (ADR-039), so a config exported from an older release keeps working.

Every entry links the ADR that records *why* the change was made; the full
reasoning lives in [`docs/dev/DECISIONS.md`](docs/dev/DECISIONS.md).

## [Unreleased]

## [0.9.0] — 2026-08-28

### Added

- **Um guia de primeira execução.** Depois do nome, quem nunca viu recebe um
  convite para uma volta de sete paradas: as abas de tarefa, a pasta do dataset,
  a divisão básico/avançado, GPU ou CPU, o botão de treinar, o histórico e os
  datasets. Cada parada acende o elemento de que está falando e traz um cartão
  ao lado; sai com ✕, "Pular" ou Esc, e o botão "guia" no cabeçalho reabre
  quando quiser ([ADR-104](docs/dev/DECISIONS.md)).
- **Ponto de info no dataset.** O diretório base e as três subpastas ganharam o
  mesmo "i" dos hiperparâmetros, explicando o que o VisionForge procura dentro
  da pasta raiz e para que serve cada divisão — inclusive por que o teste é
  avaliado uma vez só ([ADR-104](docs/dev/DECISIONS.md)).

- **Parâmetros básicos na frente, avançados recolhidos — nas cinco tarefas.** A
  classificação básico/avançado existia desde a reclamação "muita coisa junta" e
  nunca tinha sido usada. Ficam à vista os que mudam entre um experimento e
  outro (épocas, learning rate, batch size, seed, e a loss onde ela existe); o
  resto entra numa seção que abre sozinha se algum valor ali dentro estiver fora
  do padrão. Recolhidos: 6 na classificação e na detecção, 4 na anomalia, 3 na
  regressão e na segmentação ([ADR-102](docs/dev/DECISIONS.md)).


- **Três arquiteturas de atenção: `vit_b_16`, `swin_t` e `convnext_tiny`.** Vêm
  do torchvision, sem dependência nova. Foram medidas antes de entrar, e ainda
  bem: `swin_t` e `convnext_tiny` **colapsam** com Adam a 1e-3 (0.25 em quatro
  classes) exatamente como VGG e AlexNet. Com AdamW a 1e-4 chegam a 0.88 e 0.91 —
  o `convnext_tiny` é hoje o melhor classificador da lista
  ([ADR-100](docs/dev/DECISIONS.md)).
- **O formulário sugere o que foi medido para cada arquitetura.** Escolher
  `swin_t` mostra "prevê uma classe só: medimos 0.25 de acurácia" com um botão
  que aplica AdamW e 1e-4. Sugere e explica; não reescreve o seu campo por conta
  própria ([ADR-100](docs/dev/DECISIONS.md)).
- **`image_size` sugerido a partir do dataset.** O CIFAR-10 tem imagens de
  32×32 e treiná-lo a 224 amplia sete vezes, gastando ~50× mais computação em
  pixels que o redimensionamento inventou. O tamanho é lido da mediana do
  dataset, com piso de 64. Para ViT e Swin fica fixo em 224 — neles outro
  tamanho não é mais lento, é erro ([ADR-100](docs/dev/DECISIONS.md)).


- **O treino avisa quando não aprendeu nada.** Um modelo que prevê a mesma
  classe para todas as imagens ainda reporta uma acurácia — e num problema
  binário balanceado ela é exatamente 0.50, fácil de confundir com "aprendeu
  pouco". Agora o run detecta isso (uma classe só prevista, ou loss que nunca
  caiu) e diz, no log e no `run.json`, com a causa mais provável e o que mudar
  ([ADR-099](docs/dev/DECISIONS.md)).
- **Learning rate sugerido por arquitetura e otimizador.** Medimos a grade
  inteira: VGG16 e AlexNet com Adam a 1e-3 **colapsam** (0.25 em 4 classes),
  enquanto resnet50 e efficientnet com SGD a 1e-3 subtreinam. A separação segue
  o batch normalization. A sugestão é 1e-4 para Adam em VGG/AlexNet, 1e-2 para
  SGD e 1e-3 no resto — oferecida, nunca imposta
  ([ADR-099](docs/dev/DECISIONS.md)).
- **Ponto de info em cada parâmetro.** As explicações saíram de baixo dos campos
  e foram para um `i` ao lado do rótulo. O texto é o mesmo; o que muda é que as
  linhas do formulário voltam a ter a mesma altura — que era o que fazia o painel
  parecer desalinhado. Vale para classificação, detecção, regressão, segmentação,
  anomalia e tarefas próprias.

### Changed

- **O número de workers agora se decide sozinho.** O padrão passou a ser `-1`,
  que na hora de carregar mede a memória livre da máquina e divide pelo custo de
  um worker (no Windows, ~1 GB cada, por pool de loader). O campo continua na
  tela mostrando o que resolveu — "automático ≈ 3 agora" — e o botão "auto"
  entrega o número para quem quiser fixá-lo. O `0` mantém o sentido de sempre
  (carregar no processo principal) e um valor acima do orçamento continua sendo
  cortado com aviso, porque não começar é pior do que começar com menos
  ([ADR-103](docs/dev/DECISIONS.md)).
- **A detecção não manda mais `workers: 2` fixo.** O formulário do YOLO carregava
  esse literal desde o primeiro WinError 1455; agora usa o mesmo campo
  automático das outras tarefas ([ADR-103](docs/dev/DECISIONS.md)).

- **Defaults revistos por tarefa.** Classificação e regressão começam em
  `resnet18` (em vez de `resnet50`) e 20 épocas (em vez de 10); segmentação
  começa em `unet`, porque `deeplabv3_resnet50` a 512px esgota uma GPU modesta.
  **Anomalia (30 épocas) e detecção (100, o padrão do Ultralytics) ficaram como
  estavam** — já são adequados ao que essas tarefas precisam, e mexer neles por
  simetria seria pior que a assimetria ([ADR-100](docs/dev/DECISIONS.md)).


- **`early_stopping_patience = 0` agora desliga o early stopping, e é o
  padrão**, nas cinco tarefas e no SDK de tarefas próprias. Antes o valor era
  recusado (`ge=1`) e a leitura ingênua do laço faria o oposto do esperado: a
  primeira época sem melhora já satisfaz `1 >= 0` e encerraria o treino. O
  padrão virou 0 porque o anterior nunca disparava — com `epochs=10` ele exigia
  dez épocas seguidas sem melhora dentro de dez épocas, ou seja, prometia uma
  proteção que não existia.
- **Campo numérico vazio virou uma intenção, não um engano.** Ele restaurava o
  valor anterior em silêncio; onde zero desliga o parâmetro, esvaziar agora
  significa zero.

### Fixed

- **O formulário tinha os defaults antigos.** Os painéis das tarefas montam o
  estado inicial a partir de literais próprios, então cada default corrigido nos
  últimos dias continuava com o valor velho para quem usa a interface — inclusive
  `deterministic: false`, que **anulava a correção de reprodutibilidade**: todo
  treino iniciado pela GUI continuava não reprodutível. Também `early stop` em 10
  (em vez de 0), regressão em `resnet50`/50 épocas e segmentação em
  `deeplabv3_resnet50` ([ADR-102](docs/dev/DECISIONS.md)).
- **O campo de early stopping recusava 0.** Três painéis tinham `min={1}`, então
  digitar 0 — o valor que desliga — virava 1 em silêncio. Permitir no backend não
  adianta se a entrada arredonda ([ADR-102](docs/dev/DECISIONS.md)).


- **O "feature extraction" não congelava a rede em VGG e AlexNet.** A regra era
  congelar todos os filhos menos o último — e nessas duas o último filho é um
  bloco `classifier` com **três** camadas densas (102M + 16M + 4M). O modo cuja
  promessa inteira é manter o backbone parado deixava **89% e 96%** da rede
  treinável. A cabeça é a última camada Linear, não o último filho; corrigido na
  classificação e na regressão, que tinham a mesma regra e o mesmo defeito. A
  segmentação já estava certa e foi deixada como está: lá o último filho é o
  decoder, e treiná-lo com o backbone congelado é justamente o que se faz
  ([ADR-101](docs/dev/DECISIONS.md)).
- **As checagens de colapso chegaram às outras quatro tarefas.** Antes só a
  classificação detectava; a detecção não tinha nenhuma. Agora cada tarefa é
  perguntada no próprio vocabulário: regressão prevendo sempre o mesmo valor,
  segmentação pintando tudo de uma classe, detecção sem achar nenhuma caixa
  ([ADR-101](docs/dev/DECISIONS.md)).
- **Aviso ao congelar pesos que nunca foram treinados.** Feature extraction sobre
  um modelo sem pesos pré-treinados congela ruído: num `unet` são 31,4 milhões de
  parâmetros aleatórios fixos e 195 treináveis
  ([ADR-101](docs/dev/DECISIONS.md)).


- **A seta dos seletores girava torta.** Era o caractere `▾` girando dentro da
  própria caixa de texto — e o centro da caixa de um glifo não é o centro do
  triângulo, então ele descrevia um arco em vez de girar no lugar. Virou um SVG
  centrado no próprio eixo, nos três lugares onde aparecia.
- **A explicação de um parâmetro não é mais cortada pela borda do painel.** Os
  cards do formulário usam `backdrop-filter`, que cria um contexto de
  empilhamento por card e prendia a caixa dentro dele. Ela agora é renderizada
  num portal, como os menus já eram, e é encaixada dentro da janela — um campo
  colado na margem mostra a explicação inteira.
- **O rótulo e a dica do determinístico deixaram de mentir** — o "(lento)" no
  painel de classificação e o "reprodutível, mais lento" nos quatro painéis de
  tarefa, depois que a medição do
  ADR-098 mostrou que ele é neutro ou mais rápido em treinos curtos.


## [0.8.0] — 2026-08-21

### Added

- **A GUI agora oferece continuar um treino que parou.** No histórico, o run
  ganha um selo `⏸ 2/4` e, no detalhe, um botão `▶ RETOMAR`. A configuração vem
  do `run.json` do próprio run, não do formulário: continuar com outros
  hiperparâmetros criaria uma pasta cujo histórico descreve dois experimentos
  diferentes. Sweeps e K-fold não aparecem — os treinos deles ficam em
  subpastas, e continuar a pasta-mãe não continuaria nada
  ([ADR-095](docs/dev/DECISIONS.md)).
- **Um treino interrompido pode continuar de onde parou, em qualquer tarefa.**
  O estado do otimizador e do scheduler passa a ser gravado a cada época num
  `resume.pt` ao lado do checkpoint — separado de propósito, porque o
  `best_model.pth` é carregado por cinco coisas que só querem os pesos. Retomar
  continua o mesmo run: uma curva contínua, um `run.json`. Vale para
  classificação, regressão, segmentação, o autoencoder de anomalia e o laço
  torchvision de detecção; no YOLO quem retoma é o próprio Ultralytics, que já
  guarda otimizador e EMA no `weights/last.pt` (as épocas já registradas voltam
  do `run.json` para a curva não começar no meio). O PatchCore fica de fora de
  propósito: não tem época para retomar
  ([ADR-092](docs/dev/DECISIONS.md), [ADR-093](docs/dev/DECISIONS.md)).

### Changed

- **Todo treino agora é reprodutível por padrão.** Dois runs da mesma config com
  a mesma seed davam 0.8263 e 0.7481 — `deterministic` era `False` por padrão, e
  o docstring justificava isso com "3-5× mais lento". Medimos: em runs de 2
  épocas numa RTX 5060 Ti, resnet18@224 empatou e resnet50@224 e resnet18@320
  ficaram **mais rápidos** determinísticos (-19% e -16%). O auto-tuning do cuDNN
  cobra adiantado e só se paga em treinos longos. Agora o padrão é `True` em
  todas as tarefas, e dois runs da mesma config saem idênticos até o último
  dígito. Importa porque o ruído entre execuções (0.117 entre seeds, 0.135 entre
  folds) era **do mesmo tamanho** dos efeitos medidos em ablações
  ([ADR-098](docs/dev/DECISIONS.md)).
- **O `num_workers` agora é limitado pela memória que a máquina tem.** Um worker
  não carrega o modelo — ele carrega imagens, enquanto o modelo vive na GPU — então
  o que ele custa é *commit* de memória, ~1 GB por worker por pool de loader. O
  projeto agora lê esse orçamento e reduz o pedido quando não cabe, avisando. Na
  máquina onde o WinError 1455 aconteceu (10.7 GB livres, 3 pools, 8 pedidos) o
  cálculo dá 1 worker, e o treino teria rodado ([ADR-098](docs/dev/DECISIONS.md)).

- **O CI voltou a rodar antes da promoção.** O gatilho de push apontava para um
  branch `develop` que não existe neste repositório, então as verificações só
  rodavam quando a `main` era promovida — depois do ponto em que achar um
  problema é barato. Agora roda em `development` também.
- **O CI agora roda o self-test inteiro.** As seis tarefas × cinco estratégias
  (27 casos, cada um um treino real através de um servidor de verdade) só rodavam
  quando alguém digitava `visionforge selftest`. Agora rodam a cada push — é a
  única verificação que exercita API, SSE, trainers e `run.json` juntos, que é
  onde os últimos defeitos moravam ([ADR-097](docs/dev/DECISIONS.md)).
- **O CI agora treina uma época de YOLO de verdade.** O backend padrão de
  detecção só era exercitado com um `YOLO` simulado — foi assim que uma época
  fantasma passou meses despercebida. Um job `detection` instala o extra e treina
  em cada pull request; sem checkpoint local ele monta o `yolo11n.yaml` do zero,
  então não depende de download ([ADR-097](docs/dev/DECISIONS.md)).
- **O `eslint` virou porta no CI.** Ele estava fora do gate porque seis erros
  antigos deixariam o job vermelho no dia em que entrasse. Os seis foram
  corrigidos — inclusive o histórico, que ficava montado enquanto fechado e
  refazia à mão doze `setState` de reset que uma montagem nova já dá de graça
  ([ADR-096](docs/dev/DECISIONS.md)).

### Fixed

- **O F1 da anomalia era sempre zero no autoencoder.** Um modelo com AUROC 0.79
  reportava F1 0.00 (intervalo bootstrap [0.0, 0.0] em 1000 reamostragens). O
  limiar de decisão vinha do percentil dos scores do **loader de treino**, que
  aplica augmentation: rotação preenche cantos e flip move estrutura, os dois
  aumentam o erro de reconstrução. O percentil ficava acima de qualquer score que
  o modelo produz em imagem limpa, e nada era marcado como anomalia. Agora o
  limiar vem das mesmas imagens de treino lidas como a inferência as lê —
  retreinando nos mesmos dados: limiar 0.0323 → 0.0150 e **F1 0.0000 → 0.3373**
  ([ADR-098](docs/dev/DECISIONS.md)).
- **A curva ROC não desenhava a macro-média** que seu próprio docstring prometia
  — justamente o número que o `test_auc_roc` reporta — e o eixo de épocas da
  curva de loss marcava 1.5 e 2.5, que não existem
  ([ADR-098](docs/dev/DECISIONS.md)).

- **O histórico mostrava as métricas erradas para três das cinco tarefas.**
  Segmentação e anomalia apareciam com a célula de métricas **vazia**, e
  regressão mostrava a loss em vez do R² — o mapa de métricas do histórico só
  tinha entradas para classificação e detecção, e as outras caíam nas chaves da
  classificação, que elas nunca escrevem. Os números sempre estiveram no
  `run.json`; a lista é que não sabia procurá-los. Pelo mesmo motivo, toda
  tarefa standalone aparecia arquivada no bloco "classification", justamente no
  filtro que existe para separá-las.
- **Todo treino YOLO registrava uma época fantasma.** O Ultralytics dispara o
  mesmo callback de fim de época mais uma vez depois do laço, para anotar as
  métricas do `best.pt` — e essa passada era gravada como época. Um treino de 10
  épocas dizia 11, um de 100 dizia 101; e quando os números dessa passada
  empatavam ou superavam os da melhor época, o `best_epoch` passava a apontar
  para uma época que nunca rodou (dois runs aqui dizem `best_epoch=6` num treino
  de 5 épocas). Achado por um teste que treina de verdade, e não com o YOLO
  simulado ([ADR-097](docs/dev/DECISIONS.md)).
- **Um `useContext` chamado depois de um `return` antecipado.** No bloco de
  scheduler do painel de parâmetros, o hook só rodava em parte dos renders — na
  primeira vez que aquele schema viesse sem propriedades, a ordem dos hooks
  saía do lugar. Nunca disparou, mas era questão de tempo
  ([ADR-096](docs/dev/DECISIONS.md)).
- **O botão de parar um treino agora para o treino.** A fila criava o sinal de
  cancelamento e o marcava quando você pedia para parar, mas nada entregava esse
  sinal ao trainer: o endpoint respondia 200, o log dizia "vai parar na próxima
  época" e o treino ia até o fim. O sinal agora percorre o mesmo caminho do
  progresso — a rota entrega ao bloco, o bloco entrega ao trainer — e uma busca
  de hiperparâmetros cancelada para entre os trials em vez de começar o próximo
  ([ADR-094](docs/dev/DECISIONS.md)).
- **Retomar não deixa mais uma pasta de run vazia para trás.** O treino criava a
  pasta nova com data e hora, apontava o TensorBoard para ela e só depois
  descobria que tinha estado para continuar — abandonando a pasta e mandando os
  eventos do run retomado para onde ninguém ia olhar. A pasta agora é decidida
  antes de qualquer coisa ser criada ([ADR-093](docs/dev/DECISIONS.md)).
- **O Grad-CAM voltou a mostrar se o modelo acertou.** Ele recuperava os nomes
  das classes relendo a pasta de treino do run, porque o `run.json` nunca os
  guardava — então renomear um dataset fazia a verdade sumir em silêncio, e a
  sobreposição caía para índices sem dizer por quê. O mapeamento agora é gravado
  no `run.json`, que é onde deveria estar desde o início: ele é propriedade do
  checkpoint treinado, não de onde os dados moram
  ([ADR-091](docs/dev/DECISIONS.md)).

## [0.7.0] — 2026-08-11

### Added

- **Uma introdução na primeira execução.** A tela escurece, "Bem-vindo" entra e
  sai, e a interface pergunta o seu nome. Nas visitas seguintes só o
  cumprimento `Bem-vindo, <nome>` por ~2s — nunca pergunta de novo. O nome fica
  no header e trocar é um clique nele. Guardado localmente, porque é preferência
  de quem está na máquina, não estado do servidor
  ([ADR-090](docs/dev/DECISIONS.md)).

- **O treino avisa quando termina.** O título da aba passa a `✓ <run> —
  VisionForge` (ou `✗` se falhou), e uma notificação do navegador aparece quando
  a página está em segundo plano. A permissão é pedida ao iniciar um treino, não
  ao abrir a página — prompt antes de você ter feito qualquer coisa é o que se
  nega por reflexo, e a negativa gruda ([ADR-089](docs/dev/DECISIONS.md)).

- **Dá para parar um treino em andamento**, e ele mantém o que já conquistou —
  melhor checkpoint, métricas, gráficos e `run.json`, com `total_epochs`
  registrando até onde foi. A parada acontece na virada de época, que é onde o
  checkpoint já está escrito. Antes a única saída era matar o servidor, o que
  derrubava a fila inteira junto ([ADR-088](docs/dev/DECISIONS.md)).

## [0.6.0] — 2026-08-08

### Added

- **Detecção passa a reportar intervalo de confiança no mAP.** Era a única das
  cinco tarefas sem — e por um motivo real: mAP ordena todas as detecções do
  conjunto por confiança e percorre a curva precisão/recall, então é propriedade
  do *conjunto*, não média de números por imagem. Não há acumulador para somar,
  e a métrica é recomputada a cada reamostragem. Um sorteio que perde uma classe
  inteira é descartado em vez de entrar na média, e o `n_resamples` reportado
  conta só os sobreviventes ([ADR-087](docs/dev/DECISIONS.md)).

- **O painel de detecção mostra exemplos das classes e o layout detectado.**
  As caixas anotadas viram miniaturas recortadas, uma por classe — contagem não
  diz se o rótulo está no lugar certo, e a imagem inteira não diz a qual caixa
  ele se refere. Cada split também informa qual das duas convenções YOLO foi
  resolvida (`train/images` ou `images/train`), que é como um dataset convertido
  pela metade aparece antes de ocupar a GPU.

## [0.5.1] — 2026-08-06

## [0.5.0] — 2026-08-05

### Added

- **Cada hiperparâmetro tem uma linha explicando o que ele faz**, e os painéis
  passam a separar o básico (épocas, batch, learning rate, seed) do avançado,
  que começa recolhido. Nenhum parâmetro saiu da tela — o corte é por frequência
  de ajuste, não por importância, e um valor avançado fora do padrão abre a
  seção sozinho ([ADR-085](docs/dev/DECISIONS.md)).

- **Detecção aceita filtros de pré-processamento.** A Ultralytics é dona do
  próprio pipeline de dados, então os filtros são aplicados uma vez numa cópia
  temporária que o `data.yaml` passa a apontar — o que também é ~30x menos CPU
  que filtrar por imagem a cada época. A cópia é chaveada por conteúdo (um sweep
  de 20 trials materializa uma vez), removida ao fim mesmo quando o treino
  falha, e o `run.json` continua registrando o dataset **original**
  ([ADR-084](docs/dev/DECISIONS.md)).

## [0.4.0] — 2026-08-05

### Added

- **`/api/health`** informa a versão e o bundle da SPA com que o processo subiu.
  Serve para diagnosticar o caso em que um `visionforge gui` deixado aberto
  durante uma alteração passa a servir JavaScript novo com backend velho — os
  estáticos são lidos do disco a cada requisição, mas os módulos Python só na
  partida ([ADR-086](docs/dev/DECISIONS.md)).

- **Uma chave liga e desliga a data augmentation**, em todas as cinco tarefas.
  Desligada, os parâmetros somem da tela e o `run.json` registra o estado — em
  vez de você ter que zerar cada campo à mão, que em detecção são 15. Os valores
  ficam guardados, então religar devolve o ajuste. Normalização e tamanho da
  imagem saíram da seção: não são augmentation, valem para treino, validação e
  teste ([ADR-083](docs/dev/DECISIONS.md)).

- **O histórico mostra em qual dataset cada run foi treinado** — selo no card,
  caminho e fingerprint no detalhe do run, e um veredito "mesmos dados?" ao
  comparar runs. Nada novo é medido: os dados já estavam no `run.json`. O nome
  aparece em 69 dos 78 runs existentes porque cai para `config.data.base_dir`;
  a verificação por hash só vale de 26/07/2026 em diante, e o comparador diz
  isso em vez de adivinhar ([ADR-082](docs/dev/DECISIONS.md)).

## [0.3.1] — 2026-08-05

### Fixed

- **`WinError 1455: o arquivo de paginação é muito pequeno` while starting a
  run** now says what it means. Every DataLoader worker is a separate process
  that re-imports torch and its CUDA DLLs — on Windows the start method is
  spawn, not fork — so the cost is roughly a gigabyte per worker, per loader.
  Windows reported the shortfall against whichever DLL happened to be loading,
  which pointed at torch and hid the cause. The GUI now explains it and names
  `data.num_workers` ([ADR-081](docs/dev/DECISIONS.md)).

### Changed

- **Data modules own their loaders and shut the worker pools down.** Each call
  to `train_loader()` built a *new* DataLoader, so a second call meant a second
  set of worker processes for the same split; and nothing ever stopped them
  explicitly. Splits are now built once and every block closes its data module
  in a `finally`, so a run that raises does not leave its workers behind. The
  test split also no longer asks for `persistent_workers` — it is read once —
  which drops the peak worker count for a classification run from 12 to 8
  ([ADR-081](docs/dev/DECISIONS.md)).

## [0.3.0] — 2026-08-04

### Changed

- **"Testar modelo" takes one labelled folder** instead of a dataset root plus
  three split names, and dispatches per task — so regression, segmentation and
  anomaly can be tested at all, where they previously answered with a raw
  `Input should be 'binary' or 'multiclass'` and only classification and
  detection ever worked. The folder is given in the label shape the run was
  trained with; regression points at its `.csv` manifest, since its data model
  has manifests rather than split folders
  ([ADR-080](docs/dev/DECISIONS.md)).

### Fixed

- **`visionforge doctor` recommended the CPU wheel on a CUDA machine.** Driver
  6xx renamed the `nvidia-smi` header field to `CUDA UMD Version`, which the
  parser did not match, so a recent driver looked like no GPU at all — an
  RTX 5060 Ti on driver 610.74 was told to `pip install "visionforge-studio[cpu]"`.
  Both spellings are accepted now.
- **The documented frontend type-check was a no-op.** The root `tsconfig.json`
  is a solution file with `"files": []`, so `npx tsc --noEmit` type-checked
  nothing and always passed — which is how a `base_dir` reference that no longer
  existed survived a green check. The command is now `npm run typecheck`
  (`tsc -b`), and CI gained a `frontend` job running vitest and the SPA build;
  neither had ever run there.
- **Esc in the run history could desynchronise from the backdrop.** Its handler
  read the step-back function from a ref written during render; it is now a
  `useCallback` the effect depends on directly.
- **Detection runs produced no plots and no checkpoint.** Ultralytics resolves a
  relative `project` path under its own `runs_dir`, so every artifact landed in
  `runs/detect/outputs/...` and the real run directory held only `data.yaml` and
  `run.json` — which is why every post-training action on a detection run
  reported a missing `best.pt`. The path is now absolute, and the run collects
  both confusion matrices, all four Box curves and the validation prediction
  sample ([ADR-079](docs/dev/DECISIONS.md)).
- **Custom-task runs showed no plots** although the engine was drawing one:
  `run.json` hardcoded `graphics: []`, orphaning the primary-metric curve on
  disk. It is now declared, alongside a train-loss curve
  ([ADR-079](docs/dev/DECISIONS.md)).

## [0.2.0] — 2026-08-03

> Tagged locally and superseded before it was published; 0.3.0 is the first
> release after 0.1.0 to reach PyPI. The entries below shipped as part of it.

### Added

- **Bootstrap confidence intervals on a single run's test metrics.** Every
  classification run reports `0.8734 [0.8412, 0.9021]` instead of a bare number,
  written to `run.json` as `metric_cis` and shown in the result tiles, the
  run-detail panel and the model card. Always on — the metrics are recomputed
  with vectorized arithmetic (700x faster than one sklearn call per resample,
  pinned against sklearn to 1e-16), so it costs ~0.02 s and needs no knob
  ([ADR-074](docs/dev/DECISIONS.md)).
- **The same intervals for regression, segmentation and anomaly** — MSE/RMSE/MAE/R²,
  mIoU/Dice/pixel-accuracy, AUROC/F1. The image is always the resampling unit,
  never the smaller thing inside it: segmentation sums per-image confusion
  matrices rather than resampling pixels, which would report an interval far
  tighter than the evidence supports ([ADR-076](docs/dev/DECISIONS.md)).
- **A training queue.** A second submission no longer gets a 409 — it lines up and
  starts on its own when the GPU frees, so an evening's experiments can be
  submitted and left. `GET /api/queue` lists what is waiting, a not-yet-started
  job can be dropped, and the bottom bar shows `⧗ fila N` with a panel
  ([ADR-075](docs/dev/DECISIONS.md)).
- **Test-set diagnostics for every task, not just classification**: predicted-vs-actual
  scatter and residual histogram (regression), per-class IoU and confusion matrix
  (segmentation), score histogram with the decision threshold (anomaly)
  ([ADR-077](docs/dev/DECISIONS.md)).
- **Grad-CAM shows the true class next to the predicted one**, with wrong
  predictions outlined in red. Class names are recovered from the training
  folder and ground truth from each image's parent folder — never guessed: a
  count mismatch shows the index, and an unlabeled folder shows nothing
  ([ADR-077](docs/dev/DECISIONS.md)).

### Fixed

- **The markdown model card returned 500 for every task except classification.**
  It hardcoded the classification epoch columns and formatted each cell with
  `:.4f`, so the `"?"` fallback for a missing key always raised. Columns now come
  from the run's own history ([ADR-077](docs/dev/DECISIONS.md)).
- **Per-run actions crashed on a researcher-defined task's run.** `test`,
  `gradcam`, `batch_predict` and `export_onnx` read `config.task`, which a custom
  run does not have, fell through to the classification path and died rebuilding
  a ResNet. They now answer 400 naming the task
  ([ADR-077](docs/dev/DECISIONS.md)).
- **A custom task whose config used `Literal`, `Path` or any non-builtin
  annotation failed to load** with a confusing Pydantic "is not fully defined"
  error pointing at the user's file. Task modules are now registered in
  `sys.modules` before execution, which is where Pydantic resolves the
  stringified annotations the scaffold generates.
- The install docs recommended `cu121` and never listed `cu128`, walking anyone
  with an RTX 50-series card into a build that imports fine and fails at the
  first kernel launch. The `docker build` example also carried a literal `\n`
  instead of a line continuation.
- **A grid axis could not reach a scheduler's dependent parameters.** Putting
  `step` on the axis while the scalar control still said `none` left `step_size`
  and `gamma` unrendered, so the sweep ran them on defaults with no way to see or
  sweep them. The form now shows the union over the scalar kind and every kind on
  the axis ([ADR-078](docs/dev/DECISIONS.md)).
- **The image viewer covered the plot it was showing.** The caption and close
  button floated over the figure, hiding the strip matplotlib uses for the
  x-axis and legend. They now occupy their own rows, so nothing is covered and
  nothing is cut ([ADR-078](docs/dev/DECISIONS.md)).
- **An unknown preprocessing filter returned 500.** Typing `blur` — the natural
  guess for `gaussian_blur` or `median_blur` — produced a server fault with no
  hint. It is now a 422 that lists every registered filter
  ([ADR-078](docs/dev/DECISIONS.md)).
- **`bump-my-version bump` could never complete**, so the documented one-command
  release (ADR-073) had never once worked: the pytest pre-commit hook ran
  `uv run`, which re-locked because the version had just changed, and pre-commit
  aborted on the modified `uv.lock`. The hook now runs `--frozen`.

### Changed

- **Clicking outside a dialog steps back one level** instead of dismissing the
  whole stack: plot → run detail → history list → closed, with Esc mirroring it.
  The header's × still closes everything at once
  ([ADR-078](docs/dev/DECISIONS.md)).
- A busy server **queues** a submission instead of refusing it with 409. The
  per-endpoint validation errors (422 for a bad config, 404 for an unknown custom
  task) are unchanged and still happen before anything is enqueued
  ([ADR-075](docs/dev/DECISIONS.md)).

## [0.1.0] — 2026-07-29

First public release. Everything below already shipped on `main`; this is the
point at which it becomes a version other people can install and cite.

> **Installed from PyPI as `visionforge-studio`.** The bare `visionforge` name
> is an unrelated project by another author. The import name, the `visionforge`
> CLI command and the project itself are unchanged.

### Tasks

- **Classification** — ResNet 18/34/50/101, EfficientNet B1/B7, VGG 16/19,
  AlexNet, plus any `timm` backbone (ADR-051) or a drop-in custom model
  (ADR-048/049). Binary, multiclass and multilabel.
- **Object detection** — Ultralytics YOLOv8/9/10/11/12/26 and RT-DETR, plus a
  torchvision backend (Faster R-CNN, SSD, RetinaNet) with its own loop
  (ADR-035). Full Ultralytics hyperparameter surface (ADR-040).
- **Image regression** — CSV-manifest datasets, CNN backbone + linear head,
  MSE/RMSE/MAE/R².
- **Semantic segmentation** — DeepLabV3, FCN, LR-ASPP and a hand-rolled U-Net;
  mean IoU, Dice, pixel accuracy, with `ignore_index` respected everywhere.
- **Anomaly detection** — convolutional autoencoder and PatchCore, image-level
  AUROC on an MVTec-style layout.
- **Your own task** — define a whole new task family in one documented Python
  file (`visionforge new-task`), with sweeps, replicates and the live monitor
  for free (ADR-058).

### Strategies

- Single run, K-fold cross-validation (classification, regression,
  segmentation), grid / random / Optuna-TPE sweeps (ADR-052), transfer learning
  (ADR-046/047), model comparison, and multi-seed replicates reporting
  `mean ± 95% CI` (ADR-056).
- **Paired significance testing** (ADR-061) — compares N configurations over
  the *same* seeds, picks and justifies a paired t-test or Wilcoxon, reports
  Cohen's `d_z`, a bootstrap CI of the difference and Holm-Bonferroni
  correction. It refuses to compare runs whose seeds do not line up, and flags
  when the seed count makes significance unreachable, so "not significant" is
  never read as "no effect".
- **Paper-ready output** — every advanced report is also written as a
  `booktabs` LaTeX table with notes stating what each interval covers.

### Reproducibility

- Versioned `run.json` for every run (ADR-013) carrying the full config, seed,
  per-epoch history, `environment` (Python, torch, CUDA, cuDNN, GPU model —
  ADR-057) and a `dataset_fingerprint` (ADR-061), so "same data" is checkable
  rather than assumed.
- `training.deterministic` in **every** task and in the custom-task SDK
  (ADR-062). Detection defaults to `True` to mirror `YOLO.train`; the rest
  default to `False` because pinning cuDNN costs throughput.
- Config `schema_version` with migrations (ADR-039); YAML round-trips between
  the GUI and the CLI.

### Interface

- React SPA served by the same Python process — no separate frontend to run.
- Canonical panel layout across all tasks (ADR-059): experiment name, YAML
  export/import, strategy selector, model, training, dataset stats,
  preprocessing filters and augmentation with live preview.
- History grouped by task family with wrapping filters, multi-select delete and
  run comparison (ADR-063/064); every dropdown is drawn by the app, so no
  operating-system popup breaks the dark theme.
- Datasets is its own surface: one-shot download from torchvision, Roboflow,
  Kaggle or Hugging Face (ADR-055).
- Post-training: per-checkpoint testing on new data, batch prediction to CSV,
  Grad-CAM, ONNX export with a PyTorch-vs-runtime latency benchmark, and
  TensorBoard scalars per run (ADR-054).

### Verification

- `visionforge doctor` — detects GPU/CUDA and prints the exact torch install
  line for the machine (ADR-042).
- `visionforge selftest` — trains every task through the real API on synthetic
  data and asserts the run, the report shape and the live-progress contract
  (ADR-060). Offline, ~90 s, CI-ready.
- A full matrix on **real** datasets (ADR-065) is recorded in
  [`docs/dev/VALIDATION.md`](docs/dev/VALIDATION.md): 21 cases across
  five tasks and five strategies, all passing.
- 1274 backend tests, 102 frontend tests, ruff + mypy clean, gated in CI.

### Fixed in the run-up to this release

- Transfer learning trained correctly but streamed no live progress, leaving
  the GUI's progress bar dead for the whole run (ADR-065).
- Multi-trial strategies (K-fold, sweeps, replicates) emitted no progress
  events, so the bar advanced on wall-clock only.
- Opening the history after a K-fold returned a 500: cross-validation wrote
  timezone-aware timestamps while every other writer used naive local time.
- The preprocessing preview served a stale "final" image from the browser
  cache, one pipeline behind.
- The torchvision detection backend never seeded, so `seed: 42` was a claim
  nothing backed (ADR-062).
- Sweeps and replicates silently accepted a metric no trial reported, ranking
  every trial 0.0 and crowning an arbitrary winner (ADR-060).
- Replicated comparison ranked descending regardless of metric direction, so a
  MAE of 4.02 beat a MAE of 0.99 (ADR-061).

[Unreleased]: https://github.com/marcus-vreis/VisionForge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/marcus-vreis/VisionForge/releases/tag/v0.1.0
