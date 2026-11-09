# Backlog do VisionForge

Gerado em 2026-11-09 a partir da auditoria geral ([AUDIT-2026-11-09.md](AUDIT-2026-11-09.md)),
dos achados dos revisores, do backlog anterior e das issues #67, #86 e #118. Cada task foi
conferida no código antes de entrar aqui e tem uma issue com problema, evidência, proposta,
critério de aceite, arquivos e dependências. As issues são a fonte da verdade; este arquivo é
o mapa.

**78 tasks** em 78 PRs pequenos. Prioridade: **P0** afeta números
publicáveis, privacidade ou segurança; **P1** próximas semanas; **P2** dívida e polimento.
Esforço: S até meio dia, M um a dois dias, L mais que isso. ✋ = precisa de decisão do mantenedor.

## Ordem de execução

A ordem põe primeiro o que muda números e o que fecha uma porta de segurança com pouco código,
depois documentação e produto. PRs sem arquivos em comum podem correr em paralelo.

| # | Issue | Task | Prioridade | Esforço | Branch | Título |
|---|---|---|---|---|---|---|
| 1 | [#154](https://github.com/marcus-vreis/VisionForge/issues/154) | STAT-1 | P0 | M | `fix/stats-augment-wilcoxon` | fix(stats): augment, transforms do K-fold, grupo da cabeça no fine-tuning e piso do Wilcoxon com empates (ADR-177) |
| 2 | [#155](https://github.com/marcus-vreis/VisionForge/issues/155) | SEC-3 | P0 | S | `fix/credentials-422-echo` | fix(gui): POST /api/credentials não devolve a chave no 422 |
| 3 | [#156](https://github.com/marcus-vreis/VisionForge/issues/156) | SEC-1 ✋ | P0 | M | `fix/gui-host-origin-guard` | fix(gui): rejeitar Host e Origin não-loopback em todas as rotas (DNS rebinding) |
| 4 | [#157](https://github.com/marcus-vreis/VisionForge/issues/157) | SEC-2 ✋ | P0 | M | `fix/gui-security-headers` | fix(gui): CSP e cabeçalhos de segurança no SPA, sem preconnect ao Google Fonts, /docs só sob flag |
| 5 | [#158](https://github.com/marcus-vreis/VisionForge/issues/158) | SEC-6 | P1 | S | `fix/third-party-telemetry-optout` | fix(privacy): desligar telemetria de terceiros (ultralytics, Hugging Face, MLflow) num módulo único |
| 6 | [#159](https://github.com/marcus-vreis/VisionForge/issues/159) | SEC-7 | P1 | S | `fix/detection-onnx-no-install` | fix(detection): exportar ONNX sem deixar o ultralytics instalar pacotes |
| 7 | [#160](https://github.com/marcus-vreis/VisionForge/issues/160) | SEC-4 | P0 | S | `chore/deps-security-bump` | chore(deps): atualizar pillow, onnx, aiohttp, filelock, urllib3 e starlette no uv.lock |
| 8 | [#161](https://github.com/marcus-vreis/VisionForge/issues/161) | SEC-5 | P1 | S | `fix/frontend-devdeps` | fix(frontend): mover shadcn, tailwindcss e tipos para devDependencies e remover fontes não usadas |
| 9 | [#162](https://github.com/marcus-vreis/VisionForge/issues/162) | STAT-3 ✋ | P1 | M | `fix/stats-ci-dual` | fix(significance): intervalo da diferença pareada coerente com o teste (t ou Hodges-Lehmann) em vez do bootstrap percentil |
| 10 | [#163](https://github.com/marcus-vreis/VisionForge/issues/163) | STAT-2 | P1 | S | `chore/augment-parity-tests` | test(data): paridade de augment=False em todos os caminhos de treino |
| 11 | [#164](https://github.com/marcus-vreis/VisionForge/issues/164) | SEC-8 ✋ | P1 | M | `fix/bundle-public-weights` | fix(bundle): bundle público sem caminhos absolutos dentro de weights/*.pt e config.* |
| 12 | [#165](https://github.com/marcus-vreis/VisionForge/issues/165) | SEC-9 | P1 | M | `fix/hf-dataset-revision` | fix(datasets): fixar a revision do load_dataset do Hugging Face e registrá-la |
| 13 | [#166](https://github.com/marcus-vreis/VisionForge/issues/166) | PAP-3 ✋ | P1 | M | `fix/seed-variance-public-runs` | fix(seed-variance): tornar summary.json regenerável com os run.json públicos do estudo |
| 14 | [#167](https://github.com/marcus-vreis/VisionForge/issues/167) | PAP-13 | P2 | S | `fix/benchmark-order` | fix(workflow-benchmark): alternar a ordem nativo/VisionForge entre os pares |
| 15 | [#168](https://github.com/marcus-vreis/VisionForge/issues/168) | STAT-4 | P2 | S | `fix/metric-direction-vocabulary` | fix(significance): direção e domínio das métricas por vocabulário único (mape, msle, nll, bce, huber...) |
| 16 | [#169](https://github.com/marcus-vreis/VisionForge/issues/169) | STAT-5 | P2 | S | `fix/non-finite-metric-stop` | fix(training): métrica de validação não finita interrompe o treino com aviso de saúde claro |
| 17 | [#170](https://github.com/marcus-vreis/VisionForge/issues/170) | STAT-6 ✋ | P2 | M | `fix/anomaly-single-class-auroc` | fix(anomaly): AUROC indefinido com uma só classe no teste vira null e aviso, não 0.5 |
| 18 | [#171](https://github.com/marcus-vreis/VisionForge/issues/171) | DOC-1 ✋ | P0 | S | `docs/no-cloud-claims` | docs(readme): tirar "no cloud" e "sem nuvem" das promessas públicas |
| 19 | [#172](https://github.com/marcus-vreis/VisionForge/issues/172) | OPS-3 | P1 | S | `chore/gitignore-codex` | chore(repo): ignorar .codex/ no .gitignore |
| 20 | [#173](https://github.com/marcus-vreis/VisionForge/issues/173) | OPS-4 ✋ | P1 | S | `chore/pytest-pre-push` | chore(dev): mover o hook pytest do pre-commit para pre-push |
| 21 | [#174](https://github.com/marcus-vreis/VisionForge/issues/174) | OPS-1 ✋ | P0 | S | `chore/repo-settings` | chore(repo): ativar proteção de branches e metadados no GitHub |
| 22 | [#175](https://github.com/marcus-vreis/VisionForge/issues/175) | OPS-2 ✋ | P0 | S | `chore/dependabot` | chore(deps): adicionar dependabot.yml e ligar secret scanning e security updates |
| 23 | [#176](https://github.com/marcus-vreis/VisionForge/issues/176) | SEC-10 ✋ | P1 | M | `docs/third-party-notices` | docs(legal): THIRD_PARTY_NOTICES (AGPL, LGPL, OFL) e seção Rede e privacidade |
| 24 | [#177](https://github.com/marcus-vreis/VisionForge/issues/177) | PAP-1 ✋ | P0 | S | `docs/paper-overleaf-sync` | docs(paper): trazer a versão editada no Overleaf de volta para paper/paper.md |
| 25 | [#178](https://github.com/marcus-vreis/VisionForge/issues/178) | PAP-5 | P1 | S | `docs/paper-facts` | docs(paper): corrigir K-fold, saídas da máquina, AGPL e contagem de testes no Software design |
| 26 | [#179](https://github.com/marcus-vreis/VisionForge/issues/179) | PAP-4 | P1 | M | `docs/paper-h3-reversal` | docs(paper): relatar a H3 rejeitada, definir a reversão e o contraste Holm exploratório |
| 27 | [#180](https://github.com/marcus-vreis/VisionForge/issues/180) | PAP-6 ✋ | P1 | S | `docs/paper-references` | docs(paper): completar paper.bib com Holm, datasets, arquiteturas e vizinhos de 2026, com teste de chaves |
| 28 | [#181](https://github.com/marcus-vreis/VisionForge/issues/181) | PAP-2 ✋ | P0 | L | `docs/paper-public-sources` | docs(paper): publicar as fontes da Research impact statement ou retirar os números |
| 29 | [#182](https://github.com/marcus-vreis/VisionForge/issues/182) | OPS-7 | P1 | S | `chore/coverage-gate` | ci(coverage): subir o gate de cobertura de 70 para ~90 com uma fonte única |
| 30 | [#183](https://github.com/marcus-vreis/VisionForge/issues/183) | OPS-5 ✋ | P1 | S | `docs/community-files` | docs(repo): SECURITY.md, CODE_OF_CONDUCT.md, PR template, CODEOWNERS e config de issues |
| 31 | [#184](https://github.com/marcus-vreis/VisionForge/issues/184) | DOC-2 | P1 | S | `docs/readme-pypi-links` | docs(readme): links absolutos para o PyPI, exemplo de run que funciona e limites corretos |
| 32 | [#185](https://github.com/marcus-vreis/VisionForge/issues/185) | OPS-6 | P1 | S | `docs/contributing-facts` | docs(contributing): corrigir afirmações falsas e documentar o modelo de branches |
| 33 | [#186](https://github.com/marcus-vreis/VisionForge/issues/186) | DOC-6 | P1 | M | `docs/architecture-map` | docs(architecture): atualizar mapa de módulos e política de extras, com teste contra nova defasagem |
| 34 | [#187](https://github.com/marcus-vreis/VisionForge/issues/187) | DOC-8 | P1 | S | `docs/tokens-guide` | docs(gui): seção "Tokens e chaves" dizendo onde ficam, como remover e o escopo mínimo |
| 35 | [#188](https://github.com/marcus-vreis/VisionForge/issues/188) | DOC-3 | P1 | M | `docs/site-release-0-21` | docs(site): publicar a 0.21.0 no site e travar a consistência de versão no Playwright |
| 36 | [#189](https://github.com/marcus-vreis/VisionForge/issues/189) | OPS-11 ✋ | P1 | M | `docs/release-checklist` | docs(release): checklist pós-release e script que mostra o diff de versão do site, INSTALL e capturas |
| 37 | [#190](https://github.com/marcus-vreis/VisionForge/issues/190) | GUI-1 | P1 | M | `feat/classification-replicates` | feat(gui): Réplicas na aba Classificação |
| 38 | [#191](https://github.com/marcus-vreis/VisionForge/issues/191) | GUI-4 | P1 | M | `feat/code-tour-regression` | feat(code-tour): aba Código por tarefa, começando pela regressão |
| 39 | [#192](https://github.com/marcus-vreis/VisionForge/issues/192) | GUI-5 | P1 | S | `feat/code-tour-segmentation` | feat(code-tour): código real da segmentação na aba Código |
| 40 | [#193](https://github.com/marcus-vreis/VisionForge/issues/193) | GUI-6 | P1 | S | `feat/code-tour-anomaly` | feat(code-tour): código real da anomalia (autoencoder e PatchCore) na aba Código |
| 41 | [#194](https://github.com/marcus-vreis/VisionForge/issues/194) | GUI-7 | P1 | S | `feat/code-tour-detection-torchvision` | feat(code-tour): código real da detecção torchvision (Faster R-CNN, SSD, RetinaNet) |
| 42 | [#195](https://github.com/marcus-vreis/VisionForge/issues/195) | DOC-9 | P1 | M | `docs/code-tab-colab-guides` | docs(gui): guias da aba Código (funções reais de treino) e do treino no Colab |
| 43 | [#196](https://github.com/marcus-vreis/VisionForge/issues/196) | DOC-4 | P1 | M | `docs/readme-features` | docs(readme): descrever Colab, Hub HF, abas Resultados/Código, explorador, audit-data/bundle e Docker; alinhar EN e PT |
| 44 | [#197](https://github.com/marcus-vreis/VisionForge/issues/197) | DOC-5 | P1 | M | `docs/guides-facts` | docs(guides): corrigir QUICKSTART, MODELS, TASKS, DATASETS e INSTALL para o que o produto faz hoje |
| 45 | [#198](https://github.com/marcus-vreis/VisionForge/issues/198) | GUI-9 | P1 | M | `feat/plotter-vector-figures` | feat(plotter): figuras em PDF/SVG e PNG a 300 dpi para artigo |
| 46 | [#199](https://github.com/marcus-vreis/VisionForge/issues/199) | GUI-10 | P1 | M | `feat/cli-strategies` | feat(cli): visionforge run com réplicas, comparação pareada, sweep e K-fold das tarefas standalone |
| 47 | [#200](https://github.com/marcus-vreis/VisionForge/issues/200) | GUI-3 | P1 | M | `feat/history-rerun` | feat(history): carregar a config de um run no formulário e repetir com N seeds |
| 48 | [#201](https://github.com/marcus-vreis/VisionForge/issues/201) | GUI-2 ✋ | P1 | L | `feat/paired-comparison-gui` | feat(gui): Comparação pareada entre variantes (replicated-comparison) em todas as tarefas |
| 49 | [#202](https://github.com/marcus-vreis/VisionForge/issues/202) | GUI-11 | P1 | M | `chore/server-lazy-imports` | perf(server): não importar torch, torchvision, ultralytics e matplotlib ao subir a GUI |
| 50 | [#203](https://github.com/marcus-vreis/VisionForge/issues/203) | GUI-12 | P1 | M | `chore/spa-code-splitting` | perf(frontend): code splitting do bundle de 1 MB por painel e overlay |
| 51 | [#204](https://github.com/marcus-vreis/VisionForge/issues/204) | OPS-8 | P1 | L | `chore/api-split-evaluation` | refactor(api): quebrar routes.py por domínio, começando pela avaliação e pela execução de experimento |
| 52 | [#205](https://github.com/marcus-vreis/VisionForge/issues/205) | OPS-9 | P1 | M | `chore/openapi-tags-responses` | refactor(api): tags por domínio e HTTPException documentadas no OpenAPI (python:S8415) |
| 53 | [#206](https://github.com/marcus-vreis/VisionForge/issues/206) | OPS-10 | P1 | M | `chore/sonar-test-smells` | chore(quality): fechar os smells de teste e o falso positivo S5332 com NOSONAR na linha certa |
| 54 | [#207](https://github.com/marcus-vreis/VisionForge/issues/207) | GUI-13 | P2 | S | `fix/a11y-skip-link` | fix(a11y): skip link, alvos de toque de 44 px e metas do index.html |
| 55 | [#208](https://github.com/marcus-vreis/VisionForge/issues/208) | GUI-15 ✋ | P2 | S | `fix/i18n-unexpected-errors` | fix(i18n): moldura traduzida nos erros 500 e 422 e fechamento do backlog do ADR-116 |
| 56 | [#209](https://github.com/marcus-vreis/VisionForge/issues/209) | OPS-19 | P2 | S | `fix/cli-doctor-encoding` | fix(cli): doctor nomeia CUDA_VISIBLE_DEVICES e --help sai em UTF-8 no console cp1252 |
| 57 | [#210](https://github.com/marcus-vreis/VisionForge/issues/210) | OPS-13 | P2 | M | `chore/test-wait-until` | test(helpers): trocar time.sleep de polling por wait_until com timeout e falha explícita |
| 58 | [#211](https://github.com/marcus-vreis/VisionForge/issues/211) | OPS-14 | P2 | S | `chore/clean-git-env-fixture` | test(env): fixture clean_git_env compartilhada e teste de que GIT_DIR herdado não vaza |
| 59 | [#212](https://github.com/marcus-vreis/VisionForge/issues/212) | STAT-7 ✋ | P2 | S | `docs/cv-epoch-selection` | docs(cv): declarar que a época de cada fold é escolhida no próprio fold de validação |
| 60 | [#213](https://github.com/marcus-vreis/VisionForge/issues/213) | DOC-14 | P2 | M | `docs/exporter-guides` | docs(gui): páginas curtas para os exportadores MLflow, W&B, Zenodo, GitHub e DVC |
| 61 | [#214](https://github.com/marcus-vreis/VisionForge/issues/214) | OPS-15 | P2 | S | `docs/adr-process` | docs(adr): documentar o processo de ADR e o template em DECISIONS.md |
| 62 | [#215](https://github.com/marcus-vreis/VisionForge/issues/215) | OPS-16 | P2 | S | `docs/frontend-readme` | docs(frontend): README do frontend, engines do Node e proxy do Vite configurável |
| 63 | [#216](https://github.com/marcus-vreis/VisionForge/issues/216) | OPS-18 ✋ | P2 | S | `docs/custom-api-stability` | docs(custom): declarar a estabilidade da API de user_tasks/user_models e travar a assinatura com teste |
| 64 | [#217](https://github.com/marcus-vreis/VisionForge/issues/217) | OPS-12 ✋ | P1 | S | `chore/close-118` | chore(tracking): fechar a issue #118 e abrir as continuações |
| 65 | [#218](https://github.com/marcus-vreis/VisionForge/issues/218) | OPS-17 ✋ | P2 | S | `docs/roadmap` | docs(roadmap): criar docs/ROADMAP.md e consertar links do roadmap arquivado |
| 66 | [#219](https://github.com/marcus-vreis/VisionForge/issues/219) | DOC-10 ✋ | P1 | M | `docs/site-landing` | docs(site): landing com Colab, registro e publicação, Resultados/Código, explorador, auditoria e seis etapas |
| 67 | [#220](https://github.com/marcus-vreis/VisionForge/issues/220) | DOC-13 | P2 | S | `feat/site-404-jsonld` | feat(site): página 404 própria e dados estruturados JSON-LD |
| 68 | [#221](https://github.com/marcus-vreis/VisionForge/issues/221) | DOC-15 ✋ | P2 | M | `docs/readme-images` | docs(images): recapturar as imagens do README com script reproduzível, sem versão impressa, incluindo a aba Código |
| 69 | [#222](https://github.com/marcus-vreis/VisionForge/issues/222) | DOC-11 ✋ | P2 | L | `docs/site-guides` | docs(site): guias /docs/colab, /docs/publicar e /docs/codigo com trechos de código reais |
| 70 | [#223](https://github.com/marcus-vreis/VisionForge/issues/223) | SEC-11 ✋ | P1 | M | `feat/ultralytics-optional` | feat(license): decidir o papel do ultralytics (AGPL) no pacote MIT |
| 71 | [#224](https://github.com/marcus-vreis/VisionForge/issues/224) | GUI-8 ✋ | P2 | M | `feat/code-tour-ultralytics` | feat(code-tour): mostrar o que o VisionForge passa ao Ultralytics e o laço instalado da biblioteca |
| 72 | [#225](https://github.com/marcus-vreis/VisionForge/issues/225) | SEC-13 | P2 | M | `chore/license-gate` | ci(license): gate de licenças copyleft que confere THIRD_PARTY_NOTICES contra o ambiente |
| 73 | [#226](https://github.com/marcus-vreis/VisionForge/issues/226) | SEC-12 ✋ | P2 | L | `feat/offline-guard` | feat(privacy): modo offline garantido com teste que prova que treinar não toca a rede |
| 74 | [#227](https://github.com/marcus-vreis/VisionForge/issues/227) | GUI-16 ✋ | P2 | M | `feat/device-mps` | feat(device): GPU da Apple (MPS) no treino |
| 75 | [#228](https://github.com/marcus-vreis/VisionForge/issues/228) | PAP-10 | P2 | S | `chore/paper-submission-guard` | ci(paper): bloquear placeholders e TODOs no commit de submissão |
| 76 | [#229](https://github.com/marcus-vreis/VisionForge/issues/229) | PAP-9 ✋ | P2 | S | `docs/usability-protocol` | docs(usability): registrar vieses e regras de coleta antes do primeiro participante |
| 77 | [#230](https://github.com/marcus-vreis/VisionForge/issues/230) | PAP-11 ✋ | P2 | M | `docs/preprint-todos` | docs(seed-variance): fechar os TODOs do preprint (literatura 2023-2026, DOI do estudo e data) |
| 78 | [#231](https://github.com/marcus-vreis/VisionForge/issues/231) | PAP-12 ✋ | P2 | S | `docs/paper-affiliation` | docs(paper): confirmar afiliação e apoio e unificar a identidade git |

## Por tema

### Estatística e dados (7)

Afeta números publicáveis: resultados que não seguem a config, intervalos e testes.

- [#154](https://github.com/marcus-vreis/VisionForge/issues/154) **STAT-1** · P0 · M — fix(stats): augment, transforms do K-fold, grupo da cabeça no fine-tuning e piso do Wilcoxon com empates (ADR-177).
- [#163](https://github.com/marcus-vreis/VisionForge/issues/163) **STAT-2** · P1 · S — test(data): paridade de augment=False em todos os caminhos de treino. Depende de STAT-1.
- [#162](https://github.com/marcus-vreis/VisionForge/issues/162) **STAT-3** · P1 · M ✋ — fix(significance): intervalo da diferença pareada coerente com o teste (t ou Hodges-Lehmann) em vez do bootstrap percentil. Depende de STAT-1.
- [#168](https://github.com/marcus-vreis/VisionForge/issues/168) **STAT-4** · P2 · S — fix(significance): direção e domínio das métricas por vocabulário único (mape, msle, nll, bce, huber...). Depende de STAT-3.
- [#169](https://github.com/marcus-vreis/VisionForge/issues/169) **STAT-5** · P2 · S — fix(training): métrica de validação não finita interrompe o treino com aviso de saúde claro.
- [#170](https://github.com/marcus-vreis/VisionForge/issues/170) **STAT-6** · P2 · M ✋ — fix(anomaly): AUROC indefinido com uma só classe no teste vira null e aviso, não 0.5.
- [#212](https://github.com/marcus-vreis/VisionForge/issues/212) **STAT-7** · P2 · S ✋ — docs(cv): declarar que a época de cada fold é escolhida no próprio fold de validação.

### Segurança, privacidade e licença (13)

Servidor local, dependências vulneráveis, telemetria de terceiros, AGPL.

- [#156](https://github.com/marcus-vreis/VisionForge/issues/156) **SEC-1** · P0 · M ✋ — fix(gui): rejeitar Host e Origin não-loopback em todas as rotas (DNS rebinding).
- [#157](https://github.com/marcus-vreis/VisionForge/issues/157) **SEC-2** · P0 · M ✋ — fix(gui): CSP e cabeçalhos de segurança no SPA, sem preconnect ao Google Fonts, /docs só sob flag. Depende de SEC-1.
- [#155](https://github.com/marcus-vreis/VisionForge/issues/155) **SEC-3** · P0 · S — fix(gui): POST /api/credentials não devolve a chave no 422.
- [#160](https://github.com/marcus-vreis/VisionForge/issues/160) **SEC-4** · P0 · S — chore(deps): atualizar pillow, onnx, aiohttp, filelock, urllib3 e starlette no uv.lock.
- [#176](https://github.com/marcus-vreis/VisionForge/issues/176) **SEC-10** · P1 · M ✋ — docs(legal): THIRD_PARTY_NOTICES (AGPL, LGPL, OFL) e seção Rede e privacidade. Depende de SEC-6, DOC-1.
- [#223](https://github.com/marcus-vreis/VisionForge/issues/223) **SEC-11** · P1 · M ✋ — feat(license): decidir o papel do ultralytics (AGPL) no pacote MIT. Depende de SEC-10.
- [#161](https://github.com/marcus-vreis/VisionForge/issues/161) **SEC-5** · P1 · S — fix(frontend): mover shadcn, tailwindcss e tipos para devDependencies e remover fontes não usadas.
- [#158](https://github.com/marcus-vreis/VisionForge/issues/158) **SEC-6** · P1 · S — fix(privacy): desligar telemetria de terceiros (ultralytics, Hugging Face, MLflow) num módulo único.
- [#159](https://github.com/marcus-vreis/VisionForge/issues/159) **SEC-7** · P1 · S — fix(detection): exportar ONNX sem deixar o ultralytics instalar pacotes.
- [#164](https://github.com/marcus-vreis/VisionForge/issues/164) **SEC-8** · P1 · M ✋ — fix(bundle): bundle público sem caminhos absolutos dentro de weights/*.pt e config.*.
- [#165](https://github.com/marcus-vreis/VisionForge/issues/165) **SEC-9** · P1 · M — fix(datasets): fixar a revision do load_dataset do Hugging Face e registrá-la.
- [#226](https://github.com/marcus-vreis/VisionForge/issues/226) **SEC-12** · P2 · L ✋ — feat(privacy): modo offline garantido com teste que prova que treinar não toca a rede. Depende de SEC-6, SEC-10.
- [#225](https://github.com/marcus-vreis/VisionForge/issues/225) **SEC-13** · P2 · M — ci(license): gate de licenças copyleft que confere THIRD_PARTY_NOTICES contra o ambiente. Depende de SEC-10.

### Produto e GUI (15)

Réplicas e comparação na GUI, aba Código por tarefa, CLI, desempenho, acessibilidade.

- [#190](https://github.com/marcus-vreis/VisionForge/issues/190) **GUI-1** · P1 · M — feat(gui): Réplicas na aba Classificação.
- [#199](https://github.com/marcus-vreis/VisionForge/issues/199) **GUI-10** · P1 · M — feat(cli): visionforge run com réplicas, comparação pareada, sweep e K-fold das tarefas standalone.
- [#202](https://github.com/marcus-vreis/VisionForge/issues/202) **GUI-11** · P1 · M — perf(server): não importar torch, torchvision, ultralytics e matplotlib ao subir a GUI.
- [#203](https://github.com/marcus-vreis/VisionForge/issues/203) **GUI-12** · P1 · M — perf(frontend): code splitting do bundle de 1 MB por painel e overlay. Depende de SEC-5.
- [#201](https://github.com/marcus-vreis/VisionForge/issues/201) **GUI-2** · P1 · L ✋ — feat(gui): Comparação pareada entre variantes (replicated-comparison) em todas as tarefas. Depende de GUI-1, STAT-3.
- [#200](https://github.com/marcus-vreis/VisionForge/issues/200) **GUI-3** · P1 · M — feat(history): carregar a config de um run no formulário e repetir com N seeds. Depende de GUI-1.
- [#191](https://github.com/marcus-vreis/VisionForge/issues/191) **GUI-4** · P1 · M — feat(code-tour): aba Código por tarefa, começando pela regressão.
- [#192](https://github.com/marcus-vreis/VisionForge/issues/192) **GUI-5** · P1 · S — feat(code-tour): código real da segmentação na aba Código. Depende de GUI-4.
- [#193](https://github.com/marcus-vreis/VisionForge/issues/193) **GUI-6** · P1 · S — feat(code-tour): código real da anomalia (autoencoder e PatchCore) na aba Código. Depende de GUI-4.
- [#194](https://github.com/marcus-vreis/VisionForge/issues/194) **GUI-7** · P1 · S — feat(code-tour): código real da detecção torchvision (Faster R-CNN, SSD, RetinaNet). Depende de GUI-4.
- [#198](https://github.com/marcus-vreis/VisionForge/issues/198) **GUI-9** · P1 · M — feat(plotter): figuras em PDF/SVG e PNG a 300 dpi para artigo.
- [#207](https://github.com/marcus-vreis/VisionForge/issues/207) **GUI-13** · P2 · S — fix(a11y): skip link, alvos de toque de 44 px e metas do index.html. Depende de SEC-2.
- [#208](https://github.com/marcus-vreis/VisionForge/issues/208) **GUI-15** · P2 · S ✋ — fix(i18n): moldura traduzida nos erros 500 e 422 e fechamento do backlog do ADR-116.
- [#227](https://github.com/marcus-vreis/VisionForge/issues/227) **GUI-16** · P2 · M ✋ — feat(device): GPU da Apple (MPS) no treino.
- [#224](https://github.com/marcus-vreis/VisionForge/issues/224) **GUI-8** · P2 · M ✋ — feat(code-tour): mostrar o que o VisionForge passa ao Ultralytics e o laço instalado da biblioteca. Depende de GUI-7, SEC-11.

### Documentação, README e site (13)

O que o README, os guias e o site prometem contra o que o produto faz.

- [#171](https://github.com/marcus-vreis/VisionForge/issues/171) **DOC-1** · P0 · S ✋ — docs(readme): tirar "no cloud" e "sem nuvem" das promessas públicas.
- [#219](https://github.com/marcus-vreis/VisionForge/issues/219) **DOC-10** · P1 · M ✋ — docs(site): landing com Colab, registro e publicação, Resultados/Código, explorador, auditoria e seis etapas. Depende de DOC-3.
- [#184](https://github.com/marcus-vreis/VisionForge/issues/184) **DOC-2** · P1 · S — docs(readme): links absolutos para o PyPI, exemplo de run que funciona e limites corretos. Depende de DOC-1.
- [#188](https://github.com/marcus-vreis/VisionForge/issues/188) **DOC-3** · P1 · M — docs(site): publicar a 0.21.0 no site e travar a consistência de versão no Playwright.
- [#196](https://github.com/marcus-vreis/VisionForge/issues/196) **DOC-4** · P1 · M — docs(readme): descrever Colab, Hub HF, abas Resultados/Código, explorador, audit-data/bundle e Docker; alinhar EN e PT. Depende de DOC-1, DOC-2.
- [#197](https://github.com/marcus-vreis/VisionForge/issues/197) **DOC-5** · P1 · M — docs(guides): corrigir QUICKSTART, MODELS, TASKS, DATASETS e INSTALL para o que o produto faz hoje. Depende de GUI-1.
- [#186](https://github.com/marcus-vreis/VisionForge/issues/186) **DOC-6** · P1 · M — docs(architecture): atualizar mapa de módulos e política de extras, com teste contra nova defasagem.
- [#187](https://github.com/marcus-vreis/VisionForge/issues/187) **DOC-8** · P1 · S — docs(gui): seção "Tokens e chaves" dizendo onde ficam, como remover e o escopo mínimo.
- [#195](https://github.com/marcus-vreis/VisionForge/issues/195) **DOC-9** · P1 · M — docs(gui): guias da aba Código (funções reais de treino) e do treino no Colab.
- [#222](https://github.com/marcus-vreis/VisionForge/issues/222) **DOC-11** · P2 · L ✋ — docs(site): guias /docs/colab, /docs/publicar e /docs/codigo com trechos de código reais. Depende de DOC-9, DOC-10, DOC-15.
- [#220](https://github.com/marcus-vreis/VisionForge/issues/220) **DOC-13** · P2 · S — feat(site): página 404 própria e dados estruturados JSON-LD. Depende de DOC-3.
- [#213](https://github.com/marcus-vreis/VisionForge/issues/213) **DOC-14** · P2 · M — docs(gui): páginas curtas para os exportadores MLflow, W&B, Zenodo, GitHub e DVC. Depende de DOC-8.
- [#221](https://github.com/marcus-vreis/VisionForge/issues/221) **DOC-15** · P2 · M ✋ — docs(images): recapturar as imagens do README com script reproduzível, sem versão impressa, incluindo a aba Código. Depende de DOC-4, GUI-4.

### Paper e pesquisa (11)

JOSS, estudo de seeds e estudo de usabilidade.

- [#177](https://github.com/marcus-vreis/VisionForge/issues/177) **PAP-1** · P0 · S ✋ — docs(paper): trazer a versão editada no Overleaf de volta para paper/paper.md.
- [#181](https://github.com/marcus-vreis/VisionForge/issues/181) **PAP-2** · P0 · L ✋ — docs(paper): publicar as fontes da Research impact statement ou retirar os números. Depende de PAP-1, PAP-13.
- [#166](https://github.com/marcus-vreis/VisionForge/issues/166) **PAP-3** · P1 · M ✋ — fix(seed-variance): tornar summary.json regenerável com os run.json públicos do estudo.
- [#179](https://github.com/marcus-vreis/VisionForge/issues/179) **PAP-4** · P1 · M — docs(paper): relatar a H3 rejeitada, definir a reversão e o contraste Holm exploratório. Depende de PAP-1, PAP-3.
- [#178](https://github.com/marcus-vreis/VisionForge/issues/178) **PAP-5** · P1 · S — docs(paper): corrigir K-fold, saídas da máquina, AGPL e contagem de testes no Software design. Depende de PAP-1.
- [#180](https://github.com/marcus-vreis/VisionForge/issues/180) **PAP-6** · P1 · S ✋ — docs(paper): completar paper.bib com Holm, datasets, arquiteturas e vizinhos de 2026, com teste de chaves. Depende de PAP-1.
- [#228](https://github.com/marcus-vreis/VisionForge/issues/228) **PAP-10** · P2 · S — ci(paper): bloquear placeholders e TODOs no commit de submissão.
- [#230](https://github.com/marcus-vreis/VisionForge/issues/230) **PAP-11** · P2 · M ✋ — docs(seed-variance): fechar os TODOs do preprint (literatura 2023-2026, DOI do estudo e data). Depende de PAP-3, PAP-6, PAP-10.
- [#231](https://github.com/marcus-vreis/VisionForge/issues/231) **PAP-12** · P2 · S ✋ — docs(paper): confirmar afiliação e apoio e unificar a identidade git. Depende de PAP-1.
- [#167](https://github.com/marcus-vreis/VisionForge/issues/167) **PAP-13** · P2 · S — fix(workflow-benchmark): alternar a ordem nativo/VisionForge entre os pares.
- [#229](https://github.com/marcus-vreis/VisionForge/issues/229) **PAP-9** · P2 · S ✋ — docs(usability): registrar vieses e regras de coleta antes do primeiro participante.

### Repositório, CI e contribuição (19)

Governança do GitHub, hooks, cobertura, refatoração da API, Sonar.

- [#174](https://github.com/marcus-vreis/VisionForge/issues/174) **OPS-1** · P0 · S ✋ — chore(repo): ativar proteção de branches e metadados no GitHub.
- [#175](https://github.com/marcus-vreis/VisionForge/issues/175) **OPS-2** · P0 · S ✋ — chore(deps): adicionar dependabot.yml e ligar secret scanning e security updates.
- [#206](https://github.com/marcus-vreis/VisionForge/issues/206) **OPS-10** · P1 · M — chore(quality): fechar os smells de teste e o falso positivo S5332 com NOSONAR na linha certa.
- [#189](https://github.com/marcus-vreis/VisionForge/issues/189) **OPS-11** · P1 · M ✋ — docs(release): checklist pós-release e script que mostra o diff de versão do site, INSTALL e capturas.
- [#217](https://github.com/marcus-vreis/VisionForge/issues/217) **OPS-12** · P1 · S ✋ — chore(tracking): fechar a issue #118 e abrir as continuações.
- [#172](https://github.com/marcus-vreis/VisionForge/issues/172) **OPS-3** · P1 · S — chore(repo): ignorar .codex/ no .gitignore.
- [#173](https://github.com/marcus-vreis/VisionForge/issues/173) **OPS-4** · P1 · S ✋ — chore(dev): mover o hook pytest do pre-commit para pre-push.
- [#183](https://github.com/marcus-vreis/VisionForge/issues/183) **OPS-5** · P1 · S ✋ — docs(repo): SECURITY.md, CODE_OF_CONDUCT.md, PR template, CODEOWNERS e config de issues.
- [#185](https://github.com/marcus-vreis/VisionForge/issues/185) **OPS-6** · P1 · S — docs(contributing): corrigir afirmações falsas e documentar o modelo de branches. Depende de OPS-4.
- [#182](https://github.com/marcus-vreis/VisionForge/issues/182) **OPS-7** · P1 · S — ci(coverage): subir o gate de cobertura de 70 para ~90 com uma fonte única.
- [#204](https://github.com/marcus-vreis/VisionForge/issues/204) **OPS-8** · P1 · L — refactor(api): quebrar routes.py por domínio, começando pela avaliação e pela execução de experimento. Depende de GUI-11.
- [#205](https://github.com/marcus-vreis/VisionForge/issues/205) **OPS-9** · P1 · M — refactor(api): tags por domínio e HTTPException documentadas no OpenAPI (python:S8415).
- [#210](https://github.com/marcus-vreis/VisionForge/issues/210) **OPS-13** · P2 · M — test(helpers): trocar time.sleep de polling por wait_until com timeout e falha explícita.
- [#211](https://github.com/marcus-vreis/VisionForge/issues/211) **OPS-14** · P2 · S — test(env): fixture clean_git_env compartilhada e teste de que GIT_DIR herdado não vaza.
- [#214](https://github.com/marcus-vreis/VisionForge/issues/214) **OPS-15** · P2 · S — docs(adr): documentar o processo de ADR e o template em DECISIONS.md. Depende de OPS-6.
- [#215](https://github.com/marcus-vreis/VisionForge/issues/215) **OPS-16** · P2 · S — docs(frontend): README do frontend, engines do Node e proxy do Vite configurável. Depende de SEC-5, OPS-6.
- [#218](https://github.com/marcus-vreis/VisionForge/issues/218) **OPS-17** · P2 · S ✋ — docs(roadmap): criar docs/ROADMAP.md e consertar links do roadmap arquivado. Depende de OPS-12.
- [#216](https://github.com/marcus-vreis/VisionForge/issues/216) **OPS-18** · P2 · S ✋ — docs(custom): declarar a estabilidade da API de user_tasks/user_models e travar a assinatura com teste. Depende de DOC-5.
- [#209](https://github.com/marcus-vreis/VisionForge/issues/209) **OPS-19** · P2 · S — fix(cli): doctor nomeia CUDA_VISIBLE_DEVICES e --help sai em UTF-8 no console cp1252.

## Decisões do mantenedor

Estas tasks esperam uma escolha que muda produto, licença, conta do GitHub ou o paper.
Onde há um default proposto, ele está na issue.

- STAT-3: método do IC da diferença pareada (dual do teste t/Hodges-Lehmann, recomendado, ou BCa para todos); muda números já exibidos na GUI e no LaTeX.
- STAT-6: AUROC com uma só classe vira null + aviso (recomendado) ou o dataset é recusado no carregamento.
- STAT-7: oferecer cross_validation.select_on: last (schema novo + ADR) além da nota nas legendas.
- SEC-1: com --host 0.0.0.0, exigir --allowed-host (proposto) ou aceitar qualquer Host nesse modo.
- SEC-2: /docs desligado por padrão com VISIONFORGE_DOCS=1 (proposto) ou ligado com exceção de CSP para cdn.jsdelivr.net.
- SEC-8: pesos (weights/*.pt) fora do bundle público (proposto) ou publicados com train_args reescritos; o paper precisa dos pesos?
- SEC-10: revisar o texto legal sobre AGPL antes do merge.
- SEC-11: ultralytics obrigatório e só avisar (A), extra 'detection' (B) ou B + selo 'sem AGPL' no backend torchvision (C). Bloqueia GUI-8.
- SEC-12: assumir publicamente o modo offline garantido (flag nova, citação no paper).
- GUI-2: variantes em um eixo (proposto) ou livres; teto de seeds na GUI.
- GUI-8: mostrar código AGPL do Ultralytics na aba Código (a, b com notas tolerantes, ou c com teto de versão).
- GUI-15: confirmar que logger, linhas de época/trial, STOPPED_NOTE e descrições de schema ficam em inglês de propósito.
- GUI-16: quem valida MPS num Mac Apple Silicon; sem testador, experimental sem anúncio ou espera.
- DOC-1: redação final da garantia de privacidade em README/INSTALL.
- DOC-10/DOC-11: publicar no site recursos só de development com marca 'a partir da 0.22' ou esperar a tag.
- DOC-15: instalar o Chromium do Playwright (~170 MB) nesta máquina.
- PAP-1: entregar o texto do Overleaf e dizer se substitui ou mescla o paper.md.
- PAP-2: tornar público o repositório privado de pesquisa (commit 7620659), depositar no Zenodo ou retirar os números da Research impact statement.
- PAP-3: run.json do estudo no repositório (~2,5 MB) ou depósito Zenodo; a exportação roda na máquina do usuário.
- PAP-6: mudar título/posicionamento do paper de 'local-first' para o desenho estatístico (opcional).
- PAP-9: facilitador do estudo de usabilidade, pré-registro externo e n-alvo (aguarda orientador).
- PAP-11: liberar as fichas de literatura do repositório privado e definir onde o preprint será submetido.
- PAP-12: houve vínculo com empresa ou financiamento a declarar no paper?
- OPS-1: ruleset (checks obrigatórios, bypass do dono), ligar delete_branch_on_merge (hoje off de propósito), apagar as 29 branches mescladas, descrição e topics: configuração de conta.
- OPS-2: ligar secret scanning, push protection e Dependabot security updates (conta); frequência do Dependabot.
- OPS-4: pre-push (default) ou filtro por arquivo no pre-commit; instalar o hook pre-push em cada clone/worktree.
- OPS-5: contato para vulnerabilidades e Code of Conduct, private vulnerability reporting e Discussions.
- OPS-11: criar token com escrita no repositório do site para disparar o deploy pelo cd.yml.
- OPS-12: fechar a #118 agora ou só após a release 0.22.
- OPS-17: critérios de 1.0 e conteúdo do roadmap público.
- OPS-18: nível de compromisso de estabilidade da API de tarefas custom e TASK_API_VERSION.

## Ainda sem task

Itens das fontes que ficaram de fora desta rodada, para a próxima triagem.

- AUDIT achado 19: cobertura fraca em utils/selftest.py (46 %), gui/server.py (69 %), gui/updater.py (77 %), __main__.py (80 %), colab_remote_runner.py (80 %); nenhuma task de testes de CLI (OPS-7 só sobe o gate global).
- AUDIT achado 18 (parte): 74 warnings da suíte e '-W error' para as categorias próprias não viraram task (OPS-13 trata só os sleeps).
- AUDIT seção 6 ganhos rápidos sem task: 'visionforge gui --check', badge de versão do PyPI no site lido em build time, 'visionforge doctor --json' no bug_report.yml, página /docs/changelog no site gerada do CHANGELOG, 'Cite this repository' com DOI de versão.
- AUDIT seção 6 apostas maiores sem task: 'visionforge reproduce <bundle>' (re-treina 1 seed e compara com o IC), catálogo de experimentos em SQLite no lugar de varrer outputs/, entry-point visionforge.exporters para exportadores de terceiros, figura 'tempo até o primeiro treino' no paper.
- reviews product-competitive medium: risco de diluição por sete integrações externas em uma semana (manutenção contra APIs de terceiros) não virou task; é decisão de produto.
- Tracking (MLflow/W&B) só para classificação: fica como issue de continuação em OPS-12, sem task de código.
- Anomalia: campo val_auroc é calculado no split de teste a cada época (nome engana; sem vazamento de seleção, checkpoint por train_loss): anotar no backlog.
- Ideia do tema product-gui: tour da aba Código para tarefas custom lendo o código do pesquisador por inspect.
- Citação do SUS (Brooke) só quando os resultados de usabilidade entrarem no paper, conferida na fonte.
- Seletor de custom_model na GUI (docs/custom/MODELS.md promete campo inexistente): DOC-5 corrige o texto, a feature fica como issue separada sem task.

## Fusões e ajustes do crítico

- PAP-7 fundida em STAT-3: mesmo problema (IC bootstrap percentil em significance.py:253 incoerente com o teste), mesmos arquivos; sources somadas, decisão de método única.
- PAP-8 fundida em PAP-6: ambas editam paper.bib e o State of the field; um PR de referências com teste de chaves.
- DOC-7 fundida em OPS-6: mesmas linhas do CONTRIBUTING.md (delete-branch, Node, BlockRegistry, codespell); o 'engines' do package.json foi para OPS-16.
- DOC-12 fundida em OPS-11: mesmo checklist pós-release no RELEASING.md; o script sync_site_release.py entrou em OPS-11.
- GUI-14 fundida em OPS-9: tags e responses= tocam os mesmos decorators de todos os routers e o mesmo snapshot openapi_paths.json; S8415 (73) é a mesma dívida do achado 17.
- Parte do preconnect do Google Fonts da GUI-13 movida para SEC-2 (mesmo frontend/index.html, mesmo achado R8); GUI-13 ficou com skip link, toque 44 px e metas, depois de SEC-2.
- Remoção das fontes não usadas (geist, space-grotesk, jetbrains-mono) da GUI-12 movida para SEC-5 (mesmo package.json); GUI-12 depende de SEC-5.
- paper/paper.md:112 saiu de DOC-1 e entrou em PAP-5 (paper depende do retorno do Overleaf, PAP-1); a frase de AGPL no paper saiu de SEC-10 para PAP-5; a tagline 'no cloud' saiu de SEC-10 e fica só em DOC-1.
- Issue de continuação 'aba Código para as demais tarefas' saiu de OPS-12: já coberta por GUI-4..8.
- GUI-1 rebaixada de P0 para P1: é produto (não altera números publicados nem privacidade); continua a primeira do bloco de produto.
- GUI-8 rebaixada de P1 para P2 e agora depende de SEC-11: mostrar código AGPL de terceiros só faz sentido depois da decisão de licença.
- DOC-11 rebaixada de P1 para P2: depende de DOC-9, DOC-10 e DOC-15 e de decisão de publicação.
- OPS-6 perdeu a dependência de OPS-1: delete_branch_on_merge está desligado de propósito (memória do usuário), então o texto é 'apague sua branch' sem esperar configuração; ligar ou não virou decisão em OPS-1.
- STAT-4 passou a depender de STAT-3 (mesmo arquivo significance.py; evita conflito); STAT-3 depende de STAT-1 (diff em stage no mesmo arquivo).
- Conferência por amostragem (crítico, 2026-11-09, worktree em fix/stats-augment-wilcoxon sobre d004a38): STAT-1 frase errada do pos_embedding confirmada em DECISIONS.md:7594 e CHANGELOG.md:145, correção do piso já em significance.py:257; SEC-1 reproduzido (Host evil.example -> 200); SEC-3 reproduzido (422 com o sentinela); SEC-2 confirmado (sem CSP, preconnect em index.html:20-21); SEC-6 confirmado (só DVC_NO_ANALYTICS em src); SEC-8 confirmado (weights/*.pt em _CONVENTIONAL, without_local_paths só no run.json); STAT-3/PAP-4 confirmados (4 células IC excluindo zero com p>0,05; H3 'rejected'); GUI-1/GUI-4/GUI-11/GUI-9/DOC-1/DOC-5/OPS-4/PAP-3 confirmados nas linhas citadas. Nenhuma task conferida estava errada.
- OPS-19 nova (crítico): AUDIT achados 20 e 21 não tinham task; 21 está meio refutado (já existe reconfigure UTF-8 em __main__.py:349-352), sobra só o --help impresso antes dele.
- Refutados pelos especialistas e mantidos fora: AUDIT 15 (PyPI já tem provenance PEP 740; classifiers Windows/Linux já existem, macOS não deve entrar sem CI); Sentry do ultralytics não dispara (exige ARGV[0]=='yolo'); segmentation fine-tuning com children[-1] não é bug (último filho é a cabeça densa); prévia de augmentation ignora augment sem efeito visível; backlog linhas 35/38 sobre LaTeX/Markdown já resolvidas; AUDIT 13 tinha a causa errada (core/__init__ eager, não blocks.anomaly), corrigida em GUI-11; #86 com lista de hotspots desatualizada (lista atual em OPS-8); AUDIT 18 recontado (40 sleeps em 24 arquivos).

## Como manter

- Uma task fechada fecha a issue pelo PR (`Closes #N`). Este arquivo não precisa ser editado a cada PR;
  ele é regenerado na próxima triagem.
- Achado novo vira issue com o mesmo formato: problema com arquivo:linha, evidência, proposta,
  critério de aceite checável, arquivos, dependências e se precisa de ADR.
