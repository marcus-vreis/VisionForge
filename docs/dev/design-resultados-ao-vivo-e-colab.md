# Design: aba "Resultados" ao vivo e treino no Google Colab

Status: proposta, nada implementado. Precisa da aprovação do Marcus e de um ADR
(ADR-160 em diante) antes do código. Data: 2026-11-04.

Convenção: **[fato]** foi verificado no código ou na fonte citada;
**[inferência]** é estimativa minha, ainda sem medição.

---

## 1. Terceira aba "Resultados" no overlay de treino

### O que existe hoje
- [fato] O overlay tem duas abas, `log` e `story` (`SheetTab`,
  `frontend/src/components/TrainingOverlay.tsx`).
- [fato] Cada `epoch_end` do SSE já traz `train_loss`, `val_loss`,
  `val_accuracy` (ou `val_<métrica>`) e `elapsed_s`
  (`core/training_loop.py::report_epoch`, `tasks/engine.py`). A detecção traz
  mAP50/mAP50-95/P/R e as losses YOLO.
- [fato] `overfittingSince()` já existe em `frontend/src/lib/training-story.ts`
  (ADR-119).
- [fato] O Grad-CAM existe só pós-treino (`POST /runs/{id}/gradcam`,
  `gui/api/torch_gradcam.py`). Atende classificação, regressão e segmentação;
  detecção e anomalia ficam de fora.
- [fato] `RunEventLog` guarda todos os eventos da run em memória e os reenvia
  quando o stream reconecta (ADR-159).

### Proposta em três fases

**Fase A: só frontend, sem custo de treino.**
- Curvas até a época atual (loss de treino e de validação, métrica principal),
  montadas a partir dos `epoch_end` que o cliente já recebe.
- Card "melhor época": o argmin/argmax da métrica do checkpoint, que é a mesma
  regra de `save_epoch_checkpoint` (`val_loss` para classificação, mAP50-95
  para detecção etc.).
- Alerta de overfitting reaproveitando `overfittingSince`.
- Em grid, comparação e CV: as curvas da unidade em andamento, mais uma
  tabela com as unidades já fechadas (`trial_end`).

**Fase B: campos baratos no `epoch_end`.**
- `lr`: um `optimizer.param_groups[0]["lr"]` por época, custo zero.
- Matriz de confusão de validação, só em classificação: as predições já são
  calculadas em `_eval_epoch`, então custa O(N) inteiros. Enviar apenas para
  até ~30 classes (30×30 inteiros por época); acima disso, só o top-k das
  confusões.
- Para não inchar o `run.json`, isso não entra no `history`. Fica só no
  stream e, opcionalmente, num `live/epoch_metrics.jsonl` dentro do run dir.

**Fase C: Grad-CAM automático de amostras de validação.**
- Amostras: K = 4 a 8 imagens de validação fixas, escolhidas uma única vez com
  um `torch.Generator` próprio, nunca o RNG global (ver a regra 1 abaixo).
  Estratificar por classe quando der.
- Custo [inferência]: K forwards e K backwards parciais, contra milhares por
  época. Num ResNet-18 a 224 px em GPU fica na casa de dezenas de ms, e o
  overlay em PIL leva ~10 a 30 ms por imagem na CPU. Comparado à época, é
  desprezível. Em CPU com época curta, deixa de ser.
- VRAM [inferência]: o pico equivale a um batch de K com grafo, menor que o
  pico do treino quando `batch_size ≥ K`. Rodar depois do `_eval_epoch` e
  antes do checkpoint, com `model.eval()`.
- Frequência: a cada N épocas (padrão 5), mais a última e toda época que vira
  a melhor. Botão "gerar agora", que vale na próxima fronteira de época,
  pelo mesmo mecanismo do stop (ADR-088). Desligado por padrão para runs em
  CPU.
- Disco: só `insights/latest/` e `insights/best/` são sobrescritos. Nada de
  um diretório por época (100 épocas × 8 PNGs = 800 arquivos sem uso).

**Regras que não se negociam (reprodutibilidade, artigo):**
1. Não consumir o RNG global nem o gerador dos loaders: o insight não pode
   mudar a trajetória do treino. É preciso um teste que compare o `history`
   com e sem insights, mesma seed, em CPU com `deterministic=True`.
2. Não tocar em `.grad` dos parâmetros: usar
   `torch.autograd.grad(score, activation)` em vez de `.backward()`.
3. Restaurar `model.train()` e o estado do BN, ou seja, rodar em `eval`.
4. Falha no insight vira log de aviso; nunca derruba a run.

### Por tarefa

| Tarefa | O que mostrar ao vivo | Observação |
| --- | --- | --- |
| Classificação | Grad-CAM + matriz de confusão parcial | Caminho pronto em `gradcam.py` |
| Regressão | Grad-CAM do alvo + scatter predito × real das K amostras | Grad-CAM já suportado |
| Segmentação | Máscara predita sobreposta (mais útil que CAM) | Grad-CAM existe, mas a máscara diz mais |
| Detecção | Caixas preditas em K imagens de validação | Ultralytics é dono do loop: `on_fit_epoch_end` + `model.predict`; nada de CAM |
| Anomalia | Mapa de anomalia nativo de K amostras | Já é um heatmap; PatchCore não tem épocas (usar a fase "scoring") |

### Contrato
- Evento SSE novo: `{"event": "insight", "epoch", "kind": "gradcam" | "boxes" |
  "mask" | "anomaly_map", "images": [<caminhos relativos ao run dir>],
  "trial_index"?}`.
- Rota de leitura: `GET /runs/{id}/insights/{path}`, servindo só dentro de
  `run_dir/insights/` (o mesmo cuidado de path traversal das rotas de
  artefato).
- `run.json`: `"insights": {"every_n": 5, "samples": [...], "best_epoch": n}`.
  Campo opcional, sem quebrar o schema; avaliar se exige `schema_version` e
  registrar no ADR.

### Riscos
- DataParallel/AMP: o hook precisa do módulo desembrulhado
  (`checkpoint_components(..., unwrap_model=True)` já resolve isso para o
  checkpoint).
- `persistent_workers` no loader de validação: pegar as K amostras pelo
  `dataset`, sem iterar o loader.
- Volume do log de eventos em runs longas: um evento por época mais um
  `insight` a cada N épocas fica pequeno.

**Recomendação:** A, depois B, depois C, cada uma num PR. A já entrega a maior
parte do valor sem tocar no treino.

---

## 2. Treino no Google Colab

### Restrições verificadas
Fonte: FAQ do Colab, https://research.google.com/colaboratory/faq.html,
lida em 2026-11-04.
- [fato] Proibido em qualquer runtime gerenciado: "connecting to remote
  proxies", hospedagem de arquivos e serviços web sem relação com computação
  interativa.
- [fato] No plano gratuito sem saldo de compute units, também é proibido:
  "remote control such as SSH shells, remote desktops", "bypassing the notebook
  UI to interact primarily via a web UI" e rodar workers de computação
  distribuída. A runtime pode ser encerrada sem aviso.
- [fato] Duração: no máximo 12 h no gratuito, até 24 h no Pro+. Há timeout
  por inatividade, e o Google não publica os limites.
- [fato] O "runtime local" faz o contrário do que queremos: a interface do
  Colab usa a máquina do usuário, sem dar GPU a ela. O Drive não monta em
  runtime local.

### Opções avaliadas

| Opção | Funciona? | Termos | Veredito |
| --- | --- | --- | --- |
| Runtime local do Colab | Não dá GPU | ok | Descartar: não resolve o problema |
| Worker do VisionForge no Colab + túnel (ngrok/cloudflared) até a GUI local | Tecnicamente sim | Viola "remote proxies" e, no gratuito, "bypassing the notebook UI" e "distributed computing workers" | **Descartar**: risco de ban da conta do usuário |
| Notebook gerado, executado pelo usuário, treino headless via CLI, dataset e saída no Drive | Sim | Uso interativo normal de notebook | **Recomendado** |
| Colab Enterprise / VM do GCP Marketplace | Sim, sem limites de sessão | Pago, exige projeto GCP | Fase futura, opcional |

### Arquitetura recomendada: "Exportar para o Colab"

1. Na GUI, a opção "Colab" fica no seletor de dispositivo, mas não executa
   nada local. O que ela gera é um **pacote de execução**:
   - `config.yaml` resolvido (o mesmo `ExperimentConfig` validado);
   - `visionforge_colab.ipynb` gerado a partir de um template versionado, com
     estas células:
     1. montar o Drive;
     2. `pip install visionforge-studio==<versão local>`, com versão
        **fixa**, para reproduzir o que foi configurado;
     3. conferir a GPU;
     4. copiar o dataset do Drive para `/content` (ler do FUSE do Drive em
        cada época é lento);
     5. `visionforge config.yaml` (a CLI já existe:
        `visionforge.__main__:main`);
     6. exportar a run com `export_experiment_bundle` para o Drive.
2. O dataset vai para o Drive de uma de duas formas:
   - **2a (sem OAuth)**: a GUI gera um `.zip` do dataset e o usuário sobe à
     mão;
   - **2b (Drive API)**: upload direto pela GUI.
3. Depois do treino, o usuário volta à GUI e usa "Importar run", apontando o
   bundle baixado ou, na fase 3, buscando pelo Drive API. O código tem
   `export_experiment_bundle` e `verify_experiment_bundle`
   (`core/experiment_bundle.py`), mas **falta a importação**: uma rota que
   verifique o manifest e extraia em `models/` do perfil.
4. Progresso: o notebook mostra o log da CLI. Espelhar o progresso na GUI
   local, lendo um `events.jsonl` que o notebook grava no Drive, é a fase 3
   e depende de uma verificação sobre escopos OAuth (abaixo).

### Drive API e OAuth
- [fato, a reconfirmar na doc do Google antes de implementar] O escopo
  `drive.file` não é sensível e só dá acesso a arquivos criados ou abertos
  pelo app. `drive` e `drive.readonly` são restritos e exigem verificação de
  segurança para um app publicado.
- Consequência: com `drive.file`, a GUI enxerga o dataset que ela mesma
  subiu, mas **não** enxerga arquivos novos que o Colab criou no Drive. A
  saída precisa voltar por um destes caminhos:
  - o Picker do Google, com o usuário escolhendo o bundle; abrir pelo Picker
    concede acesso ao arquivo;
  - um arquivo placeholder criado antes pela GUI e sobrescrito pelo notebook.
    [a verificar] Não sei se escrever pelo FUSE do Drive preserva o file ID.
- Distribuição: um app desktop open source não pode embutir client secret
  "de verdade". Usar o fluxo de app instalado (loopback + PKCE), com
  credenciais que o próprio usuário cria no Google Cloud Console ou um client
  do projeto com a tela de consentimento em "testing", limitada a 100
  usuários. O token fica em armazenamento do SO (keyring), nunca em
  `run.json` nem em log.

### Fases
1. **F1, sem OAuth (1 PR):** gerador de notebook + zip do dataset + "Importar
   run (bundle)". Entrega o valor principal e respeita os termos.
2. **F2:** upload do dataset ao Drive pela GUI com `drive.file`, com
   retomada (resumable upload; datasets de GBs).
3. **F3:** volta automática do resultado pelo Picker e progresso espelhado
   do `events.jsonl`.
4. **F4, opcional:** Colab Enterprise/GCP, para quem paga.

### Riscos
- **Sessão cai no meio** (timeout, 12 h): o resume já existe (ADR de resume:
  `resume.pt` por época). O notebook deve gravar `models_dir` no Drive para
  sobreviver e oferecer a célula "retomar".
- **Versões:** o torch do Colab muda. A CLI instalada por pip pode puxar
  outro torch; pinar só `visionforge-studio` e aceitar o torch do Colab, com a
  versão registrada no `run.json` (já é capturada por `capture_environment`).
- **Reprodutibilidade:** a GPU muda (T4/L4/A100), então a run deve marcar
  `device_used` e o ambiente. Isso já acontece.
- **Segurança:** o notebook gerado não pode conter token nem path local. Nada
  de túnel, então nenhuma porta da máquina do usuário fica exposta.
- **Termos:** se o Google endurecer as regras para "headless training", o
  modelo F1 continua sendo uso interativo legítimo: o usuário roda as células.

**Recomendação:** F1 primeiro, sem OAuth e sem túnel. Só seguir para F2/F3
se houver demanda real, porque o custo de manter OAuth num app open source é
alto em credenciais, verificação do Google e suporte.

### Atualização 2026-11-04: a CLI oficial do Colab muda a recomendação

O Google publicou a `google-colab-cli` (repositório `googlecolab/google-colab-cli`,
Apache-2.0). Ela aloca um runtime, roda um script, transfere arquivos e libera a
VM, tudo por linha de comando e pela conta do usuário. Isso torna possível a
automação completa que a seção acima descartava por falta de API.

**Fatos verificados, lidos no código e na doc do repositório em 2026-11-04:**

| Pergunta | Resposta | Fonte |
| --- | --- | --- |
| Roda no Windows? | Não: o README diz "Linux and macOS only". No WSL desta máquina há Ubuntu 24.04 com Python 3.12.3, e a CLI pede `>=3.12`. | `README.md`, `pyproject.toml` |
| GPU no plano gratuito? | A doc chama a T4 de "standard free-tier GPU", mas o skill do próprio repositório avisa que "most accounts can only get CPU"; um 400 no `colab new --gpu` significa sem cota. Só se descobre na conta. | `docs/01_session_management.md`, `skills/colab-operator/SKILL.md` |
| A saída chega ao vivo? | Sim. O `--timeout` de `colab run` é um limite de *silêncio*: "a script that keeps printing can run as long as it needs"; `0` tira o limite (mudança de 2026-10-24). | `docs/05_run_command.md` |
| A VM é liberada? | `colab run` sem `--keep` libera a VM num `finally`, mesmo se o script falhar. Se o processo local morrer, a VM fica até o backend recolher por inatividade, consumindo compute units. | `docs/05_run_command.md`, `SKILL.md` |
| Login | OAuth do usuário (`colab auth`, cliente embutido) ou ADC; escopos `userinfo.email` e `colaboratory`. O token fica no keyring; o VisionForge não precisa vê-lo. | `docs/04_automation_and_utility.md` |
| Upload de arquivos | Pela Contents API do Jupyter, em JSON com base64. Serve para configs e zips pequenos; para datasets de GBs a própria doc admite limite. | `docs/03_file_management.md` |
| Uso de GPU e saldo | `colab usage` lê o saldo e a taxa de compute units (`/tun/m/ccu-info`); `nvidia-smi` roda na VM via `exec`. | `docs/04_automation_and_utility.md` |

**Arquitetura que isso permite (inferência, a confirmar no spike prático):**

1. **Backend "Colab"** ao lado da fila local: o VisionForge chama a CLI
   (`wsl.exe -d <distro> colab ...` no Windows; direto em Linux/macOS).
2. **Dataset:** a VM baixa o dataset de onde ele já está (Drive, Hugging Face,
   URL), nunca por `colab upload`. Zips pequenos podem subir pela CLI.
3. **Treino:** `colab run` de um script que instala o `visionforge-studio` e roda
   `visionforge train`. O script imprime os eventos (`start`, `epoch_end`,
   `batch`, `insight`...) como linhas JSON no stdout. A CLI repassa a saída ao
   vivo, e o VisionForge alimenta com ela o mesmo stream SSE da aba Resultados,
   sem túnel e sem porta aberta. O `batch` a cada 2 s mantém o `--timeout` longe
   do limite.
4. **Resultado:** `colab download` do run dir compactado ao fim, importado como
   uma run local. O histórico fica no VisionForge, mas a importação de bundle
   (seção F1) continua por fazer.
5. **Limpeza:** ao iniciar, o VisionForge lista `colab sessions` e oferece parar
   as que sobraram de uma execução interrompida.

**Riscos novos:**
- A CLI fala com endpoints internos do Colab (`/tun/m/...`) e se atualiza com
  frequência (`colab update`), então uma versão nova pode quebrar o contrato.
  Pinar a versão e testar a cada atualização.
- No Windows a dependência do WSL é uma barreira de instalação para o usuário.
- A conta pode não ter GPU; a GUI precisa dizer isso, e não falhar em silêncio.

**Próximo passo (precisa do usuário):** o spike prático, no WSL:
`pip install google-colab-cli` num venv, `colab auth` (login no navegador),
`colab new --gpu T4`, um `colab run` de 3 minutos imprimindo uma linha a cada
segundo, `colab usage` e `colab stop`. Ele responde a duas coisas que só a conta
diz: se há GPU no plano dela e qual a latência real da saída.
