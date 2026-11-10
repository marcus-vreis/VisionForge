# Treinar no Google Colab

O VisionForge treina uma **classificação** numa máquina do Colab, na sua conta
Google, e traz o resultado de volta como uma execução local. Esta página diz o
que o código faz hoje. Tudo aqui sai do código atual
(`gui/api/colab_routes.py`, `colab_connection.py`, `colab_jobs.py` e
`gui/api/routes.py`) e das decisões
[ADR-163](https://github.com/marcus-vreis/VisionForge/blob/development/docs/dev/DECISIONS.md)
(treino no Colab pela CLI oficial) e
[ADR-164](https://github.com/marcus-vreis/VisionForge/blob/development/docs/dev/DECISIONS.md)
(dataset no Google Drive, com o consentimento na GUI). O que o código não define
ou que nunca rodou com uma conta real está em *O que não foi verificado*.

## Como funciona

- O VisionForge não fala com o Colab: ele comanda a `google-colab-cli`, fixada
  na versão `0.7.5` (`PINNED_VERSION`), instalada num venv próprio em
  `~/.visionforge/colab-cli/venv`, nunca no ambiente do projeto.
- **Cada perfil entra com a sua conta.** Toda chamada da CLI roda com `HOME` e
  `USERPROFILE` apontando para `~/.visionforge/colab/<perfil>/`; o login fica em
  `~/.visionforge/colab/<perfil>/.config/colab-cli/token.json`. O VisionForge só
  confere se o `token.json` existe e nunca o lê. A pasta base muda com
  `VISIONFORGE_HOME` (veja [tokens e chaves](https://github.com/marcus-vreis/VisionForge/blob/development/docs/gui/tokens.md)).
- O perfil vem no cabeçalho `X-VF-Profile` de cada chamada.
- O login é o fluxo de copiar e colar do Google: a GUI mostra o endereço, você
  autoriza no navegador e cola o código de volta. A conta é identificada pelo
  e-mail que `colab whoami` informa.
- O treino usa a mesma fila e os mesmos endpoints de `/experiment/...` de uma
  execução local. Há uma fila por conta Google, ao lado da fila local, então um
  job local e um job do Colab podem rodar ao mesmo tempo.
- O resultado volta como pacote da execução (`export_experiment_bundle`), é
  verificado e importado como execução local marcada com a origem `colab`. O
  limite é 2 GB; o Histórico marca a execução. O pacote não leva os gráficos
  PNG: a execução importada tem `run.json` e checkpoint.
- A VM é liberada ao fim do job, também quando ele falha ou é parado.

## Rotas

Todas ficam sob `/api` e usam o perfil do cabeçalho.

| Rota | O que faz |
|---|---|
| `GET /api/colab/status` | versão fixada e instalada, estado da instalação, se o perfil entrou, a conta e se há login pendente |
| `POST /api/colab/install` | instala a CLI fixada no venv do VisionForge, em segundo plano |
| `POST /api/colab/login/start` | começa o login do Google e devolve o endereço para abrir no navegador; nada é alocado no Colab |
| `POST /api/colab/login/finish` | entrega à CLI o código que o Google mostrou |
| `POST /api/colab/login/cancel` | cancela o login em andamento |
| `POST /api/colab/logout` | esquece o login do perfil, para entrar com outra conta |
| `GET /api/colab/usage` | saldo, taxa por hora, alocações ativas e horas restantes da conta |
| `GET /api/colab/sessions` | as máquinas vivas na conta, com a marca `ours` quando o VisionForge as iniciou |
| `POST /api/colab/sessions/stop` | libera uma máquina pelo nome e devolve as que continuam vivas |
| `POST /api/colab/experiment/run` | enfileira o treino de uma classificação no Colab |
| `POST /api/colab/drive/continue` | avisa que o acesso ao Drive foi permitido, para o job continuar |

Respostas de erro que o código define:

- `409` em `login/start`, `usage`, `sessions`, `sessions/stop` e
  `experiment/run`: a CLI não está instalada na versão fixada ou o perfil não
  entrou (`colab.not_installed`, `colab.not_logged_in`).
- `502` quando a CLI falha (`ColabError`), sem repassar o código de login.
- `404` em `drive/continue` quando nenhum job está esperando a permissão.

## O pedido de treino

`POST /api/colab/experiment/run` recebe (`ColabRunRequest`):

- `config`: a configuração do experimento. O servidor valida tudo, menos que a
  pasta `data.base_dir` exista neste computador, porque o dataset está em outro
  lugar.
- `source`, de onde a VM tira o dataset:
  - `url`: um zip num endereço `http(s)` público, com as pastas de treino,
    validação e teste da config dentro;
  - `provider`: um dataset público que a VM baixa sozinha (`torchvision` ou
    `huggingface`), com `dataset` e `limit` opcional; nenhuma credencial sai
    deste computador;
  - `drive`: a pasta do dataset da própria config, dentro do Google Drive para
    desktop (ADR-164).
- `accelerator`: `T4` (padrão), `L4`, `A100`, `H100` ou `G4`.
- `drive_export`: também grava o pacote da execução em
  `MyDrive/VisionForge/runs/`. Uma falha aí não derruba a execução.

## Dataset no Google Drive

- O servidor traduz `data.base_dir` para `/content/drive/MyDrive/...` e manda a
  contagem de arquivos local. A pasta "Meu Drive" é achada, nesta ordem: a pasta
  digitada (`my_drive_root`), o `DefaultMountPoint` do Drive para desktop no
  registro, ou um componente "My Drive"/"Meu Drive" logo abaixo de uma letra de
  unidade ou da sua pasta pessoal. Uma pasta fora do Drive, ou vazia, dá `400`
  antes de gastar uma máquina.
- A montagem pede o consentimento no navegador e **não é guardada**: cada máquina
  nova pede de novo. A GUI mostra "Permitir acesso ao Drive", que abre a página
  em outra aba; ao voltar à janela, a GUI chama
  `POST /api/colab/drive/continue` e o servidor manda o Enter que a CLI espera.
  Ninguém precisa de um terminal. "Já permiti" é o recurso quando o foco não
  dispara.
- Se a montagem falha com `400`, a CLI é reiniciada e tenta de novo com um
  endereço novo, até 3 vezes; a espera pela pessoa é de 600 s por tentativa e o
  silêncio da CLI, de 120 s.
- A VM copia a pasta para o disco dela e falha se achar menos arquivos do que
  este computador tem, o que costuma ser uma pasta que o Drive ainda envia.

## Limites e cuidados

- **Só classificação.** O treino no Colab é de classificação (ADR-163, fase 2).
- Os eventos do treino vêm do `stdout` do `colab exec`. O código de saída do
  `exec` não é confiável na versão fixada, então o fim do job é um evento
  próprio; um treino que termina sem ele falha com a última saída.
- Um job na fila que chega à montagem do Drive sem ninguém na tela espera até
  3 × 10 min, falha e libera a máquina.
- O saldo mostrado em horas restantes é o saldo dividido pela taxa da máquina;
  só aparece quando a conta informa unidades de computação.
- `GET /api/colab/sessions` existe para uma execução interrompida não deixar uma
  máquina ligada: as que o VisionForge iniciou levam `ours`.

## O que não foi verificado

- Nada aqui foi rodado com uma conta Google real ao escrever este guia; é leitura
  do código e das decisões. Os testes do repositório usam uma CLI falsa.
- A detecção do "Meu Drive" nunca rodou numa máquina com o Google Drive para
  desktop instalado (risco registrado na ADR-164).
- A CLI é jovem e muda; a versão fixada e o texto em inglês que o código lê dela
  (como "Please visit" e "Mounted at") são a proteção. Outra versão pode quebrar.
