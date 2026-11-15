# Versionar uma execução com DVC

No **Histórico**, **Versionar com DVC** grava a execução num repositório git
**deste computador**, com os pesos sob o DVC (ADR-176). Informe a pasta do
repositório (git com o DVC iniciado na raiz). **Preparar pasta** cria isso: a
pasta (se a pasta-mãe existir), `git init` quando falta e `dvc init`, com
commit só dos três arquivos que o `dvc init` escreve. Serve para comparar
execuções com `dvc metrics show`, `dvc params diff` e `dvc plots show`.

Só funciona na máquina do servidor: as rotas rodam git e DVC numa pasta do
servidor e respondem apenas a um cliente local, recusando páginas de outro
site. A pasta fica lembrada no navegador, não no perfil.

## O que é enviado

Uma pasta `runs/<experimento>/<execução>/` no repositório:

- no git: o README, o `run.json` público (`artifacts.model` aponta os
  pesos), os gráficos, `params.yaml`, `metrics.json`, `history.csv`,
  `dvc.yaml` (sem estágios) e `model.<ext>.dvc`;
- no DVC: os pesos, `model.<ext>`, via `dvc add`.

O commit é `git commit -- <pasta>`: o que você já tinha em stage fora dela
continua em stage. Com um remoto padrão configurado, roda também o
`dvc push` dessa execução.

## O que não é enviado

- O dataset: o `run.json` já traz a impressão digital e a origem.
- Nenhum `git push`: publicar o repositório fica com você.
- Nenhuma configuração de remoto: use `dvc remote add` fora da página; ela só
  usa o remoto padrão.
- Os caminhos locais do `run.json` e do `params.yaml`.

## Token e escopo mínimo

O VisionForge não guarda token do DVC. As credenciais do remoto (S3, Google
Drive etc.) são do próprio DVC. A exportação roda o git e o DVC reais, sem
terminal e com `DVC_NO_ANALYTICS=1`: o DVC vem do extra `[dvc]` ou do PATH, e
o git, do PATH. Sobre credenciais em geral:
[Tokens e chaves](https://github.com/marcus-vreis/VisionForge/blob/development/docs/gui/tokens.md).

## Se falhar, e repetir

A saída do git e do DVC fica só no log do servidor; a página diz qual
ferramenta e qual passo falharam. O envio só vale para a prévia confirmada
(o resumo cobre o repositório e o SHA-256 dos pesos). Exportar de novo sem
mudança não cria commit. Se o `dvc push` falhar, o commit permanece e a
página avisa; exportar de novo repete o push. Os pesos ocupam o tamanho do
checkpoint duas vezes em disco (pasta e cache do DVC). Arquivos removidos da
execução continuam no repositório até você apagá-los.

## O que não foi verificado

O `dvc push` para um remoto real (S3, Google Drive) **não foi conferido**
com conta real; a documentação do projeto não registra esse teste.

Decisão e motivos: ADR-176 em [`DECISIONS.md`](https://github.com/marcus-vreis/VisionForge/blob/development/docs/dev/DECISIONS.md).
