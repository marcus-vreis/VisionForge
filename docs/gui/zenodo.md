# Criar um rascunho no Zenodo

No **Histórico**, abra uma execução e use **Criar rascunho no Zenodo**. O
VisionForge monta o bundle da execução e envia como **rascunho não
publicado** (ADR-174). A prévia é local e não precisa de token: mostra o nome,
o tamanho, os membros do arquivo e o que ficou de fora. O envio só aceita o
bundle da prévia: se algo mudou, o servidor recusa e pede outra prévia.

## O que é enviado

- Um bundle zip com o `run.json`, a config, o histórico e as predições
  selecionadas e um manifesto com SHA-256 e tamanho de cada arquivo.
- O `run.json` do bundle é público: `run_dir` vira `.`, os artefatos viram
  nomes de membros do bundle e os caminhos locais ficam só com o nome final.
  A execução no disco não muda.
- Os checkpoints `weights/*.pt` ficam fora: podem trazer caminhos locais
  dentro deles. O manifesto lista cada um que ficou fora e o motivo (ADR-188).
  A config vai reescrita, sem caminhos locais.

## O que não é enviado

- Nenhum metadado do registro: título, autores e licença ficam para você
  preencher no Zenodo. O VisionForge **nunca publica**.
- O dataset e qualquer pasta fora da execução.
- Os gráficos: não são membros do bundle.
- Atenção: um caminho local dentro de um texto que não seja caminho de
  unidade, UNC ou da pasta do usuário não é reescrito. A prévia mostra o
  conteúdo; confira antes de enviar.

## Token e escopo mínimo

São dois tokens por perfil: um do zenodo.org e outro do Sandbox
(sandbox.zenodo.org, contas separadas), e a página escolhe o site. O token
vai só por HTTPS, como Bearer, e só a links do site escolhido. Crie um token
de acesso pessoal **só com `deposit:write`**. Sem `deposit:actions` ele não
consegue publicar, então a publicação fica com você, no site. Detalhes e como
remover: [Tokens e chaves](https://github.com/marcus-vreis/VisionForge/blob/development/docs/gui/tokens.md).

## Se falhar, e repetir

A resposta do Zenodo nunca aparece na tela. Um 401 ou 403 vira "o Zenodo
recusou o token"; qualquer outra falha vira uma mensagem genérica. **Cada
envio cria um rascunho novo**: enviar duas vezes deixa dois rascunhos, e você
guarda um e descarta o outro no Zenodo. Um registro publicado não pode ser
apagado, só versionado.

## O que não foi verificado

O fluxo online (criar o rascunho e enviar o arquivo) tem testes com simulação;
**não foi conferido com uma conta real** do Zenodo nem do Sandbox.

Decisão e motivos: ADR-174 em [`DECISIONS.md`](https://github.com/marcus-vreis/VisionForge/blob/development/docs/dev/DECISIONS.md).
