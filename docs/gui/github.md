# Enviar o relatório de uma execução ao GitHub

No **Histórico**, **Enviar o relatório ao GitHub** grava a execução num
repositório **privado que já existe**, em um único commit (ADR-175). Informe o
repositório no formato `usuário/repositório`. A prévia é local: mostra a
pasta, cada arquivo com o tamanho e o README, e o commit só sai se bater com
essa prévia.

## O que é enviado

Uma pasta `runs/<experimento>/<execução>/` com:

- um README gerado do `run.json`: resumo, métricas finais, uma tabela por
  época (no máximo 200 linhas) e os gráficos embutidos;
- o `run.json` público, sem caminhos da máquina e com `artifacts.model` nulo;
- os gráficos em `figures/` (PNG e JPEG, até 5 MB cada, no máximo 20).

Os arquivos entram num commit só, na branch padrão, sem `force`. Num
repositório vazio, o README vai primeiro pela API de conteúdo, a única forma
de fazer o primeiro commit.

## O que não é enviado

- **Pesos, nunca.** Eles têm outros destinos: Hugging Face Hub e Zenodo.
- O dataset.
- Os caminhos locais do `run.json`.
- Épocas além da 200ª na tabela do README (o `run.json` guarda todas).
- Um repositório novo: o VisionForge **não cria** repositório.

## Token e escopo mínimo

Token fine-grained, limitado ao repositório de experimentos, com a permissão
**Contents: leitura e escrita**. Ele fica salvo por perfil e vai só para
`https://api.github.com`. O repositório é lido antes de qualquer escrita e,
se não for privado, o envio é recusado. Só github.com; GitHub Enterprise não é
suportado. Detalhes: [Tokens e chaves](https://github.com/marcus-vreis/VisionForge/blob/development/docs/gui/tokens.md).

## Se falhar, e repetir

A resposta do GitHub nunca aparece. Os erros 401, 403 e 404 têm mensagem
própria (o 404 também significa que o token não enxerga um repositório
privado); o resto é uma falha genérica. Enviar de novo troca os arquivos da
pasta da execução em **um commit novo**. Arquivos que saíram da execução
continuam no repositório até você apagá-los.

## O que não foi verificado

As chamadas à API (blobs, árvore, commit, atualização da referência) estão
cobertas por testes com simulação; **não foram conferidas com uma conta real**
do GitHub.

Decisão e motivos: ADR-175 em [`DECISIONS.md`](https://github.com/marcus-vreis/VisionForge/blob/development/docs/dev/DECISIONS.md).
