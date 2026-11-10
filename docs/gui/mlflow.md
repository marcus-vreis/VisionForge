# Acompanhar treinos no MLflow local

O MLflow do VisionForge **não sai deste computador**: grava num banco SQLite
local e não usa servidor nem conta (ADR-170). Serve para comparar execuções
na interface do MLflow, sem trocar o TensorBoard nem o `run.json`, que
continuam existindo por conta própria.

Em **Perfis**, ative o MLflow no perfil. A escolha vale para os treinos de
**classificação** enviados depois, a partir de um treino local. Outras tarefas
e outros modos de classificação não são exportados. O ambiente do servidor
precisa do extra `visionforge-studio[mlflow]`; sem ele o treino roda
normalmente e o Histórico mostra que a exportação ficou indisponível.

## O que é enviado

- Os parâmetros públicos da config (valores simples, com o nome achatado,
  como `model.name`).
- As métricas de cada época.
- O `run.json` final, anexado ao run do MLflow depois da avaliação.
- Os runs ficam agrupados por perfil, em `outputs/mlflow/mlflow.db`.

## O que não é enviado

- Nada sai da máquina: esta exportação não faz envio pela rede.
- Parâmetros cujo nome sugere segredo ou caminho (`token`, `password`,
  `secret`, `credential`, `path`, `uri`, `url` e as chaves terminadas em
  `_dir`) ficam de fora. É a mesma seleção usada pelo Hugging Face Hub e pelo
  Weights & Biases.
- O dataset, os pesos e os gráficos não são copiados.
- Atenção: o `run.json` anexado é o registro completo, com os caminhos da
  sua máquina. Como o MLflow é local, ele não vai a lugar nenhum, mas não
  compartilhe a pasta `outputs/mlflow` sem olhar o conteúdo. *Não verificado:*
  em que pasta o MLflow guarda esse anexo; o código só define o banco.

## Token e escopo mínimo

Não há token: o MLflow local não pede credencial. A página
[Tokens e chaves](https://github.com/marcus-vreis/VisionForge/blob/development/docs/gui/tokens.md) cobre os serviços que pedem.

## Se falhar, e repetir

Uma falha no MLflow nunca derruba o treino: o erro é registrado no
`tracking.json` da execução e o Histórico mostra o estado. O botão de
**repetir** reconcilia a execução com o banco: reabre o mesmo run e grava só
as métricas que faltam, sem duplicar o histórico. A repetição só vale para
uma execução de classificação já terminada, num perfil com o MLflow ativo.

Decisão e motivos: ADR-170 em
[`DECISIONS.md`](https://github.com/marcus-vreis/VisionForge/blob/development/docs/dev/DECISIONS.md).
