# Acompanhar treinos no MLflow local

O MLflow do VisionForge **não sai deste computador**: grava num banco SQLite
local e não usa servidor nem conta (ADR-170). Serve para comparar execuções
na interface do MLflow, sem trocar o TensorBoard nem o `run.json`, que
continuam existindo por conta própria.

Em **Perfis**, ative o MLflow no perfil. A escolha vale para os treinos
locais enviados depois, de qualquer tarefa: classificação, regressão,
segmentação, anomalia, detecção e tarefas próprias (ADR-197). Sweeps, K-fold,
comparações, réplicas, continuações e os modos de avaliação e inferência da
classificação não são exportados. O ambiente do servidor precisa do extra
`visionforge-studio[mlflow]`; sem ele o treino roda normalmente e o Histórico
mostra que a exportação ficou indisponível.

## Quando cada tarefa chega

- **Classificação:** época a época, durante o treino, e o `run.json` depois
  da avaliação.
- **Regressão, segmentação, anomalia, detecção e tarefas próprias:** de uma
  vez, quando o treino termina. O VisionForge lê o `run.json` da execução e
  manda cada linha do `history` como uma época, com os nomes do `run.json`
  (`val_miou`, `val_rmse`, `map50`, `val_<métrica>` de uma tarefa própria...).
  `train_loss` e `val_loss` aparecem como `loss/train` e `loss/val`, como na
  classificação. Um treino interrompido pelo botão de parar vai como `KILLED`;
  um treino que falhou com erro não é exportado.

## O que é enviado

- Os parâmetros públicos da config (valores simples, com o nome achatado,
  como `model.name`).
- As métricas de cada época, com os nomes do `run.json`.
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
as métricas que faltam, sem duplicar o histórico. A repetição vale para uma
execução já terminada de qualquer tarefa que tenha passado pela exportação,
num perfil com o MLflow ativo.

Decisão e motivos: ADR-170 e ADR-197 em
[`DECISIONS.md`](https://github.com/marcus-vreis/VisionForge/blob/development/docs/dev/DECISIONS.md).
