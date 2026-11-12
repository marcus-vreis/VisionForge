# Enviar treinos ao Weights & Biases

Um perfil pode enviar os treinos de qualquer tarefa (classificação,
regressão, segmentação, anomalia, detecção e tarefas próprias) a um projeto
**privado** do Weights & Biases (ADR-172, ADR-197). É opcional e vem desligado. Em
**Perfis**, ative o W&B, informe o projeto (padrão `visionforge`) e, se
quiser, a organização (`entity`); vazia, vale o usuário dono da chave. O
ambiente do servidor precisa do extra `visionforge-studio[wandb]`.

A classificação envia época a época, durante o treino. As outras tarefas
enviam de uma vez quando o treino termina, a partir do `run.json` da
execução; um treino que falhou com erro não é enviado. Sweeps, K-fold,
comparações, réplicas e continuações não são enviados.

## O que é enviado

- Os parâmetros públicos da config.
- As métricas de cada época, no passo da época, com os nomes do `run.json`
  (`train_loss` e `val_loss` aparecem como `loss/train` e `loss/val`).
- As métricas finais do `run.json`, como resumo do run, com os mesmos nomes:
  a lista do Hugging Face Hub mais `best_val_miou` e `box_loss`; numa tarefa
  própria, todo número cujo nome só tem letras, dígitos e `_`.

## O que não é enviado

- O `run.json`: ele traz o caminho do dataset e as pastas da config.
- Código, git, console, estatísticas do sistema, nome da máquina,
  `requirements.txt` e linha de comando: tudo isso é desligado no envio.
- Parâmetros com nome de segredo ou caminho, como no MLflow.
- Dataset, pesos e gráficos.
- O que o `wandb login` do ambiente guarda: o envio usa só a chave do perfil.
- Atenção: o log `logs/debug.log` do W&B fica neste computador e cita a sua
  pasta.

## Privado antes do primeiro envio

O VisionForge cria o projeto como privado, ou lê o que já existe. Se o
projeto existente não for privado, ele **recusa e não envia nada**, sem
alterar o projeto. Um projeto criado privado pode ser tornado público depois,
no site. Servidor W&B próprio não é suportado.

## Token e escopo mínimo

A chave de API fica salva por perfil, em texto puro neste computador, e a
interface só informa se existe uma. Crie uma chave nova para o VisionForge.
Escopo mínimo: *não verificado*, o código não o define. Onde fica, como
remover e o aviso de servidor na rede:
[Tokens e chaves](https://github.com/marcus-vreis/VisionForge/blob/development/docs/gui/tokens.md).

## Se falhar, e repetir

A falha vai para o `wandb.json` da execução e o treino segue. O **repetir**
do Histórico reabre o mesmo run (o identificador vem do perfil e da pasta da
execução) e o W&B descarta os passos que já tem, então só as épocas que
faltam são acrescentadas. Vale para qualquer tarefa já enviada uma vez. Sem
chave salva, a repetição é recusada.

## O que não foi verificado

O envio foi conferido com o `wandb` real em modo offline, numa execução de
uma época de classificação; as outras tarefas foram conferidas só com
simulação. O caminho online (criar o projeto privado e enviar) está coberto só
por testes com simulação: **não foi exercitado com conta real**.

Decisão e motivos: ADR-172 em [`DECISIONS.md`](https://github.com/marcus-vreis/VisionForge/blob/development/docs/dev/DECISIONS.md).
