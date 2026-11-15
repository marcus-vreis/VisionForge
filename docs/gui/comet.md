# Enviar treinos ao Comet

Um perfil pode enviar os treinos de qualquer tarefa (classificação,
regressão, segmentação, anomalia, detecção e tarefas próprias) a um projeto
**privado** do Comet (ADR-200), no mesmo contrato do Weights & Biases
(ADR-172, ADR-197). É opcional e vem desligado. Em **Perfis**, ative o Comet,
informe o projeto (padrão `visionforge`) e, se quiser, o workspace; vazio,
vale o workspace padrão da chave. Projeto e workspace só aceitam minúsculas,
números, ponto, hífen e sublinhado. O ambiente do servidor precisa do extra
`visionforge-studio[comet]`.

A classificação envia época a época, durante o treino. As outras tarefas
enviam de uma vez quando o treino termina, a partir do `run.json` da
execução; um treino que falhou com erro não é enviado. Sweeps, K-fold,
comparações, réplicas e continuações não são enviados.

## O que é enviado

- Os parâmetros públicos da config.
- As métricas de cada época, no passo da época, com os nomes do `run.json`
  (`train_loss` e `val_loss` aparecem como `loss/train` e `loss/val`).
- As métricas finais do `run.json`, com os mesmos nomes e a mesma lista do
  W&B, e o estado final em `visionforge_status` (`FINISHED`, `KILLED` ou
  `FAILED`).
- O nome do experimento é o nome da pasta da execução (data e hora).
- O que o SDK acrescenta por conta própria: `curr_step`, `curr_epoch` e a
  duração do experimento (`process duration`).

## O que não é enviado

- O `run.json`: ele traz o caminho do dataset e as pastas da config.
- Código, git (metadados e patch), console, pacotes instalados, nome da
  máquina, usuário do sistema, IP, CPU, GPU, disco, rede, linha de comando,
  grafo do modelo e CO2: tudo isso é desligado no envio, argumento por
  argumento, e nenhuma variável `COMET_*` do ambiente religa.
- O auto-log de frameworks (o `comet_ml` intercepta o import do torch e de
  outras bibliotecas), os relatórios de erro do SDK ao Sentry e os relatórios
  internos ao Comet: o VisionForge desliga os três antes de importar o SDK.
- O e-mail de resumo que o SDK manda ao fim de um experimento novo: o título
  dele traz o usuário do sistema.
- Parâmetros com nome de segredo ou caminho, como no MLflow e no W&B.
- Dataset, pesos e gráficos.
- O que o `comet login` ou a variável `COMET_API_KEY` guardam: o envio usa só
  a chave do perfil.

## Privado antes do primeiro envio

O VisionForge cria o projeto como privado, ou lê o que já existe. Se o
projeto existente for público, ele **recusa e não envia nada**, sem alterar
o projeto. Um projeto criado privado pode ser tornado público depois, no
site. O endereço do servidor é o do SDK: um `COMET_URL_OVERRIDE` no ambiente
manda para outro servidor, e servidor Comet próprio não foi testado.

## Token e escopo mínimo

A chave de API fica salva por perfil, em texto puro neste computador, e a
interface só informa se existe uma. Crie uma chave nova para o VisionForge.
Escopo mínimo: *não verificado*, o código não o define. Onde fica, como
remover e o aviso de servidor na rede:
[Tokens e chaves](https://github.com/marcus-vreis/VisionForge/blob/development/docs/gui/tokens.md).

## Se falhar, e repetir

A falha vai para o `comet.json` da execução e o treino segue. O **repetir**
do Histórico reabre o mesmo experimento (a chave do experimento vem do perfil
e da pasta da execução). Diferente do W&B, o Comet guarda um passo repetido;
por isso o `comet.json` anota a última época que o Comet confirmou, e a
repetição envia só as épocas depois dela. Se o servidor caiu antes dessa
confirmação, as épocas já enviadas podem aparecer duas vezes. Sem chave salva,
a repetição é recusada.

O SDK mantém um experimento aberto por processo e encerra o anterior quando
abre outro. Por isso, enquanto um envio ao Comet está aberto (um treino de
classificação em andamento, por exemplo), outro envio falha com aviso e pode
ser repetido depois.

## O que não foi verificado

O `comet_ml` 3.58.12 real foi rodado em modo offline com a mesma
configuração: o arquivo do experimento só trouxe a config, as métricas, o nome
e o estado, sem usuário, máquina ou caminho. O caminho online (criar o
projeto privado, enviar e repetir) está coberto só por testes com simulação:
**não foi exercitado com conta real**.

Decisão e motivos: ADR-200 em [`DECISIONS.md`](https://github.com/marcus-vreis/VisionForge/blob/development/docs/dev/DECISIONS.md).
