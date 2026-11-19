# Modo servidor — um VisionForge para várias pessoas (experimental)

> **Leia antes de instalar.** O modo servidor está em construção. Nesta versão
> ele identifica quem entrou, lê datasets das pastas de cada pessoa e das
> compartilhadas, **treina um job por vez** nas cinco tarefas, com as
> estratégias (validação cruzada, busca, comparação, réplicas), e mostra a fila,
> o progresso e o histórico de cada um. A avaliação depois do treino, o envio de
> datasets pela tela e o Colab ainda não estão disponíveis, e a tela não os
> mostra. E ele é para um **grupo que já confia entre
> si**: o treino roda dentro do processo do servidor, sem isolamento entre as
> pessoas. Não ofereça este servidor a desconhecidos.

Esta página sai do código atual (`gui/server_mode.py`) e das decisões ADR-212
a ADR-217. O que não foi testado está marcado como *não verificado*.

## O que muda em relação ao uso local

No seu computador, o VisionForge confia em quem abre a página: o perfil é uma
escolha do navegador. Num servidor isso não basta, porque qualquer pessoa
poderia escolher o perfil de outra. No modo servidor:

- **O login fica num proxy.** O VisionForge não tem tela de senha. Um proxy
  reverso que faz login (oauth2-proxy, Authelia) fica na frente, com a conta do
  GitHub, do Google ou da instituição, e diz ao VisionForge quem entrou.
- **Só o proxy é atendido.** Um pedido que chega direto à porta do VisionForge,
  sem passar pelo proxy, é recusado (403).
- **Cada conta tem a sua pasta.** O nome que o proxy informa vira o perfil da
  pessoa, criado no primeiro acesso em `outputs/profiles/`. O que o navegador
  mandar como perfil é ignorado.
- **O que não está pronto responde 404**, com a mensagem "Esta função não está
  disponível quando o VisionForge roda como servidor".
- **A tela mostra só o que funciona.** Ao abrir, a página pergunta ao servidor
  o que ele oferece e esconde o resto (veja "Como a tela muda").

## Como a tela muda

Num servidor compartilhado a página é a mesma, com menos botões:

- **Sem "procurar pasta".** O seletor abriria uma janela na tela do servidor,
  não na sua. O caminho do dataset é digitado, e logo abaixo do campo a página
  lista as pastas que o servidor aceita: a sua primeiro, depois as
  compartilhadas. Copie uma delas e complete com o nome do dataset.
- **As estratégias aparecem como na instalação local** (validação cruzada,
  busca, comparação, réplicas). Uma busca ou uma variante só pode variar campos
  de número, de escolha e o nome do modelo: o dataset, os pesos e as pastas
  ficam os do formulário (veja "Estratégias num servidor").
- **Sem o botão "datasets"** (baixar e Laboratório), sem "criar dataset de
  exemplo", sem as prévias de aumento e de pré-processamento e sem a auditoria.
- **Sem Colab**, sem aviso de atualização e sem as abas de tarefas próprias.
- **No Histórico**, um run mostra métricas, gráficos e "Carregar no
  formulário". Retomar, testar, Grad-CAM, exportar ONNX, baixar o ZIP e
  publicar ainda não aparecem.
- **O menu do seu nome** mostra a conta com que você entrou e a sua pasta. Não
  há outro perfil para escolher, e as integrações (MLflow, W&B, Comet) ainda
  não são configuráveis ali.
- **Guia:** só o tour da interface. O guia do primeiro treino depende do
  dataset de exemplo.

Na primeira visita a página pergunta o seu nome. Ele serve só para a saudação
e fica no seu navegador; a conta continua sendo a do login.

## Como ligar

```powershell
visionforge gui --server --trusted-proxy 127.0.0.1 --allowed-host visionforge.meulab.org
```

| Opção | Para que serve |
| --- | --- |
| `--server` | Liga o modo servidor. Sem ela nada muda. |
| `--trusted-proxy <endereço>` | IP (ou bloco CIDR, como `172.18.0.0/16`) do proxy de login. Obrigatória: sem ela o servidor não sobe. Pode repetir. |
| `--user-header <nome>` | Cabeçalho em que o proxy informa a conta. O padrão é `X-Forwarded-User` (oauth2-proxy); no Authelia é `Remote-User`. |
| `--datasets-dir <pasta>` | Pasta de datasets que todos podem **ler**. Pode repetir. A pasta precisa existir. |
| `--allowed-host <nome>` | O nome público pelo qual as pessoas chegam (o do proxy). |

Deixe o VisionForge escutando só em `127.0.0.1` (o padrão) quando o proxy roda
na mesma máquina. Se o proxy roda em outra máquina ou em outro contêiner, use
`--host` com o endereço da rede interna e feche a porta para o resto no
firewall.

### O que o proxy precisa fazer

1. Exigir login em **todas** as rotas, menos `GET /api/health` (que serve para
   o proxy saber se o VisionForge está de pé).
2. Enviar a conta autenticada no cabeçalho combinado e **apagar** esse
   cabeçalho de qualquer pedido que venha do navegador. O oauth2-proxy e o
   Authelia fazem isso por padrão; num proxy montado à mão, confira.
3. Servir por HTTPS.

*Não verificado:* nenhuma instalação real com oauth2-proxy ou Authelia foi
feita ainda; o comportamento acima foi testado com pedidos simulados.

## Onde ficam os datasets

No modo servidor uma pessoa só consegue apontar para duas áreas do disco:

- **A pasta dela:** `outputs/profiles/<perfil>/datasets/`, criada no primeiro
  acesso. Ainda não há envio de dataset pela tela, então hoje quem põe arquivos
  ali é quem administra o servidor.
- **As pastas compartilhadas:** as que o servidor recebeu em `--datasets-dir`.
  Servem para os datasets que o grupo todo usa.

`GET /api/session` devolve essas pastas em `dataset_dirs`, a da pessoa
primeiro. O caminho do dataset precisa ser completo e estar dentro de uma
delas; qualquer outro é recusado com a mensagem "Neste servidor o dataset
precisa estar na sua pasta de datasets ou numa pasta compartilhada". Os nomes
de subpasta (`train`, `images`, `train.csv`) não podem conter `..` nem começar
na raiz do disco.

Um limite conhecido: no manifesto CSV da regressão, uma linha pode citar uma
imagem por caminho completo. O servidor não abre esse arquivo, mas a contagem
de "imagens ausentes" revela se ele existe.

## O que um treino pode usar

Ao enviar um treino, o servidor confere o formulário inteiro antes de pôr o job
na fila:

- **Dataset e pesos** precisam estar na sua pasta (`outputs/profiles/<perfil>/`)
  ou numa pasta compartilhada.
- **Tudo o que o run grava** (modelos, gráficos, logs, relatórios) vai para a
  sua pasta. Uma pasta compartilhada é só de leitura.
- **Nome do modelo** precisa ser um nome (`resnet18`, `yolo11n`,
  `convnext_tiny.fb_in22k`), não um caminho nem um endereço. Sem isso, um nome
  como `../pesos` faria o servidor abrir um arquivo qualquer como checkpoint.
- **Modelo próprio** (`custom_model`) não é aceito: é código trazido por quem
  usa.

## A fila

Cada pessoa vê os próprios jobs por inteiro. Dos outros aparece só um número:
quantos estão rodando ou esperando na frente do seu primeiro job (ou, se você
não tem job, quantos há na fila). Um run de outra pessoa responde como um run
que não existe: não dá para ler o progresso, o resultado nem parar.

## O que responde hoje

| Função | No modo servidor |
| --- | --- |
| Abrir a página, esquemas dos formulários, informações do sistema | sim |
| Histórico: listar, abrir e apagar os **seus** runs | sim |
| Tokens de publicação (Hugging Face, GitHub, Zenodo, W&B, Comet) do **seu** perfil | sim |
| `GET /api/session` (quem entrou, as pastas de dataset e o que está fechado) | sim |
| Ler um dataset para conferir: divisões, contagem por classe, amostras e miniaturas, nas cinco tarefas | sim, só na **sua** pasta de datasets e nas pastas compartilhadas |
| Baixar dataset, dataset de exemplo, prévia de aumento e de pré-processamento, auditoria e Laboratório de datasets | ainda não |
| Treinar um run simples (classificação, detecção, segmentação, regressão, anomalia) | sim, com dataset e pesos das **suas** pastas ou das compartilhadas |
| Fila | sim: os **seus** jobs e, dos outros, só quantos estão na frente |
| Progresso, resultado e gráficos de um run | sim, só dos **seus** |
| Estratégias: validação cruzada, busca (grid, random, Optuna), comparação de modelos, réplicas e comparação replicada, nas cinco tarefas | sim, com as mesmas regras do treino simples e a regra dos eixos abaixo |
| Retomar, testar, Grad-CAM, exportar ONNX, publicar | ainda não (cada rota será revista) |
| Seletor de pasta, "abrir pasta", atualizar o VisionForge, Colab, tarefas próprias, credenciais de download | desligados no modo servidor |

## Estratégias num servidor

Uma estratégia treina vários runs a partir de um config base. O config base
segue as regras do treino simples: dataset e pesos nas suas pastas ou nas
compartilhadas, tudo o que é gravado na sua pasta.

Busca, comparação e variantes mudam campos do config a cada treino, depois
dessa conferência. Por isso, num servidor, o que elas podem mudar é limitado:

| O que a estratégia tenta variar | Resposta |
| --- | --- |
| Um número, um liga/desliga ou uma escolha (`training.learning_rate`, `training.batch_size`, `training.optimizer`) | aceito |
| O nome do modelo (`model.name`, `model.timm_model`) | aceito, se cada valor for um nome e não um caminho ou endereço |
| Um caminho (`data.base_dir`, `model.weights_path`, tudo em `output`) | 422 |
| Um nome dentro do dataset (`data.train_dir`, `data.train_csv`) | 422 |
| Uma seção inteira (`data`, `model`), o `block` ou o próprio espaço de busca | 422 |

A recusa vem antes da fila: o pedido responde 422 com o nome do campo e nada
é treinado. Para comparar datasets ou pesos diferentes, envie um job para cada.

Na classificação, a validação cruzada, as buscas, o transfer learning e a
comparação de modelos vão dentro do próprio pedido de treino (o campo
`block`); todos são aceitos. Os dois valores de `block` que não treinam
(`batch_prediction` e `export_onnx`) respondem 422 até a avaliação ser revista.

Uma estratégia é **um job** na fila, do primeiro ao último treino. Uma busca
longa segura a fila para as outras pessoas até terminar ou ser parada; não há
limite de treinos por job.

## O que já foi testado de verdade

Em 18/11/2026, numa pasta temporária, com o servidor real em
`--server --trusted-proxy 127.0.0.1 --datasets-dir <pasta>` e duas pessoas
simuladas pelo cabeçalho que o proxy enviaria:

- Ana treinou uma classificação de 1 época em CPU (ResNet-18, 48 imagens de
  treino de uma pasta compartilhada); o run terminou em 9 s e foi para a pasta
  dela.
- Bia enviou o dela 40 ms depois: entrou na fila, rodou em seguida e foi para a
  pasta dela. Cada uma vê só o próprio run no histórico.
- Bia não conseguiu ler o status, o resultado nem parar o run da Ana (status
  parado, 404 e 404), e um gráfico do run da Ana respondeu 200 para Ana e 403
  para Bia.
- Um treino apontando para uma pasta fora do alcance foi recusado com 422 e não
  entrou na fila.

*Não verificado:* um proxy de login real na frente, GPU, as outras quatro
tarefas, e mais de duas pessoas ao mesmo tempo.

No mesmo dia, a tela foi aberta num navegador atrás de um proxy de teste que
só acrescenta o cabeçalho da conta: a página abriu sem chamar nenhuma rota
fechada, o campo do dataset listou as duas pastas, o passo Estratégia mostrou
só "Treino simples" nas cinco tarefas, e o detalhe de um run treinado ali
mostrou métricas, gráficos e "Carregar no formulário". *Não verificado:* a
tela em celular e com leitor de tela neste modo.

Em 19/11/2026, no mesmo arranjo (servidor real, proxy de teste, CPU): um grid
search de classificação com dois valores de learning rate rodou os dois
treinos de 1 época, gravou tudo na pasta da pessoa e apareceu no Histórico
como um grupo de 2 runs; o mesmo pedido com um eixo `data.base_dir` e uma
comparação replicada com uma variante `model.weights_path` responderam 422; o
passo Estratégia mostrou as seis opções da classificação. *Não verificado:*
as estratégias das outras quatro tarefas treinando de verdade (só os pedidos
foram testados), Optuna, e uma estratégia parada no meio neste modo.

## Perguntas comuns

**Os runs antigos do perfil "default" aparecem?** Não. No modo servidor
ninguém usa o perfil padrão; cada conta tem a própria pasta. Os runs antigos
continuam no disco, em `outputs/models/`.

**Duas contas parecidas podem cair na mesma pasta?** Não. O nome da pasta
termina com um resumo (hash) do nome completo da conta, então `a.b@lab.org` e
`a-b@lab.org` ficam separadas. Maiúsculas não contam: `Ana@lab.org` e
`ana@lab.org` são a mesma conta.

**Os tokens ficam cifrados?** Ainda não: são arquivos de texto na pasta de
configuração do usuário do sistema que roda o servidor (veja [Tokens](tokens.md)).
Quem administra a máquina consegue lê-los.
