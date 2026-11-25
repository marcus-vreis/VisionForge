# Modo servidor — um VisionForge para várias pessoas (experimental)

> **Leia antes de instalar.** O modo servidor está em construção. Nesta versão
> ele identifica quem entrou, lê datasets das pastas de cada pessoa e das
> compartilhadas, **treina um job por vez** nas cinco tarefas, com as
> estratégias (validação cruzada, busca, comparação, réplicas), e mostra a fila,
> o progresso e o histórico de cada um. Um run pronto pode ser retomado, testado
> em outro dataset, explicado com Grad-CAM, usado em lote e exportado para
> ONNX. Cada pessoa pode baixar um dataset, enviar um `.zip` do próprio
> computador, criar o de exemplo e usar o Laboratório de datasets, gravando só
> na própria pasta. Um run pode ser
> baixado (ZIP, Markdown) e publicado no Hugging Face Hub, no Zenodo e no
> GitHub com o token de quem o treinou. Cada pessoa pode mandar os próprios
> treinos ao Weights & Biases e ao Comet, com a própria chave. O Colab, o DVC
> e o MLflow local não estão disponíveis, e a tela não os mostra. E ele é
> para um **grupo que já confia entre
> si**: o treino roda dentro do processo do servidor, sem isolamento entre as
> pessoas. Não ofereça este servidor a desconhecidos.

Esta página sai do código atual (`gui/server_mode.py`) e das decisões ADR-212
a ADR-227 e ADR-246. O que não foi testado está marcado como *não verificado*.

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
- **O botão "datasets" abre "Baixar" e "Explorar".** No download, o
  formulário pede o nome de uma pasta, não um caminho, e mostra para onde o
  dataset vai: uma subpasta da sua pasta de datasets. Não há Kaggle na lista,
  e a chave do Roboflow ou do Hugging Face é digitada a cada download (veja
  "Pôr um dataset no servidor"). No Laboratório, as pastas são digitadas e o
  que ele cria vai para a sua pasta (veja "O Laboratório num servidor").
- **"Criar dataset de exemplo"** grava o exemplo na sua pasta de datasets.
- **As prévias** de aumento e de pré-processamento e a **auditoria** do
  dataset funcionam como no uso local, sobre um dataset das suas pastas ou das
  compartilhadas. As imagens de uma prévia ficam numa pasta do seu perfil
  (`preview_cache`), e cada prévia nova substitui a anterior.
- **Sem Colab**, sem aviso de atualização e sem as abas de tarefas próprias.
- **No Histórico**, um run mostra métricas, gráficos, "Carregar no
  formulário" e o que dá para fazer com o modelo: retomar um run parado,
  testar em outro dataset, Grad-CAM, inferência em lote e exportar ONNX. Onde
  a ação pede uma pasta de imagens, o caminho é digitado e a página lista as
  pastas aceitas, como no campo do dataset. O run também pode ser baixado
  ("Bundle do experimento" e "Markdown") e publicado no Hugging Face Hub, no
  Zenodo e no GitHub; o DVC não aparece (veja "Tirar um run do servidor").
- **Na primeira visita** a página pergunta como chamar você e qual saudação
  prefere. A resposta fica no seu perfil, no servidor: vale em qualquer
  navegador, e "Trocar nome", no menu, muda depois. A conta e a pasta não
  mudam com o nome.
- **O menu do seu nome** mostra a conta com que você entrou e a sua pasta, e
  não há outro perfil para escolher. Em **Integrações** você liga o envio dos
  seus próximos treinos ao Weights & Biases e ao Comet: escolha o projeto e
  salve a sua chave, que fica guardada no servidor, só para o seu perfil, e
  nunca volta para a tela. Se um envio falhar, o run mostra o aviso e o botão
  de reenviar. O MLflow local não aparece: ele grava num arquivo do disco do
  servidor, que você não teria como abrir.
- **Guia:** o tour da interface e o guia do primeiro treino, que usa o
  dataset de exemplo. *Não verificado:* o guia do primeiro treino percorrido
  até o fim num servidor.

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
| `--max-actions <número>` | Quantas ações pesadas rodam ao mesmo tempo fora da fila, somando todas as pessoas. O padrão é 2. Veja "Ações que rodam na hora". |
| `--quota-gb <número>` | O máximo que a pasta de **cada pessoa** pode ocupar, em GB (aceita decimais, como `0.5`). Sem ela não há limite. Veja "Cota de disco". |
| `--max-upload-gb <número>` | O maior `.zip` que uma pessoa pode enviar de uma vez como dataset, em GB. O padrão é 5. Veja "Enviar um dataset (.zip)". |
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
  acesso. O que entra ali vem de um `.zip` enviado do seu computador, de um
  download pela tela, do dataset de exemplo, do Laboratório ou de quem
  administra o servidor (veja "Pôr um dataset no servidor").
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

## Ações que rodam na hora

Um treino espera a vez na fila. Algumas ações não: elas fazem o trabalho
dentro do próprio pedido e respondem quando terminam. Num servidor
compartilhado, no máximo `--max-actions` delas (2, se você não disser) rodam
ao mesmo tempo, somando todas as pessoas:

- testar um run em outro dataset, Grad-CAM, inferência em lote e exportar ONNX;
- baixar um dataset e auditar um dataset;
- publicar no Hugging Face Hub, no Zenodo e no GitHub, e reenviar ao W&B ou
  ao Comet.

Quando o limite está tomado, a próxima recebe o aviso "O servidor está
ocupado com ações de outras pessoas" (erro 429) e nada fica esperando: é só
tentar de novo em instantes. Um treino nunca é recusado por isso.

O que este limite **não** faz: ele não espera um treino terminar. Uma ação
dessas ainda divide o processador ou a GPU com o treino que estiver rodando,
de qualquer pessoa. As operações do Laboratório não entram na conta; elas já
têm o próprio limite, uma por pessoa e por tipo.

## Meus datasets

Em **Datasets → Meus datasets** (aba que só existe num servidor) cada pessoa vê
o que há na própria pasta de datasets, com o tamanho de cada um, e pode apagar
o que não usa mais.

- Apagar pede confirmação na própria linha ("Apagar de vez? Não dá para
  desfazer.") e remove a pasta inteira. Não há lixeira.
- Só aparecem e só podem ser apagadas as pastas que estão diretamente na
  **sua** pasta. As pastas compartilhadas e as de outras pessoas não entram.
- Enquanto você tiver um treino rodando ou na fila, o servidor recusa apagar:
  o treino pode estar lendo o dataset. Espere terminar ou cancele na fila.
- Um run já treinado continua no Histórico depois que o dataset dele é
  apagado, mas as ações que releem o dataset (como retomar o treino) deixam de
  funcionar para ele.

## Enviar um dataset (.zip)

Na mesma aba, **Enviar .zip** manda um dataset do seu computador para a sua
pasta de datasets.

1. Compacte a pasta do dataset num `.zip` (no Windows, "Enviar para → Pasta
   compactada"). Se o `.zip` tiver uma única pasta no topo, ela vira o
   dataset; a pasta `__MACOSX` que o macOS acrescenta é ignorada.
2. Em **Nome da pasta nova**, escreva o nome que o dataset vai ter na sua
   pasta. Vazio, vale o nome do arquivo sem o `.zip`. É um nome só: sem
   barras, sem dois-pontos, sem ponto no começo ou no fim.
3. Clique em **Enviar .zip** e escolha o arquivo. O envio vai em partes de
   8 MB, com uma barra de progresso. No fim o servidor confere e extrai o
   `.zip` ("conferindo e extraindo no servidor…") e o dataset aparece na
   lista. O caminho completo dele é o que vai no campo de dataset da tarefa.

Se a conexão cair no meio, aparece **Retomar envio**: o servidor diz quais
partes já chegaram e só as que faltam vão de novo. Recarregar a página e
escolher o mesmo arquivo com o mesmo nome de pasta também continua de onde
parou. **Cancelar envio** descarta o que já foi.

O que o servidor recusa, sem extrair nada:

- **Pasta que já existe** (erro 409): um dataset seu nunca é substituído.
  Escolha outro nome ou apague o antigo.
- **Arquivo grande demais** (erro 413): acima de `--max-upload-gb` (5 GB se
  quem opera não disser outro valor).
- **Sem espaço** (erro 507), com `--quota-gb`: o tamanho do `.zip` conta desde
  o começo do envio, e antes de extrair o servidor confere se o dataset
  extraído também cabe. Nesse último caso as partes ficam guardadas: libere
  espaço e retome.
- **`.zip` perigoso ou estranho** (erro 422): caminho absoluto ou com `..`,
  link simbólico, arquivo especial, arquivo com senha, nome repetido, mais de
  200.000 arquivos, um arquivo que cresce mais de 100 vezes ao ser extraído
  (bomba de compressão) ou, extraído, mais de 4 vezes `--max-upload-gb`. A
  mensagem diz qual entrada foi recusada, e o envio é descartado.

Cada pessoa tem no máximo 3 envios abertos ao mesmo tempo. Um envio parado por
24 horas é apagado na próxima vez que a mesma pessoa começa outro. Enquanto
dura, o envio fica em `outputs/profiles/<conta>/uploads/`, fora da pasta de
datasets; ninguém mais o vê (para outra conta, a resposta é 404).

Limites que o operador precisa conhecer: a extração roda dentro do pedido que
conclui o envio e não conta em `--max-actions`; um proxy com tempo limite
curto pode cortar a resposta de um `.zip` grande enquanto o servidor termina.
Só `.zip` é aceito. **Ainda não existe** a remoção automática do dataset
enviado ao fim do treino (prevista na fase 1 da ADR-212): ele fica na pasta
até a pessoa apagá-lo.

## Cota de disco

Sem `--quota-gb`, cada pessoa pode encher o disco do servidor. Com
`--quota-gb 20`, por exemplo, a pasta de cada conta pode ocupar até 20 GB.

- **O que conta:** tudo que está na pasta da pessoa (`outputs/profiles/<conta>/`):
  os runs com os modelos, os gráficos, os relatórios e os datasets que ela
  baixou ou criou. As pastas compartilhadas não contam.
- **Quando é conferido:** na hora em que a pessoa pede algo que grava dados
  novos: um treino ou uma estratégia, retomar um run, baixar ou criar um
  dataset, aplicar uma operação do Laboratório, e as ações de um run pronto
  (testar em outro dataset, Grad-CAM, inferência em lote, exportar ONNX). O
  envio de um `.zip` confere a cota com o tamanho dele, no começo e antes de
  extrair (veja "Enviar um dataset (.zip)").
- **O que acontece com a pasta cheia:** esses pedidos são recusados (erro 507)
  com o tamanho usado e o limite. Ler, abrir, baixar, publicar e **apagar**
  runs continuam funcionando.
- **Como liberar espaço:** apagando runs no Histórico ou datasets em
  **Datasets → Meus datasets**; o espaço vale logo em seguida.
- **Onde a pessoa vê:** o menu do nome mostra "disco: 1,5 GB de 20,0 GB", e
  um aviso quando a pasta está cheia.

O limite é conferido **antes** de a ação começar, não enquanto ela grava: um
treino que já está rodando termina mesmo que passe do limite, e o próximo
pedido é que será recusado. A cota não reserva espaço: se o disco do servidor
for menor que a soma das cotas, ele ainda pode encher.

## O que responde hoje

| Função | No modo servidor |
| --- | --- |
| Abrir a página, esquemas dos formulários, informações do sistema | sim |
| Histórico: listar, abrir e apagar os **seus** runs | sim |
| Tokens de publicação (Hugging Face, GitHub, Zenodo, W&B, Comet) do **seu** perfil | sim |
| `GET /api/session` (quem entrou, as pastas de dataset e o que está fechado) | sim |
| Ler um dataset para conferir: divisões, contagem por classe, amostras e miniaturas, nas cinco tarefas | sim, só na **sua** pasta de datasets e nas pastas compartilhadas |
| Baixar dataset (torchvision, Roboflow, Hugging Face) e criar o dataset de exemplo | sim, só para uma subpasta da **sua** pasta de datasets |
| Baixar do Kaggle | não num servidor (o cliente lê a chave de uma variável do processo, que seria de todos) |
| Laboratório de datasets: explorar, conferir, redividir, incorporar imagens, mesclar, exportar e importar COCO | sim; lê as **suas** pastas e as compartilhadas e grava só na sua |
| Prévia de aumento e de pré-processamento, auditoria do dataset | sim; leem as **suas** pastas e as compartilhadas, e as imagens da prévia ficam no **seu** perfil |
| Treinar um run simples (classificação, detecção, segmentação, regressão, anomalia) | sim, com dataset e pesos das **suas** pastas ou das compartilhadas |
| Fila | sim: os **seus** jobs e, dos outros, só quantos estão na frente |
| Progresso, resultado e gráficos de um run | sim, só dos **seus** |
| Estratégias: validação cruzada, busca (grid, random, Optuna), comparação de modelos, réplicas e comparação replicada, nas cinco tarefas | sim, com as mesmas regras do treino simples e a regra dos eixos abaixo |
| Retomar um run parado | sim; o config guardado é conferido de novo, como num treino novo |
| Testar em outro dataset, Grad-CAM, inferência em lote | sim, com imagens das **suas** pastas ou das compartilhadas; o resultado vai para a pasta do run |
| Exportar ONNX | sim, sempre para a pasta do run |
| Baixar o ZIP ou o Markdown do run | sim, só dos **seus** runs |
| Publicar no Hugging Face Hub, no Zenodo e no GitHub | sim, com o **seu** token, salvo no seu perfil |
| Publicar num repositório DVC | não num servidor (roda o git e o DVC do servidor numa pasta do disco dele) |
| Ver e editar o **seu** perfil: nome, saudação e integrações | sim; o perfil de outra pessoa não aparece nem pode ser editado |
| Enviar treinos ao Weights & Biases e ao Comet, e reenviar um run | sim, com a **sua** chave, salva no seu perfil |
| MLflow local (ligar e reenviar) | não num servidor (grava num arquivo do disco dele, que ninguém abriria) |
| Criar outro perfil | não num servidor (o perfil é a conta do login) |
| Listar e apagar os datasets da **sua** pasta ("Meus datasets") | sim; só as pastas diretamente na sua pasta, e não enquanto um treino seu roda ou espera |
| Enviar um dataset do seu computador como `.zip` ("Enviar .zip") | sim, em partes que se retomam, para uma pasta nova da **sua** pasta de datasets |
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
(`batch_prediction` e `export_onnx`) respondem 422: num servidor a inferência
em lote e a exportação são feitas a partir do run, no Histórico.

## Usar um run pronto num servidor

- **Só o dono age sobre o run.** Para outra pessoa o run não existe (404).
- **As imagens vêm das pastas aceitas.** A pasta do teste, do Grad-CAM e da
  inferência em lote precisa estar na sua pasta de datasets ou numa
  compartilhada, com o caminho completo; fora disso a resposta é 422.
- **O que é gravado fica na pasta do run.** Num servidor não dá para escolher
  outro destino para o CSV das predições nem para o arquivo ONNX: um pedido
  com caminho de saída responde 422.
- **O checkpoint é o do próprio run.** Se o registro do run apontar para um
  arquivo fora da sua pasta, a ação é recusada (409) e nada é carregado.
- **Retomar** confere o config guardado como um treino novo: se o dataset saiu
  das pastas aceitas desde então, a resposta é 422.

Essas ações (menos retomar, que entra na fila) rodam na hora, fora da fila, e
usam a CPU ou a GPU junto com o treino que estiver em andamento. Num servidor
com uma GPU só, um Grad-CAM ou um teste grande durante o treino de outra pessoa
disputa memória com ele.

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

No mesmo dia e arranjo, sobre um run de classificação treinado ali: o teste em
outra pasta (16 imagens de uma pasta compartilhada) gravou métricas e
gráficos no run; o Grad-CAM gerou 2 mapas; a inferência em lote gravou o CSV
de 16 predições na pasta do run; a exportação ONNX gravou o arquivo na pasta
do run e a validação contra o PyTorch passou. Um teste apontando para
`C:/Windows` e pedidos com caminho de saída para o CSV e para o ONNX
responderam 422. *Não verificado:* retomar um run parado de verdade (só o
pedido foi testado), e essas ações nas outras quatro tarefas neste modo.

Ainda em 19/11/2026: o dataset de exemplo da regressão foi criado na pasta da
pessoa (201, e 200 na segunda vez, sem regravar); um download do torchvision
(MNIST, 2 imagens por classe) gravou 40 imagens numa subpasta dela e a
contagem por classe leu de volta; downloads para a pasta compartilhada e para
a raiz da pasta própria responderam 422, o do Kaggle 400 com a explicação, e
o do Roboflow sem chave 400. *Não verificado:* um download real do Roboflow
ou do Hugging Face neste modo (só com os clientes simulados nos testes).

No Laboratório, no mesmo dia e arranjo: um dataset compartilhado de 80 imagens
foi aberto, conferido e redividido; o resultado foi gravado na pasta da
pessoa com 80 arquivos copiados e nenhum ligado, e a contagem por classe leu
de volta; um dataset fora das pastas aceitas respondeu 422. *Não verificado:*
incorporar, mesclar e as conversões COCO rodando de verdade neste modo (os
pedidos e a regra de destino são os mesmos e estão cobertos por testes; só o
redividir foi executado), e duas pessoas usando o Laboratório ao mesmo tempo
fora dos testes.

Sobre um run de classificação treinado ali, no mesmo dia e arranjo: o ZIP foi
baixado (41,5 MB), o Markdown também, e as prévias do Hugging Face Hub, do
Zenodo e do GitHub listaram os arquivos do run sem contatar nenhum serviço;
um envio ao Hub sem token salvo respondeu 409 e a prévia do DVC 404. *Não
verificado:* um envio real ao Hub, ao Zenodo ou ao GitHub neste modo (nos
testes o envio é simulado e confere de quem é o token usado).

Prévias e auditoria, no mesmo dia e arranjo, sobre um dataset compartilhado
de 80 imagens: a prévia de aumento gravou 2 variantes e a de pré-processamento
o resultado de um filtro, as duas dentro do perfil da pessoa, e as imagens
foram servidas à página (200); a auditoria contou 48, 16 e 16 imagens nas
três divisões; uma auditoria de `C:\Windows` respondeu 422. *Não verificado:*
as prévias vistas nos painéis da tela neste modo (os pedidos foram feitos
direto às rotas).

Perfil e integrações, no mesmo dia e arranjo: a lista de perfis trouxe só o
da conta; a primeira visita perguntou o nome e a saudação, e a resposta ficou
gravada no perfil; o menu mostrou Weights & Biases e Comet, sem MLflow, e
ligar e desligar o W&B foi salvo no perfil. Editar o perfil `default`
respondeu 404, ligar o MLflow 422 e criar um perfil 404. *Não verificado:* um
envio real ao W&B ou ao Comet neste modo, que pede uma chave de verdade (nos
testes o W&B é simulado e confere de quem é a chave usada no reenvio; o do
Comet só confere quem alcança a rota).

**Teste de navegador a cada mudança.** A suíte Playwright do projeto sobe um
servidor de verdade neste modo e repete, com duas contas no mesmo servidor, o
caminho principal: um pedido sem a conta é recusado; a primeira visita
pergunta o nome e ele fica no perfil; o menu traz W&B e Comet e não traz
MLflow nem outro perfil; uma pessoa treina uma época no dataset de exemplo da
própria pasta e acha o run no Histórico; a outra conta vê o Histórico vazio e
recebe 404 ao pedir esse run. Nesses testes o navegador faz o papel do proxy
de login (ele mesmo manda o cabeçalho da conta), então eles **não** provam a
configuração de um proxy real.

Cota de disco, em 2026-11-20, com `--quota-gb 0.1` e uma pasta de 173 MB: o
menu mostrou "disco: 0,2 GB de 0,1 GB" e o aviso de pasta cheia; pedir o
dataset de exemplo mostrou na tela a recusa com os dois tamanhos; um treino
foi recusado com 507; listar os runs e ler um dataset compartilhado
continuaram respondendo. *Não verificado:* liberar espaço apagando um run
pela tela (os testes automáticos cobrem pela rota) e uma pasta com dezenas de
milhares de arquivos, onde medir demora mais.

Limite de ações, em 2026-11-20: nos testes automáticos, com o app real e
`--max-actions 1`, uma auditoria segurada no meio fez a auditoria e o Grad-CAM
de outra conta responderem 429, enquanto listar runs e ler um dataset
seguiram respondendo; ao terminar, a vaga voltou. Num servidor de verdade
com `--max-actions 1`, 8 auditorias disparadas juntas tiveram 1 resposta 200 e
7 respostas 429 com o aviso, e uma auditoria logo depois respondeu 200. *Não
verificado:* o aviso visto na tela de uma ação, e quantas dessas ações uma
máquina aguenta (2 é um chute para uma GPU).

Meus datasets, em 2026-11-20: a suíte Playwright abre a aba num servidor de
verdade, vê o dataset de exemplo criado no treino, cancela na primeira
pergunta (a pasta fica), confirma na segunda e confere que a pasta sumiu. A
aba foi vista também à mão, com três datasets listados. *Não verificado:* a
recusa com um treino na fila vista na tela (os testes automáticos cobrem pela
rota) e apagar um dataset grande, de vários GB.

Enviar um `.zip`, em 2026-11-25: a suíte Playwright, num servidor de verdade no
Windows, escolhe um `.zip` pequeno, derruba a primeira parte no caminho, clica
em **Retomar envio** e confere que o dataset aparece na lista com o tamanho
certo. Os testes da rota, com a aplicação real, cobrem a parte que faltou, a
conta alheia (404), a cota (507), a pasta existente (409), o caminho com `..`,
o link simbólico e a bomba de compressão. *Não verificado:* um envio de vários
GB (tempo de envio e de extração), atrás de um proxy de verdade, e a
checagem de links e de ZIP64 no Linux.

## Tirar um run do servidor

- **Baixar.** "Bundle do experimento" baixa um ZIP com o registro do run, os
  pesos e os gráficos; "Markdown" baixa o cartão do modelo. Os dois saem pelo
  navegador, direto para o seu computador.
- **Publicar.** Hugging Face Hub, Zenodo e GitHub funcionam como no uso local:
  você salva o token no painel de cada um, confere a prévia e envia. O token
  fica guardado no servidor, separado por perfil, e só é usado nos envios do
  seu perfil. Um envio sem token salvo responde 409; ninguém usa o token de
  outra pessoa.
- **Só o dono.** Para outra pessoa o run não existe (404), em todas essas
  ações.
- **O que entra no pacote.** Só arquivos de dentro da pasta do run. Um
  arquivo ligado (link) ou fora dela é recusado, como no uso local.
- **DVC.** Desligado num servidor: ele roda o git e o DVC instalados no
  servidor sobre uma pasta do disco do servidor, com a identidade de quem
  opera a máquina. Baixe o ZIP e versione no seu computador.

O cartão em Markdown e o `run.json` do ZIP trazem os caminhos do servidor
(a pasta do run, a do dataset). Quem publica o arquivo publica esses caminhos;
os pacotes do Hub, do Zenodo e do GitHub levam um registro público, sem
esses caminhos.

## Pôr um dataset no servidor

Há quatro caminhos:

1. **O operador compartilha uma pasta** com `--datasets-dir`. Todos leem,
   ninguém grava pela tela.
2. **Baixar pela tela** (botão "datasets"). O dataset vai para
   `outputs/profiles/<seu perfil>/datasets/<nome da pasta>`. Só você lê essa
   pasta.
3. **"Criar dataset de exemplo"**, no campo do dataset de cada tarefa: um
   conjunto pequeno e sintético, gravado na mesma pasta, para um primeiro
   treino.
4. **Enviar um `.zip` do seu computador**, em **Datasets → Meus datasets**.
   Vai para a mesma pasta (veja "Enviar um dataset (.zip)").

Regras do download num servidor:

- **Destino.** Sempre uma subpasta da sua pasta de datasets. Um pedido para
  outra pasta (uma compartilhada, a de outra pessoa, a raiz da sua) responde
  422 e nada é baixado.
- **Chaves.** As chaves salvas na máquina e as variáveis de ambiente do
  servidor são do operador e não são usadas. A chave do Roboflow e o token do
  Hugging Face vão no próprio pedido, valem para aquele download e não são
  guardados. Sem token, o Hugging Face baixa só datasets públicos.
- **Kaggle.** Fechado: o cliente do Kaggle só lê a chave de uma variável do
  processo, e pôr a sua ali a entregaria aos downloads das outras pessoas.
  Baixe no seu computador e peça ao operador para compartilhar a pasta.

Limites que o operador precisa conhecer: o download roda na hora, fora da
fila, e conta em `--max-actions` (veja "Ações que rodam na hora"); o que ele
grava conta na cota de disco, quando o servidor tem uma (veja "Cota de
disco"). O cache do Hugging Face fica na conta com que o servidor roda.

## O Laboratório num servidor

O Laboratório lê um dataset e, quando cria outro (redividir, incorporar
imagens, mesclar, exportar ou importar COCO), nunca muda o original.

- **O que ele lê.** Um dataset da sua pasta ou de uma pasta compartilhada. O
  mesmo vale para a segunda pasta de cada operação: o outro dataset da mescla,
  as imagens novas, a pasta COCO. Fora disso a resposta é 422.
- **Onde ele grava.** Se o dataset aberto é seu, o resultado fica ao lado
  dele, dentro da sua pasta, como no uso local. Se é compartilhado, o
  resultado vai para a sua pasta de datasets e é uma **cópia**: nenhum arquivo
  seu fica ligado (hard link) a um arquivo da pasta compartilhada, que é do
  operador. Uma cópia ocupa o tamanho do dataset no disco.
- **Os trabalhos são seus.** Uma análise, uma conferência ou uma gravação em
  andamento só responde a quem a iniciou; para outra pessoa ela não existe
  (404). Cada pessoa roda um trabalho de cada tipo por vez, e o de uma não
  bloqueia o de outra.
- **O nome sugerido** é calculado ao lado do dataset original. Para um dataset
  compartilhado, o servidor confere o nome contra a sua pasta na hora de
  gravar e responde 409 se ele já existir ali; troque o nome.

Limite que o operador precisa conhecer: os trabalhos do Laboratório rodam na
hora, fora da fila, um por pessoa e por tipo. O que elas gravam conta na cota
de disco, quando o servidor tem uma (veja "Cota de disco").

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
