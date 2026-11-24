# Auditar dados e preservar um experimento

Uma configuração válida não garante que os dados sejam independentes ou que um
treino retomado seja idêntico a uma execução contínua. Este guia descreve as
checagens disponíveis e os limites de cada uma.

## Antes de treinar: auditar os splits

Na GUI de classificação, abra a seção de dados e use **Auditar imagens**. A
auditoria usa as pastas de treino, validação e teste configuradas no formulário.
Ela só começa quando solicitada; editar os caminhos invalida o resultado exibido.
O painel mostra duplicatas exatas, imagens ilegíveis e splits ausentes/vazios.
O CSV de grupos, a opção de grupos estritos e a busca por quase-duplicatas são
opcionais.

```bash
visionforge audit-data datasets/meu_dataset
visionforge audit-data datasets/meu_dataset --json auditoria.json
```

O comando lê imagens em `train`, `val` e `test`, verifica arquivos inválidos e
compara SHA-256 para encontrar imagens com os mesmos bytes em splits diferentes.
Não altera o dataset. Diretórios ausentes ou sem imagens válidas são achados.

Para outro layout, declare todos os splits que deseja comparar:

```bash
visionforge audit-data datasets/meu_dataset --split train=treino --split val=validacao --split test=teste
```

O relatório JSON vai para stdout. `--json` grava também um arquivo novo e recusa
sobrescrever um existente. Os códigos de saída são `0` (nenhum achado), `1`
(achados) e `2` (pedido inválido ou erro de leitura da entrada).

### Quase-duplicatas

```bash
visionforge audit-data datasets/meu_dataset --near-duplicates
```

`--near-duplicates` (no painel, "Procurar quase-duplicatas entre splits") abre
cada imagem e calcula um hash visual de 64 bits (dHash: a imagem em cinza a
9x8 pixels, um bit por par de vizinhos). Duas imagens em splits diferentes cujos
hashes diferem em até 6 bits entram no mesmo grupo, em `near_duplicate_groups`:
é a mesma foto reencodada (PNG e JPEG), redimensionada ou com o brilho mudado,
que vaza o teste para o treino. Pares no mesmo split não são achados; cópias
byte a byte ficam só em `duplicate_groups`. Imagens de uma cor só não têm
estrutura para comparar e ficam de fora. Uma imagem que não decodifica conta
como ilegível. Demora mais que a busca exata, porque decodifica todas as
imagens (ADR-228).

**Limites:** o hash exato só acha os mesmos bytes; o visual não acha recortes,
rotações, espelhamentos nem fotos diferentes do mesmo objeto, e pode juntar
imagens distintas muito parecidas (confira os caminhos). Um relatório limpo não
comprova ausência de toda forma de vazamento. Links de diretório não são seguidos;
links que escapam da raiz são reportados.

## Amostras relacionadas: declarar grupos

Fotos da mesma planta, paciente ou peça podem ter conteúdo diferente e ainda
ser dependentes. Declare essa relação em um CSV UTF-8 com as colunas `path,group`:

```csv
path,group
train/classe_a/foto1.png,planta_001
train/classe_a/foto2.png,planta_001
train/classe_b/foto3.png,planta_002
val/classe_a/foto4.png,planta_003
test/classe_b/foto5.png,planta_004
```

Os caminhos usam `/` e são relativos à raiz do dataset. O exemplo é ilustrativo:
o arquivo real precisa cobrir todas as imagens auditadas, sem repetir caminhos,
com grupos não vazios e sem imagens desconhecidas.

```bash
visionforge audit-data datasets/meu_dataset --groups groups.csv --strict-groups
```

Sem `--strict-groups`, grupos compartilhados aparecem no relatório como
informação. Com a opção, tornam a auditoria inválida. A identidade do grupo é
fornecida pelo pesquisador; o programa não a infere do nome do arquivo.

Na classificação, o mesmo CSV pode orientar o K-fold:

```yaml
block: cross_validation
cross_validation:
  n_folds: 5
  stratified: true
  shuffle: true
  group_manifest: groups.csv
```

Esse fragmento complementa uma configuração de classificação existente. Todas
as imagens do pool de treino precisam ter um grupo; linhas válidas para val/test
podem permanecer no CSV. É necessário ter pelo menos tantos grupos distintos
quanto folds. A divisão mantém cada grupo em apenas um lado de cada fold.

`stratified: true` usa `StratifiedGroupKFold`; `false`, `GroupKFold`. Com
scikit-learn anterior a 1.6, `GroupKFold` exige `shuffle: false`; a aplicação
explica essa restrição antes de construir o modelo. Sem `group_manifest`, o
comportamento anterior do K-fold continua disponível. Esta opção é de CV de
classificação, não uma promessa de CV por grupos para todas as tarefas.

Na GUI de classificação, o formulário de K-fold também aceita o caminho do CSV.
O valor acompanha a configuração ao salvar/exportar YAML.

## Mesma seed, mesmos números

Com `training.deterministic: true` (o padrão), cada treino semeia o Python, o
NumPy e o PyTorch com a `training.seed`. Também liga os kernels determinísticos
do cuDNN e do próprio PyTorch (ADR-199). Em 12/11/2026, numa RTX 5060 Ti com
torch 2.14.1+cu130, duas execuções de cada tarefa (uma época, dados do
`selftest`) repetiram as mesmas métricas.

Duas operações não têm versão determinística na GPU e geram um aviso do PyTorch
no log: a cross-entropy da segmentação (`nll_loss2d`) e o backward do
`roi_align` da detecção torchvision. Nesses dois casos, a repetição observada
não é garantida. Os números também podem mudar entre GPUs, versões do
PyTorch/CUDA e com mais de um worker no DataLoader.

## Retomar um treino

O estado de retomada formato 2 preserva modelo, optimizer, scheduler, scaler,
histórico, melhor época, patience e os estados aleatórios. Arquivos formato 1
continuam legíveis com um aviso sobre a ausência de continuidade aleatória.

Uma retomada solicitada exige um checkpoint utilizável. Arquivo ausente,
ilegível ou incompatível interrompe o pedido antes de preparar o modelo; não
inicia outro treino automaticamente. A consulta ao histórico continua tolerando
arquivos antigos ou danificados. PatchCore não oferece retomada por épocas.

A equivalência foi testada em CPU, no mesmo ambiente, com `num_workers=0`, para
os loops nativos de classificação, regressão, segmentação, autoencoder e detecção
torchvision. A quantidade de épocas e a política de early stopping são mantidas.
Não foi validada equivalência bit a bit em GPU, entre ambientes diferentes ou
com múltiplos workers/prefetch. O backend Ultralytics usa sua própria retomada.

Se o fingerprint registrado comprovar que o dataset mudou, a retomada é recusada.
Mantenha datasets e outputs em pastas separadas: quando outputs estão dentro da
raiz dos dados, o manifesto inclui os próprios artefatos variáveis do treino e
a verificação avisa que a identidade não pode ser comprovada. O método `manifest`
também não detecta alterações de conteúdo que preservem tamanho e caminho.

## Exportar e verificar os registros

No detalhe de um run no histórico, **Baixar bundle do experimento** baixa esse
registro. O download respeita o perfil ativo. Um run em execução ou aguardando
retomada precisa terminar antes da exportação.

Copie o caminho da pasta do run no histórico e substitua `CAMINHO_DO_RUN`:

```bash
visionforge bundle CAMINHO_DO_RUN evidencia.zip
visionforge verify-bundle evidencia.zip
```

O ZIP inclui `run.json` e artefatos de configuração, histórico, predições e
pesos explicitamente selecionados dentro da pasta do run. Um manifesto registra
tamanho e SHA-256; ausências e exclusões ficam anotadas. A exportação recusa um
destino já existente e publica o arquivo completo de forma atômica.

O bundle é um registro verificável: não inclui o dataset, não instala bibliotecas
e não executa arquivos. Gráficos e relatórios fora da seleção são registrados
como excluídos. Verificar hashes detecta alterações em relação ao manifesto,
mas não comprova autoria nem substitui uma assinatura digital ou validação dos
resultados. A verificação não extrai o ZIP.

O ambiente do run registra versões adicionais das bibliotecas e, em um checkout
Git reconhecido do pacote, a revisão e o indicador de alterações locais. Uma
instalação sem esse checkout informa `unknown`; o repositório do pesquisador não
é usado como se fosse o código-fonte do VisionForge.

## Reproduzir um bundle

Quem recebeu o ZIP pode retreinar uma semente dele e comparar o resultado com o
que foi registrado:

```bash
visionforge reproduce evidencia.zip --data PASTA_DO_DATASET
visionforge reproduce evidencia.zip --seed 1 --output saida_repro
```

O comando verifica o bundle (como `verify-bundle`), lê o `run.json` e a
configuração direto do ZIP, sem extraí-lo, e treina pelo mesmo caminho de
`visionforge run`. Funciona com um run único e com um conjunto de réplicas;
K-fold, sweep e comparação pareada ainda não são suportados.

- **Semente:** num run único, a semente registrada (ou `--seed`). Num conjunto
  de réplicas, `--seed` precisa ser uma das sementes do conjunto; sem ela, vale
  a primeira semente que terminou.
- **Dataset:** `--data` substitui o caminho registrado (na detecção, um arquivo
  vira `data_yaml` e uma pasta vira `base_dir`). Sem `--data`, o caminho
  registrado é usado se existir nesta máquina; um bundle público guarda só o
  nome final da pasta, então nesse caso o comando pede `--data`. Nada é baixado.
- **Saída:** um run novo chamado `<nome>_repro` (`<nome>_s<semente>_repro` nas
  réplicas), em `--output` ou na pasta `outputs` de sempre. O run original não
  é tocado.

A saída padrão é um único JSON com `bundle`, `kind`, `seed`, `run_dir`,
`environment_diff` (versões de Python, torch, torchvision e VisionForge e o
dispositivo, registrados e atuais) e, por métrica, `recorded`, `reproduced`,
`delta`, `ci95`, `pi95` e `inside`. As métricas são as finais do teste (ou da
validação, quando a tarefa não tem teste). Nas réplicas, `recorded` é o valor
da mesma semente.

`ci95` é o intervalo t de 95% registrado para a **média** das sementes
(meia-largura t·s/√n) e aparece só como contexto: uma semente sozinha cai fora
dele com frequência bem maior que 5%, sem que nada esteja errado. O julgamento
(`inside`) usa `pi95`, o intervalo de **predição** de 95% para uma semente nova,
`média ± t(0,975; n−1) · desvio · √(1 + 1/n)`, calculado da `n`, da média e do
desvio registrados (n ≥ 2) e cortado ao domínio da métrica como o `ci95`.
`verdict` vale `inside_interval`, `outside_interval` ou `no_interval` (run
único, ou réplicas sem `pi95`). Códigos de saída: 0 dentro do intervalo de
predição ou sem intervalo, 1 fora dele ou treino com falha, 2 entrada inválida
(bundle, tipo, semente ou dataset), antes de qualquer treino. Os logs do treino
vão para a saída de erro.
