# Auditar dados e preservar um experimento

Uma configuração válida não garante que os dados sejam independentes ou que um
treino retomado seja idêntico a uma execução contínua. Este guia descreve as
checagens disponíveis e os limites de cada uma.

## Antes de treinar: auditar os splits

Na GUI de classificação, abra a seção de dados e use **Auditar imagens**. A
auditoria usa as pastas de treino, validação e teste configuradas no formulário.
Ela só começa quando solicitada; editar os caminhos invalida o resultado exibido.
O painel mostra duplicatas exatas, imagens ilegíveis e splits ausentes/vazios.
O CSV de grupos e a opção de grupos estritos são opcionais.

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

**Limites:** hashes iguais detectam duplicatas exatas, não imagens visualmente
parecidas, recortes ou outra compressão da mesma foto. Um relatório limpo não
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
