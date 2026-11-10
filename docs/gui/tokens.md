# Tokens e chaves — onde ficam, como remover e o escopo mínimo

O VisionForge guarda credenciais **neste computador**, em texto puro, numa
pasta do seu usuário. Esta página diz onde, como apagar e o que pedir ao
criar cada uma. Tudo aqui sai do código atual (`utils/credentials.py`,
`gui/api/profile_tokens.py`, `gui/api/dataset_download.py`); o que o código não
define está marcado como *não verificado*.

## Onde ficam

A pasta base é `~/.visionforge/`. A variável de ambiente `VISIONFORGE_HOME`
troca essa pasta (útil para isolar testes ou apontar para outro disco).

| O quê | Arquivo | Escopo |
|---|---|---|
| Chaves de datasets: Roboflow, Kaggle, Hugging Face (download) | `credentials.json` | por usuário, não por perfil |
| Token do Hugging Face Hub (exportar modelo) | `hub-tokens/<sha256>` | por perfil |
| Chave do Weights & Biases | `wandb-tokens/<sha256>` | por perfil |
| Token do Zenodo | `zenodo-tokens/<sha256>` | por perfil |
| Token do Zenodo Sandbox | `zenodo-sandbox-tokens/<sha256>` | por perfil |
| Token do GitHub | `github-tokens/<sha256>` | por perfil |

- `credentials.json` é um JSON com uma entrada por provedor (`roboflow`,
  `kaggle`, `huggingface`). A GUI só mostra os últimos quatro caracteres; o
  valor completo não volta para a tela.
- Os arquivos `<serviço>-tokens/<sha256>` guardam só o token, sem JSON. O nome é
  o SHA-256 do slug do perfil junto com o caminho da pasta de modelos dele.
  Consequência: se a pasta de modelos do perfil mudar de caminho, o token deixa
  de ser encontrado (o painel passa a mostrar "sem token") e é preciso salvar
  de novo.
- A API nunca devolve o token; ela só informa se existe um salvo.
- O login do Google Colab (feito pelo CLI do Colab) fica em
  `colab/<slug>/.config/colab-cli/token.json`, também sob a pasta base.

## Texto puro e permissões

Nenhum desses arquivos é criptografado.

- **Linux e macOS:** `credentials.json` recebe `chmod 600` (só o dono lê e
  escreve). A pasta `<serviço>-tokens/` é criada com modo `700` e o token é
  gravado num arquivo temporário privado, depois trocado de lugar.
- **Windows:** o `chmod` do código é um *no-op*. O arquivo herda a ACL da pasta
  do seu usuário, e não há proteção além dela.
- Nunca copie `~/.visionforge/` para um repositório git, um drive sincronizado
  ou um backup compartilhado.

## Variáveis de ambiente

| Variável | Uso |
|---|---|
| `VISIONFORGE_HOME` | pasta base das credenciais, no lugar de `~/.visionforge/` |
| `KAGGLE_API_TOKEN` | token do Kaggle. Se já estiver definida, vale ela; senão a chave salva é copiada para ela, no processo do servidor. O cliente do Kaggle também aceita `~/.kaggle/access_token` |
| `HF_TOKEN` e `HUGGING_FACE_HUB_TOKEN` | o cliente do Hugging Face as usa quando o download não leva token. O VisionForge não as lê; só as inclui entre os segredos mascarados nas mensagens de erro |

A **exportação** para o Hugging Face Hub não herda `HF_TOKEN` nem o login do
ambiente: usa só o token salvo no perfil. O Roboflow não tem variável de
ambiente no código; a chave vem do campo ou da chave salva.

## Como remover

Remover apaga a cópia **neste computador**. Não revoga nada na conta do
serviço.

- **Chaves de datasets** (aba DATASETS): botão **Esquecer** no provedor, ou
  apague a entrada em `credentials.json` (ou o arquivo todo).
- **Tokens por perfil** (Hub, W&B, Zenodo, GitHub): botão de remover o token
  no painel do serviço. Remove só o token do perfil atual, naquele serviço.
  Também dá para apagar o arquivo em `<serviço>-tokens/`.
- **Revogar:** para o token deixar de funcionar de verdade, apague-o na conta
  do provedor. Se um token vazou, revogue primeiro e só depois remova a cópia
  local.

## Escopo mínimo por serviço

Crie sempre um token novo, dedicado ao VisionForge. A tabela traz só o que as
dicas da GUI (`tokenHint` em `frontend/src/i18n/pt.ts`) e os guias do
repositório pedem.

| Serviço | O que pedir | Fonte |
|---|---|---|
| **Zenodo / Zenodo Sandbox** | Personal access token só com o escopo `deposit:write`. Sem `deposit:actions` o token não consegue publicar | dica da GUI |
| **GitHub** | Fine-grained token, só para o repositório de experimentos, com a permissão **Contents: leitura e escrita** | dica da GUI |
| **Hugging Face (exportar modelo)** | Token de escrita com o menor escopo possível, no perfil selecionado | [`hugging-face-hub.md`](hugging-face-hub.md) |
| **Hugging Face (baixar dataset)** | Opcional, só para datasets privados. *Não verificado:* o código não define o escopo mínimo | dica da GUI |
| **Kaggle** | Token de API (`KGAT_...`), em Settings → API → Create New Token. O formato antigo `usuário:chave` é recusado. *Não verificado:* o código não define escopo | dica da GUI |
| **Roboflow** | API key, em Settings → API Keys. *Não verificado:* o código não define escopo | dica da GUI |
| **Weights & Biases** | Chave criada nas configurações da conta. *Não verificado:* o código não define escopo | dica da GUI |

Nos itens *não verificado*, consulte a documentação oficial do provedor ao
criar o token; este guia não afirma o que ela diz.

## Servidor exposto na rede

Se o servidor aceita conexões de outras máquinas (`visionforge gui --host`
com um endereço que não seja o da própria máquina), os painéis dos tokens por
perfil mostram um aviso: **perfis não são controle de acesso**. Quem alcança o
servidor escolhe qualquer perfil e pode enviar com o token dele, substituí-lo
ou removê-lo. As chaves de datasets em `credentials.json` nem são por perfil:
valem para qualquer pessoa que use o servidor. Em rede compartilhada, prefira
HTTPS, ou mantenha o servidor só em `localhost`.
