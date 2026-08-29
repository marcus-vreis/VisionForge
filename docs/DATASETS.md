# Datasets — como obter e como validar cada provedor

A aba **⤓ DATASETS** (barra inferior) baixa um dataset uma vez para uma pasta
local; depois é só apontar o campo de dataset de qualquer task para ela. São
quatro provedores, e eles diferem no que exigem de você.

| Provedor | Credencial | Serve para |
|---|---|---|
| **torchvision** | nenhuma | classificação (`ImageFolder`) |
| **Roboflow** | API key | detecção (layout YOLO) e classificação |
| **Kaggle** | `kaggle.json` ou variáveis de ambiente | qualquer layout — vem como o autor publicou |
| **Hugging Face** | token só se o dataset for privado | classificação (`ImageFolder`) |

Os quatro já vêm instalados (ADR-106) — não há extra para adicionar. O que
separa um provedor do outro é só a credencial.

## A chave se digita uma vez

Ao lado de cada campo de credencial há um botão **💾 Salvar**. Salvando, a
chave fica guardada em `~/.visionforge/credentials.json` e nas próximas vezes o
campo já aparece marcado como *salva*, mostrando só os últimos caracteres
(`salva: •••••••1234`) — o suficiente para você saber qual é, sem a chave
aparecer na tela.

- deixe o campo **em branco** para usar a salva
- digite outra e **↻ Substituir** para trocar
- **Esquecer** remove a chave deste computador

Um valor digitado no momento sempre vence o salvo, então dá para usar uma chave
avulsa sem perder a sua.

---

## 1. torchvision — o único sem credencial

O caminho mais curto para ter algo treinável em minutos. Cinco datasets
embutidos: CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST, KMNIST.

**Na GUI:** ⤓ DATASETS → provedor `torchvision` → escolha o dataset → pasta de
saída → ▶ BAIXAR. "Limite por classe" é opcional e serve para pegar uma
amostra rápida em vez do dataset inteiro.

**Validando pela linha de comando:**

```bash
python -c "from visionforge.gui.api.dataset_download import download_dataset; print(download_dataset(provider='torchvision', dataset='mnist', out_dir='datasets/mnist', limit=20))"
```

Esperado: `total_images=400`, `splits={'train': 320, 'val': 80, 'test': 100}`.

O `val` não vem do torchvision — ele entrega só train/test. O VisionForge
recorta 20% do treino, estratificado por classe e reprodutível entre execuções,
porque toda task espera os três splits. Para manter os dois originais, passe
`val_fraction=0`.

**Layout resultante**, pronto para o painel de Classificação:

```
datasets/mnist/
├── train/0_-_zero/…  1_-_one/…  …
├── val/  0_-_zero/…  1_-_one/…  …
└── test/ 0_-_zero/…  1_-_one/…  …
```

---

## 2. Roboflow — detecção com anotações

**Credencial:** uma API key da sua conta em `app.roboflow.com` → Settings →
API Keys.

**O que informar:**

- **Dataset**: `workspace/project` (está na URL do projeto)
- **Versão**: o número da versão do export (obrigatório)
- **Formato**: `yolov8` para detecção; `folder` para classificação

**Validando:**

```bash
python -c "from visionforge.gui.api.dataset_download import download_dataset; print(download_dataset(provider='roboflow', dataset='meu-workspace/meu-projeto', version=1, dataset_format='yolov8', api_key='SUA_CHAVE', out_dir='datasets/meu-projeto'))"
```

Esperado: uma pasta com `data.yaml` e `train/val/test`, cada um com `images/` e
`labels/` — o layout que o painel de Detecção lê. Aponte **Pasta base** para a
raiz.

Erros comuns e o que significam: `Roboflow requires an api_key` e
`Roboflow requires a version number` são validações locais, antes de qualquer
rede; `Roboflow dataset must be 'workspace/project'` quer dizer que faltou a
barra.

---

## 3. Kaggle — o mais amplo, e o menos padronizado

**Credencial:** em `kaggle.com` → seu perfil → Settings → API → *Create New
Token*. Baixa um `kaggle.json`. Coloque-o em:

- **Windows:** `C:\Users\<você>\.kaggle\kaggle.json`
- **Linux/macOS:** `~/.kaggle/kaggle.json` (e `chmod 600`)

Ou, sem arquivo, exporte as variáveis:

```bash
$env:KAGGLE_USERNAME = "seu-usuario"; $env:KAGGLE_KEY = "sua-chave"
```

**O que informar:** `owner/dataset-slug`, que é o final da URL do dataset.

**Validando:**

```bash
python -c "from visionforge.gui.api.dataset_download import download_dataset; print(download_dataset(provider='kaggle', dataset='owner/dataset-slug', out_dir='datasets/kaggle-ds'))"
```

O download baixa e descompacta. **O layout é o que o autor publicou** — o
Kaggle não impõe padrão, então pode vir sem `val`, com nomes de pasta em outra
língua, ou com tudo solto numa pasta só. Depois de baixar, abra a pasta e
confira antes de treinar; o seletor de dataset da GUI mostra o que encontrou e
diz o que está faltando.

Se as credenciais não estiverem no lugar, o erro é explícito sobre onde pôr o
`kaggle.json`.

---

## 4. Hugging Face — datasets de classificação

**Credencial:** nenhuma para datasets públicos. Para um privado, um token de
`huggingface.co/settings/tokens`.

**O que informar:** o id do dataset, ex.: `cifar10`, `beans`,
`food101`.

**Validando:**

```bash
python -c "from visionforge.gui.api.dataset_download import download_dataset; print(download_dataset(provider='huggingface', dataset='beans', out_dir='datasets/beans'))"
```

Todos os splits que o dataset publicar são materializados como
`<out>/<split>/<classe>/*.png`.

**Limite honesto:** só funciona com datasets que expõem uma coluna de imagem e
uma de rótulo. Um dataset de detecção ou de texto no HF não tem como virar
`ImageFolder`, e o erro diz isso —
`has no image+label features to materialize into an ImageFolder`.

---

## Depois de baixar, em qualquer provedor

1. Abra a task correspondente e aponte **Pasta base** para a raiz baixada.
2. O seletor detecta os splits e mostra as estatísticas por split (contagem por
   classe, balanceamento). Se faltar algum, ele diz qual e você escolhe a pasta
   na mão.
3. Treine.

Se quiser conferir que a instalação inteira está sã antes de gastar tempo com
dados reais:

```bash
visionforge selftest --quick
```

Isso gera dados sintéticos, treina uma vez cada task pela mesma API que o
navegador usa e diz o que passou. Não mede qualidade de modelo — mede se o
sistema funciona.
