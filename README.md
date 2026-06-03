# VisionForge

VisionForge é um ambiente modular e extensível para experimentação em Visão Computacional, com foco em redes neurais convolucionais (CNNs). Substitua notebooks Jupyter ad-hoc por um sistema limpo, testável e reprodutível para treinar, validar e comparar modelos.

## Recursos atuais (classificação)

- **Modelos**: ResNet 18/34/50/101, EfficientNet B1/B7, VGG 16/19, AlexNet — com pesos ImageNet, pesos custom (`.pth`/`.pt`) ou inicialização aleatória.
- **Pipeline de pré-processamento configurável**: blur gaussiano/mediano, unsharp mask, edges (Sobel), emboss, grayscale, equalize (CLAHE), autocontrast, wavelet Haar — com preview lado a lado de cada etapa.
- **Augmentation**: flip horizontal, rotação, color jitter, normalização customizável.
- **Treinamento**: early stopping, AMP (mixed precision), schedulers (none/cosine/step/plateau), DataParallel multi-GPU, seleção explícita de device.
- **Cross-validation**: K-Fold e Stratified K-Fold com normalize_mean/std recalculados por fold (sem data leakage).
- **Avaliação**: Accuracy, F1, Precision, Recall, AUC-ROC, matriz de confusão (raw + normalizada), curvas ROC e Precision-Recall.
- **Histórico de runs**: lista navegável com badges (preprocessing usado, device, métricas), comparação multi-run com diff de configuração destacado, teste do checkpoint salvo em datasets novos.
- **Reprodutibilidade**: YAML import/export com validação client-side, model card markdown (config + métricas + pipeline + augmentation + histórico de testes), seed configurável.
- **Interface**: React + FastAPI no mesmo processo Python (sem IPC), preview ao vivo de filtros, file picker nativo no servidor para pastas e checkpoints.

## 🚀 Como Executar

### 1. Pré-requisitos
- **Python 3.13+**
- **Node.js 18+** (para compilar o frontend)
- [uv](https://github.com/astral-sh/uv) (recomendado) ou `pip` para gerenciamento de dependências Python

### 2. Instalação do Ambiente e Engine

Clone o repositório e crie um ambiente virtual:

```bash
git clone https://github.com/marcus-vreis/VisionForge.git
cd VisionForge
uv venv

# Active o ambiente virtual:
# No Windows:
.venv\Scripts\activate
# No Linux/macOS:
# source .venv/bin/activate
```

Install o pacote + suas dependências **escolhendo o build de PyTorch do seu
hardware** via extra (ADR-005 — torch/torchvision não são dependências fixas, já
que o build correto depende da sua placa):

```bash
uv pip install -e ".[dev,cu121]"   # GPU NVIDIA CUDA 12.1 (ou cu118 / cu124 / cu126)
# ou
uv pip install -e ".[dev,cpu]"     # CPU-only
```

Cada extra de hardware (`cpu`, `cu118`, `cu121`, `cu124`, `cu126`) puxa
`torch` + `torchvision` do índice PyTorch correspondente, configurado em
`[tool.uv.sources]`. Sem extra de hardware, o padrão é CPU.

*(Alternativa manual: `uv pip install -e ".[dev]"` e depois instalar o PyTorch à
parte — veja o [site official do PyTorch](https://pytorch.org/get-started/locally/).)*

### 3. Compilação da Interface Web (GUI)

O VisionForge possui um frontend moderno em React que é servido pelo backend. Antes de executar a GUI, você precisa instalar e fazer o build:

```bash
cd frontend
npm install
npm run build
cd ..
```
*(O commando `npm run build` cria a pasta de distribuição e a coloca automaticamente em `src/visionforge/gui/static` para o FastAPI servir).*

---

### 4. Executando Experimentos

Você tem duas opções principais de como interagir com o VisionForge:

#### Opção A: Usando a Interface Web (Recomendado)

Suba o servidor local:

```bash
visionforge gui
```
O sistema abrirá automaticamente o navegador em `http://127.0.0.1:8000`. A partir da interface, você pode configurar todos os parâmetros do seu modelo e acompanhar o treinamento e as métricas.

*Dica para desenvolvedores Frontend:* Se quiser trabalhar na interface com recarregamento rápido, deixe o commando `visionforge gui` rodando num terminal. Em outro terminal (na pasta `frontend`), rode `npm run dev` e acesse pelo `http://localhost:5173`.

#### Opção B: Pelo Terminal (CLI)

Se preferir automatizar fluxos com arquivos `.yaml`, o VisionForge também possui uma interface de linha de commando:

```bash
visionforge run configs/baseline.yaml
```

Todos os logs detalhados, matrizes de confusão e arquivos `.pth` do modelo serão salvos automaticamente na pasta `outputs/`.
