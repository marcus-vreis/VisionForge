# VisionForge

VisionForge é um ambiente modular e extensível para experimentação em Visão Computacional, com foco em redes neurais convolucionais (CNNs). Substitua notebooks Jupyter ad-hoc por um sistema limpo, testável e reprodutível para treinar, validar e comparar modelos.

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

Install o pacote do projeto e suas dependências:

```bash
uv pip install -e ".[dev]"
```

Install o PyTorch (exemplo para placa de vídeo Nvidia com CUDA 12.1):

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
*(Consulte o [site official do PyTorch](https://pytorch.org/get-started/locally/) para versões exclusivas de CPU ou outras versões do CUDA).*

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
