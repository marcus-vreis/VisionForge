# Instalação passo a passo

Um roteiro por sistema: **Windows**, **Linux** e **macOS**. Em todos, o caminho é o mesmo,
só os comandos mudam:

1. ter o Python 3.12 ou 3.13;
2. criar uma pasta de trabalho com um ambiente virtual;
3. `pip install visionforge-studio`;
4. no Windows e no Linux com placa NVIDIA, `visionforge doctor --fix` troca o PyTorch pelo
   da sua GPU;
5. `visionforge selftest --quick` confere que tudo treina;
6. `visionforge gui` abre a interface.

Leva de 5 a 15 minutos, quase tudo de download: o PyTorch tem de 200 MB (CPU) a uns
3 GB (CUDA).

**Requisitos:** Python 3.12 ou 3.13, uns 6 GB livres em disco e internet na instalação.
Depois de instalado, tudo roda na sua máquina, sem nuvem. Uma GPU NVIDIA deixa o treino
muito mais rápido, mas não é obrigatória: os datasets de exemplo treinam em segundos na CPU.

---

## Windows 10 / 11

### 1. Python

Baixe o instalador do Python **3.12** ou **3.13** em
<https://www.python.org/downloads/windows/> e, na primeira tela, marque
**"Add python.exe to PATH"**. Confira num terminal (PowerShell ou Prompt de Comando):

```powershell
py --version
```

Precisa responder `Python 3.12.x` ou `3.13.x`. Se tiver mais de uma versão, use
`py -3.12` no lugar de `py` no passo seguinte.

### 2. Pasta de trabalho e ambiente virtual

```powershell
mkdir minha-pesquisa
cd minha-pesquisa
py -m venv .venv
.venv\Scripts\activate
```

O prompt passa a começar com `(.venv)`. Se o PowerShell recusar com
"a execução de scripts foi desabilitada", rode uma vez
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e repita a última linha, ou use o
Prompt de Comando (`cmd`), onde a mesma linha funciona sem isso.

### 3. VisionForge

```powershell
pip install visionforge-studio
```

### 4. PyTorch da sua GPU

No Windows o PyPI só tem o PyTorch de CPU, e é ele que o passo 3 instala. Com placa
NVIDIA, troque pelo de CUDA:

```powershell
visionforge doctor --fix
```

Ele lê o driver, mostra qual build vai instalar e pergunta antes; responda `y`. Sem placa
NVIDIA ele recomenda a build de CPU, que você já tem: pode responder `n`.

### 5 e 6. Conferir e abrir

```powershell
visionforge selftest --quick
visionforge gui
```

Veja [Conferir a instalação](#conferir-a-instalação) e [Primeiro uso](#primeiro-uso).

---

## Linux (Ubuntu 24.04 e parecidos)

### 1. Python

O Ubuntu 24.04 já vem com o Python 3.12; falta o módulo de ambientes virtuais e o Tk, que
o seletor de pastas da interface usa:

```bash
sudo apt update
sudo apt install python3-venv python3-tk
python3 --version
```

No Ubuntu 22.04 o Python do sistema é o 3.10, antigo demais. Instale o 3.12 (por exemplo
com `sudo add-apt-repository ppa:deadsnakes/ppa` e
`sudo apt install python3.12 python3.12-venv python3.12-tk`) e use `python3.12` no lugar de
`python3` abaixo.

### 2. Pasta de trabalho e ambiente virtual

```bash
mkdir minha-pesquisa && cd minha-pesquisa
python3 -m venv .venv
source .venv/bin/activate
```

### 3. VisionForge

```bash
pip install visionforge-studio
```

### 4. PyTorch da sua GPU

No Linux o PyTorch que vem do PyPI já é o de CUDA. Mesmo assim, com placa NVIDIA, rode:

```bash
visionforge doctor --fix
```

Se ele disser **"keep the current install"**, responda `n`: a GPU já funciona. Se
recomendar outra build (uma RTX série 50 precisa da `cu128`), responda `y`. O driver
NVIDIA precisa estar instalado (`nvidia-smi` deve funcionar no terminal). Sem placa
NVIDIA, responda `n` e treine na CPU.

### 5 e 6. Conferir e abrir

```bash
visionforge selftest --quick
visionforge gui
```

---

## macOS (Apple Silicon: M1, M2, M3, M4)

> **Só Macs com chip Apple.** O PyTorch não publica versões para Mac com processador Intel
> desde a 2.3, e o VisionForge precisa dela. Para conferir: menu  → *Sobre este Mac* →
> "Chip: Apple M…".
>
> **O treino roda na CPU.** O VisionForge usa GPU NVIDIA (CUDA); a GPU do Mac (MPS) ainda
> não é usada. Os datasets de exemplo treinam em segundos assim mesmo; um dataset grande
> vai demorar.

### 1. Python

O `python3` que vem com o macOS é antigo. Baixe o instalador do **Python 3.12** ou
**3.13** em <https://www.python.org/downloads/macos/> ("macOS 64-bit universal2
installer") e instale.

Ao terminar, o instalador abre uma pasta no Finder: dê dois cliques em
**"Install Certificates.command"**. Sem isso, os downloads de datasets dentro da interface
podem falhar com erro de certificado (SSL).

No Terminal:

```bash
python3.12 --version
```

(Se instalou o 3.13, use `python3.13` aqui e no passo seguinte.)

Prefere o Homebrew? `brew install python@3.12 python-tk@3.12` também serve; o `python-tk`
é o que o seletor de pastas usa.

### 2. Pasta de trabalho e ambiente virtual

```bash
mkdir minha-pesquisa && cd minha-pesquisa
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. VisionForge

```bash
pip install visionforge-studio
```

### 4. PyTorch

**Pule o `doctor --fix` no Mac.** O PyTorch que o passo 3 instalou já é o certo, e o
`--fix` só baixaria a mesma build de CPU de novo. Para ver o diagnóstico sem mudar nada:

```bash
visionforge doctor
```

É esperado ver `No CUDA-capable GPU detected` e `cuda_available=False`. O que importa é a
última linha, `Verdict: environment looks good`.

### 5 e 6. Conferir e abrir

```bash
visionforge selftest --quick
visionforge gui
```

**Se o botão de escolher pasta não abrir nada**, cole o caminho da pasta direto no campo:
no Finder, clique na pasta com o botão direito, segure **Option** e escolha
*Copiar "…" como Nome do Caminho*. Ou use o botão **usar dataset de exemplo**, que não
precisa escolher pasta.

---

## Conferir a instalação

```bash
visionforge doctor
visionforge selftest --quick
```

O `selftest --quick` sobe o servidor de verdade e treina uma vez cada tarefa em imagens
sintéticas, em uns 15 segundos (mais na CPU). Ele termina com `N/N cases passed`. Isso
confere que a instalação funciona, não que um modelo é bom.

## Primeiro uso

`visionforge gui` abre a interface no navegador, em <http://127.0.0.1:8000>. O terminal
precisa ficar aberto enquanto você usa; para encerrar, `Ctrl+C` nele.

1. A primeira tela pergunta seu nome e oferece um guia rápido.
2. Escolha a aba de uma tarefa (comece por **Classificação**).
3. Clique em **usar dataset de exemplo**, embaixo da pasta base. Ele cria um dataset pequeno
   em `datasets/exemplo-classificacao/` e preenche o formulário.
4. Clique em **Treinar** e acompanhe a tela de treino.
5. Abra o **histórico** (barra de baixo) para ver o run, as métricas e os gráficos.

O seletor ao lado de PT/EN, no topo, escolhe o tema: automático (segue o sistema), claro
ou escuro.

Depois, para usar os seus dados ou baixar um dataset público (botão **⤓ datasets**), veja
o [`QUICKSTART.md`](QUICKSTART.md) e o [`DATASETS.md`](DATASETS.md).

## Usar de novo outro dia

Abra o terminal na mesma pasta e ative o ambiente antes:

```bash
cd minha-pesquisa
source .venv/bin/activate        # Windows: .venv\Scripts\activate
visionforge gui
```

Os runs ficam em `minha-pesquisa/outputs/`; rode sempre da mesma pasta para ver o mesmo
histórico.

## Atualizar e desinstalar

```bash
pip install --upgrade visionforge-studio
```

A atualização mantém o PyTorch que você já tem. Para desinstalar, apague a pasta `.venv`
(e `outputs/` e `datasets/`, se não quiser os runs e os dados).

## Problemas comuns

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| `visionforge: command not found` / "não é reconhecido" | ambiente não ativado | ative o `.venv` (passo 2) e repita |
| `pip install` diz `No matching distribution found for visionforge-studio` | Python abaixo de 3.12 no `.venv` | instale o 3.12, apague o `.venv` e recrie (passo 2) |
| No Mac, `pip install` não acha `torch` | Mac com processador Intel | não suportado (veja o aviso do macOS) |
| `doctor` mostra `[FAIL] GPU driver detected but torch.cuda.is_available() is False` | PyTorch de CPU numa máquina com NVIDIA | `visionforge doctor --fix` e responda `y` |
| Download de dataset falha com `SSL: CERTIFICATE_VERIFY_FAILED` (Mac) | certificados do Python.org não instalados | rode "Install Certificates.command" (passo 1 do macOS) |
| O botão de escolher pasta não faz nada | falta o Tk (Linux) ou o diálogo nativo falhou (Mac) | Linux: `sudo apt install python3-tk`; em qualquer sistema, cole o caminho no campo |
| A porta 8000 já está em uso | outro programa (ou outro `visionforge gui`) nela | feche o outro terminal ou use `visionforge gui --port 8001` |
