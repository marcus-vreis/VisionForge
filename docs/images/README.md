# Imagens do README

Geradas por `python scripts/capture_docs_images.py` (issue #221): a GUI
deste checkout num workspace temporário, `--offline`, sem GPU, Chrome
a 1440×960, tema escuro, pt-BR. A versão do cabeçalho fica oculta na
captura. Não edite à mão: rode o script de novo e confira com
`python scripts/capture_docs_images.py --check`.

- Commit: `731f308b6e0c6efd50af47a1ed739439f9cde0d7`
- Versão: `0.21.0`
- Data: 2026-11-15

| Imagem | Mostra | Dimensões (px) |
| --- | --- | --- |
| `vf-classification.png` | Classificação, etapa Dados com o dataset de exemplo | 1440×960 |
| `vf-replicates.png` | Regressão, etapa Estratégia com Réplicas ligado | 1440×960 |
| `vf-code-tab.png` | Aba Código ao vivo do painel de treino, após 1 época na CPU | 1440×960 |
| `vf-colab.png` | Canto do Colab aberto, sem CLI instalado nem conta Google | 1440×960 |
| `vf-strategy.png` | Classificação, etapa Estratégia com as seis estratégias | 1440×960 |
| `vf-guide-menu.png` | Menu de guias do cabeçalho aberto | 1440×960 |
| `vf-learn.png` | Tela Começando do zero, abertura e primeiro capítulo | 1440×960 |
| `vf-learn-curves.png` | Tela Começando do zero, capítulo das curvas de perda | 1440×960 |
| `vf-review.png` | Classificação, etapa Revisão antes de treinar | 1440×960 |
| `vf-training.png` | Painel de treino na aba Log, após 1 época na CPU | 1440×960 |
| `vf-results.png` | Execução aberta no Histórico por Ver resultados: métricas e curvas | 1440×960 |
| `vf-run-detail.png` | Histórico, detalhe de uma execução com as ações de publicação | 1440×960 |
| `vf-transfer.png` | Classificação, etapa Modelo com transfer learning ligado | 1440×960 |
| `vf-preprocessing.png` | Etapa Imagem e aumento com um filtro e o preview dele | 1440×960 |
| `vf-datasets-explorer.png` | Laboratório de datasets, Visão geral do dataset de exemplo | 1440×960 |

## Animação

`vf-training-film.gif`: Um treino de 8 épocas na CPU, da etapa Revisão ao resultado (animação). O mesmo script filma um segundo
treino, quadro a quadro, e monta o GIF. Dos mesmos quadros e de duas
imagens acima saem os arquivos WebP que a tela Começando do zero da
interface mostra, em `frontend/src/assets/learn/`.
