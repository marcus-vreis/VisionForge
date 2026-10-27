# exp

- **ID da execução:** `run_1`
- **Data e hora:** 2026-05-22T10:00:00
- **Situação:** concluída
- **Interrompida:** sim (2 de 5 épocas)
- **Dispositivo usado:** `cpu`
- **Diretório da execução:** `fixture\run_1`
- **Arquivo do checkpoint:** `m.pth`

## Configuração

- **Arquitetura:** resnet18
- **num_classes:** 1
- **Pré-treinado:** False
- **Tarefa:** binary
- **Otimizador:** adam
- **Taxa de aprendizado:** 0.001
- **Tamanho do batch:** 4
- **Épocas (máx.):** 5
- **Seed:** 0
- **Scheduler:** step ({'kind': 'step'})
- **Precisão mista:** ativada

## Pipeline de pré-processamento

Aplicadas a cada imagem carregada, antes do data augmentation, nesta ordem:

1. **resize**
2. **blur** (radius=2)

## Data augmentation / Normalização

| Campo | Valor |
|---|---|
| flip | `True` |
| image_size | `64` |

## Métricas finais

| Métrica | Valor | IC 95% (bootstrap) |
|---|---|---|
| accuracy | 0.8000 | [0.7000, 0.9000] |
| total_epochs | 2 | — |
| missing | — | — |
| test_loss | 0.2500 | — |

Os intervalos são bootstrap percentílico sobre as 40 imagens de teste (1000 reamostragens): quantificam o ruído amostral do conjunto de teste para este modelo fixo, não a variância entre execuções. Para esta última, treine réplicas com várias seeds.

## Histórico de treino (2 época(s))

| Época | loss | phase | val_miou |
|---|---|---|---|
| 1 | 0.5000 | warmup | — |
| 2 | — | — | 0.8000 |

## Gráficos

- `cm.png`

## Testes em conjuntos de dados adicionais (1)

### ext
- **diretório-base:** `d`
- **data e hora:** t
| Métrica | Valor |
|---|---|
| accuracy | 0.5000 |
