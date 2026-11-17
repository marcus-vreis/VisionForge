# Roadmap

VisionForge is below `1.0.0`: the configuration schema and the HTTP API may still change between minor releases, while `schema_version` migrations keep exported YAML files working. This document summarizes what is already on `development` for the next release, what is planned next, and what defines `1.0.0`, without target dates.

- **What already shipped:** [`CHANGELOG.md`](../CHANGELOG.md)
- **Architectural and statistical decisions (ADRs):** [`docs/dev/DECISIONS.md`](dev/DECISIONS.md)
- **Active backlog and open issues:** [`docs/dev/BACKLOG.md`](dev/BACKLOG.md) and [GitHub Issues](https://github.com/marcus-vreis/VisionForge/issues)
- **Superseded pre-0.2 plan (historical):** [`docs/archive/ROADMAP.md`](archive/ROADMAP.md)

## On `development` (next release)

- **Independent bundle reproduction:** `visionforge reproduce <bundle>` re-trains one seed of a verified experiment bundle (a single run or a replicate group) on a local copy of its dataset, and judges the result against the group's 95% prediction interval for one new seed (ADR-210).

`0.22.0` shipped everything that was listed here, and four items that were under "Planned next": the paired comparison in the GUI for the five task families, `ultralytics` as the optional `detection` extra, the guaranteed offline mode (`--offline`) and the Apple Silicon GPU (`mps`, experimental until it is validated on a real Mac). The full list is in the [`CHANGELOG.md`](../CHANGELOG.md).

## Planned next

- **Reproduction of the other group kinds:** `visionforge reproduce` for K-fold, sweep and paired comparison bundles, and for researcher-defined tasks.

## Criteria for `1.0.0`

1. **Frozen 1.x contracts:** stable configuration schema (with tested forward migration from all `0.x` versions), stable `run.json` provenance schema, and a versioned stability guarantee for `user_tasks/` and `user_models/`.
2. **Full strategy parity:** single run, multi-seed replicates, paired variant comparison, hyperparameter sweeps and K-fold cross-validation (where statistically valid) available in both the GUI and `visionforge run` across the five task families.
3. **Verifiable empirical record:** public `run.json` artifacts and reproducible summaries for the multi-seed variance study and the companion software paper.

---

# Português (pt-BR)

O VisionForge está abaixo da `1.0.0`: o schema de configuração e a API HTTP ainda podem mudar entre versões menores, enquanto a migração por `schema_version` mantém funcionando os arquivos YAML já exportados. Este documento resume o que já está na branch `development` para a próxima release, o que vem a seguir e os critérios para a `1.0.0`, sem prazos prometidos.

- **O que já saiu em cada versão:** [`CHANGELOG.md`](../CHANGELOG.md)
- **Decisões de arquitetura e estatística (ADRs):** [`docs/dev/DECISIONS.md`](dev/DECISIONS.md)
- **Backlog ativo e issues abertas:** [`docs/dev/BACKLOG.md`](dev/BACKLOG.md) e [Issues no GitHub](https://github.com/marcus-vreis/VisionForge/issues)
- **Plano inicial pré-0.2 (histórico):** [`docs/archive/ROADMAP.md`](archive/ROADMAP.md)

## Em `development` (próxima release)

- **Reprodução independente de bundle:** `visionforge reproduce <bundle>` retreina uma semente de um pacote de experimento verificado (uma execução ou um grupo de réplicas) numa cópia local do dataset e julga o resultado pelo intervalo de predição de 95% do grupo para uma semente nova (ADR-210).

A `0.22.0` levou tudo o que estava listado aqui e quatro itens que estavam em "Próximos passos": a comparação pareada na GUI nas cinco famílias de tarefa, o `ultralytics` como extra opcional `detection`, o modo offline garantido (`--offline`) e a GPU Apple Silicon (`mps`, experimental até ser validada num Mac de verdade). A lista completa está no [`CHANGELOG.md`](../CHANGELOG.md).

## Próximos passos

- **Reprodução dos outros tipos de grupo:** `visionforge reproduce` para bundles de K-fold, sweep e comparação pareada, e para tarefas definidas pelo pesquisador.

## Critérios para a `1.0.0`

1. **Contratos 1.x estáveis:** schema de configuração estável (com migração testada a partir de todas as versões `0.x`), contrato `run.json` estável e garantia explícita de estabilidade para `user_tasks/` e `user_models/`.
2. **Paridade completa de estratégias:** treino simples, réplicas multi-semente, comparação pareada entre variantes, buscas de hiperparâmetros e validação cruzada K-fold (onde estatisticamente válida) disponíveis tanto na GUI quanto em `visionforge run` nas cinco tarefas.
3. **Registro empírico verificável:** artefatos `run.json` públicos e sumários regeneráveis para o estudo de variância entre sementes e o artigo do software.
