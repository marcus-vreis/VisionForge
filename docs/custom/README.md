# Extending VisionForge · Estendendo o VisionForge

There are two extension points, and which one you need depends on how far your
research is from the five built-in tasks.

São dois pontos de extensão, e qual deles você precisa depende de quão longe a
sua pesquisa está das cinco tarefas embutidas.

| You want to… · Você quer… | Read · Leia |
|---|---|
| use **your own architecture** in classification, regression or segmentation · usar a **sua arquitetura** em classificação, regressão ou segmentação | [`MODELS.md`](MODELS.md) |
| define **a whole new task family** — your own data, loss and metrics · definir **uma tarefa inteira** — dados, loss e métricas seus | [`TASKS.md`](TASKS.md) |

Both work **without touching the package**: VisionForge loads your code from
`user_models/` and `user_tasks/`, folders it looks for next to wherever you run
it. Nothing is fetched from the network — the trust boundary is your own
filesystem (ADR-005).

Os dois funcionam **sem tocar no pacote**: o VisionForge carrega o seu código
de `user_models/` e `user_tasks/`, pastas que ele procura ao lado de onde você
o executa. Nada é baixado da rede — a fronteira de confiança é o seu próprio
sistema de arquivos (ADR-005).

```
my-research/            ← run `visionforge gui` from here
├── user_models/        ← MODELS.md
├── user_tasks/         ← TASKS.md
├── datasets/
└── outputs/
```

`visionforge doctor` prints the paths it resolved, including how many models
and tasks it found in each.

O `visionforge doctor` imprime os caminhos que resolveu, inclusive quantos
modelos e tarefas encontrou em cada um.
