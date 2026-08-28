# user_tasks/ — your own task families · suas próprias tarefas

Define a whole new task — data, model, loss, metrics — in one Python file, and
VisionForge gives you the GUI tab, the live monitor, the run history, sweeps
and multi-seed replicates for free:

```bash
visionforge new-task cell_counting
```

The generated file already trains on synthetic data. Fill the four hooks
(`build_model`, `build_loaders`, `compute_loss`, `compute_metrics`) and a
Pydantic `Config` whose fields become the form.

Defina uma tarefa inteira — dados, modelo, loss, métricas — em um arquivo
Python, e o VisionForge entrega de graça a aba na interface, o monitor ao vivo,
o histórico, as buscas e as réplicas.

**Full guide · Guia completo:**
[`docs/custom/TASKS.md`](https://github.com/marcus-vreis/VisionForge/blob/main/docs/custom/TASKS.md)

`example_counting/` is a working example: a small CNN counting dots in
synthetic images, trains in seconds on CPU.
