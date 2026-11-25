# Your own exporter · Seu próprio exportador

A separate Python package can add a destination for finished runs (an internal
registry, a lab server, a format of your own) and `visionforge export` runs it.
VisionForge discovers it through the `visionforge.exporters` entry-point group:
install the package in the same environment and it appears. The built-in
exporters (MLflow, W&B, Comet, Zenodo, GitHub, DVC, Hugging Face Hub) are not
plugins and keep their own pages in the GUI. See ADR-244.

Um pacote Python separado pode acrescentar um destino para runs concluídos (um
registro interno, um servidor do laboratório, um formato próprio) e o
`visionforge export` o executa. O VisionForge o descobre pelo grupo de
entry-points `visionforge.exporters`: instale o pacote no mesmo ambiente e ele
aparece. Os exportadores embutidos (MLflow, W&B, Comet, Zenodo, GitHub, DVC,
Hugging Face Hub) não são plugins e mantêm as suas páginas na interface.

## Declare it · Declare

In the plugin's `pyproject.toml` · No `pyproject.toml` do plugin:

```toml
[project]
name = "visionforge-export-folder"
version = "0.1.0"
dependencies = ["visionforge-studio"]

[project.entry-points."visionforge.exporters"]
folder = "visionforge_export_folder:FolderExporter"
```

The value points to a **class** (VisionForge instantiates it with no arguments)
or to a ready **instance**. · O valor aponta para uma **classe** (o VisionForge a
instancia sem parâmetros) ou para uma **instância** pronta.

```python
# visionforge_export_folder.py
import shutil
from collections.abc import Mapping
from pathlib import Path

from visionforge.core.exporter_plugins import ExportResult


class FolderExporter:
    name = "folder"
    description = "Copies run.json to a folder (option: dest=PATH)"

    def export(self, run_dir: Path, options: Mapping[str, str]) -> ExportResult:
        dest = Path(options["dest"])
        dest.mkdir(parents=True, exist_ok=True)
        target = shutil.copy2(run_dir / "run.json", dest / f"{run_dir.name}.json")
        return ExportResult(str(target), notes=("run.json only, no weights",))
```

```bash
pip install ./visionforge-export-folder
visionforge export --list
visionforge export outputs/models/my_run --to folder --option dest=backup
```

## The contract · O contrato

| Member · Membro | Meaning · Significado |
|---|---|
| `name: str` | Non-empty; the value of `--to`. The first plugin to claim a name keeps it. · Não vazio; o valor de `--to`. O primeiro plugin a usar um nome fica com ele. |
| `description: str` | One line shown by `--list`. · Uma linha mostrada por `--list`. |
| `export(run_dir, options) -> ExportResult` | `run_dir` is a run folder that contains `run.json` (checked before the call). `options` is the `--option key=value` pairs, all strings. Raise `ValueError` or `OSError` for a failure the user can fix: the CLI prints it and exits with code 2. · `run_dir` é a pasta de um run que contém `run.json` (verificado antes da chamada). `options` são os pares `--option chave=valor`, todos strings. Levante `ValueError` ou `OSError` para uma falha que o usuário pode corrigir: a CLI a imprime e sai com código 2. |
| `ExportResult(destination, notes=())` | `destination` is a URL or path, printed on stdout; each note goes to stderr. · `destination` é uma URL ou caminho, impresso na saída padrão; cada nota vai para a saída de erro. |

A plugin that fails to import, cannot be instantiated without arguments or does
not follow the contract is skipped with a warning; it never stops the CLI. A
duplicate name is skipped the same way. · Um plugin que falha ao importar, não
instancia sem parâmetros ou não segue o contrato é ignorado com um aviso; nunca
derruba a CLI. Um nome duplicado é ignorado do mesmo modo.

## What to know first · O que saber antes

- **Trust.** An entry-point runs code from an installed package, with your
  permissions, whenever `visionforge export` runs. Install only plugins you
  trust. · Um entry-point executa código de um pacote instalado, com as suas
  permissões, sempre que o `visionforge export` roda. Instale só plugins em
  que você confia.
- **Credentials.** VisionForge passes none. Read tokens from environment
  variables in your plugin; never put them in `--option`, which lands in shell
  history. · O VisionForge não passa nenhuma. Leia tokens de variáveis de
  ambiente no plugin; nunca em `--option`, que fica no histórico do shell.
- **Stability.** `RunExporter`, `ExportResult` and the group name are the
  public surface while the version is below 1.0: they change only in a minor
  release, with a note in the CHANGELOG. The shape of `run.json` is not part of
  it: read the keys you need and tolerate the rest. · `RunExporter`,
  `ExportResult` e o nome do grupo são a superfície pública enquanto a versão
  está abaixo de 1.0: mudam só numa versão menor, com nota no CHANGELOG. O
  formato do `run.json` não faz parte dela: leia as chaves de que precisa e
  tolere o resto.
- **Scope.** The command is CLI-only; the GUI does not list plugins. · O
  comando existe só na CLI; a interface não lista plugins.
