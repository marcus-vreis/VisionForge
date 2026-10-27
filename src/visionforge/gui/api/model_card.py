"""Pure localized model-card sections read from the run.json contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from visionforge.utils.messages import tr_artifact

MARKDOWN_TABLE_SEPARATOR = "|---|---|"


def _stopped_extent(data: dict[str, Any]) -> str:
    """How far a stopped run got, in the unit that run counts (ADR-111)."""
    metrics: dict[str, Any] = data.get("metrics") or {}
    config: dict[str, Any] = data.get("config") or {}
    cv = metrics.get("cv_aggregate")
    if isinstance(cv, dict):
        return tr_artifact(
            "card.stopped_folds",
            ok=cv.get("n_folds_ok", "?"),
            n=cv.get("n_folds", "?"),
        )
    if (config.get("model") or {}).get("name") == "patchcore":
        # PatchCore has phases, not epochs: one fit counts the bank as built.
        return tr_artifact(
            "card.stopped_bank_built"
            if metrics.get("total_epochs")
            else "card.stopped_before_bank"
        )
    planned = (config.get("training") or {}).get("epochs", "?")
    return tr_artifact(
        "card.stopped_epochs", done=metrics.get("total_epochs", "?"), planned=planned
    )


def _card_status(status: Any) -> str:
    """The run's status in the card's language; an unknown one is kept as is."""
    if status is None:
        return "—"
    if status == "completed":
        return tr_artifact("card.status_completed")
    if status == "failed":
        return tr_artifact("card.status_failed")
    if status == "running":
        return tr_artifact("card.status_running")
    if status == "stopped":
        return tr_artifact("card.status_stopped")
    return str(status)


def _card_value(value: Any) -> str:
    """A metric cell: four decimals for a float, a dash for a missing value."""
    if value is None:
        return "—"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _identity_lines(run_dir: Path, data: dict[str, Any]) -> list[str]:
    """Render run identity, status and checkpoint paths."""
    lines: list[str] = []
    name = data.get("experiment", run_dir.name)
    lines.extend([f"# {name}", ""])
    # The History lists a run by its folder name, so the card carries the same
    # id; run.json's own "id" is "<experiment>_<folder>" and matched nothing the
    # user sees. A bare Path (no folder) falls back to it.
    lines.extend(
        [
            f"- **{tr_artifact('card.run_id')}:** `{run_dir.name or data.get('id', '')}`",
            f"- **{tr_artifact('card.timestamp')}:** {data.get('timestamp', '?')}",
            f"- **{tr_artifact('card.status')}:** {_card_status(data.get('status'))}",
        ]
    )
    if data.get("stopped"):
        lines.append(
            f"- **{tr_artifact('card.stopped')}:** {tr_artifact('card.yes')} "
            f"({_stopped_extent(data)})"
        )
    if data.get("device_used"):
        lines.append(
            f"- **{tr_artifact('card.device_used')}:** `{data['device_used']}`"
        )
    lines.append(
        f"- **{tr_artifact('card.run_dir')}:** `{data.get('run_dir', run_dir)}`"
    )
    artifacts = data.get("artifacts", {})
    if artifacts.get("model"):
        lines.append(f"- **{tr_artifact('card.checkpoint')}:** `{artifacts['model']}`")
    return lines


def _configuration_lines(config: dict[str, Any], data: dict[str, Any]) -> list[str]:
    """Render only configuration fields recorded by the task."""
    lines: list[str] = []
    lines.extend(
        [
            "",
            f"## {tr_artifact('card.configuration')}",
            "",
        ]
    )
    model_cfg = config.get("model") or {}
    training_cfg = config.get("training") or {}
    # Each task has its own config shape: a row whose field the task does not
    # have is left out rather than printed as "?".
    task_label = data.get("task_label")
    rows: list[tuple[str, Any]] = [
        (
            tr_artifact("card.architecture"),
            model_cfg.get("name") or task_label,
        ),
        ("num_classes", model_cfg.get("num_classes")),
        ("num_targets", model_cfg.get("num_targets")),
        (tr_artifact("card.pretrained"), model_cfg.get("pretrained")),
        (
            tr_artifact("card.task"),
            task_label or config.get("task") or data.get("task"),
        ),
        (tr_artifact("card.optimizer"), training_cfg.get("optimizer")),
        (tr_artifact("card.learning_rate"), training_cfg.get("learning_rate")),
        (tr_artifact("card.batch_size"), training_cfg.get("batch_size")),
        (tr_artifact("card.epochs_max"), training_cfg.get("epochs")),
        # "Seed" and "Scheduler" are the words a Portuguese paper uses too.
        ("Seed", training_cfg.get("seed")),
    ]
    for label, value in rows:
        if value is not None:
            lines.append(f"- **{label}:** {value}")
    if training_cfg.get("scheduler", {}).get("kind", "none") != "none":
        sched = training_cfg["scheduler"]
        lines.append(f"- **Scheduler:** {sched['kind']} ({sched})")
    if training_cfg.get("mixed_precision"):
        lines.append(
            f"- **{tr_artifact('card.mixed_precision')}:** {tr_artifact('card.enabled')}"
        )
    lines.append("")

    return lines


def _preprocessing_lines(config: dict[str, Any]) -> list[str]:
    """Render preprocessing steps in their recorded order."""
    lines: list[str] = []
    data_cfg = config.get("data", {})
    pp_steps = data_cfg.get("preprocessing", {}).get("steps", [])
    if pp_steps:
        lines.extend(
            [
                f"## {tr_artifact('card.preprocessing')}",
                "",
                tr_artifact("card.preprocessing_intro"),
                "",
            ]
        )
        for i, step in enumerate(pp_steps, start=1):
            kind = step.get("kind", "?")
            params = ", ".join(f"{k}={v!r}" for k, v in step.items() if k != "kind")
            lines.append(f"{i}. **{kind}**" + (f" ({params})" if params else ""))
        lines.append("")

    return lines


def _augmentation_lines(config: dict[str, Any]) -> list[str]:
    """Render the transforms and image size used by the task."""
    lines: list[str] = []
    data_cfg = config.get("data", {})
    tx = dict(data_cfg.get("transforms") or {})
    # Anomaly and segmentation resize to data.image_size; the image_size inside
    # transforms is a classification default they never read, so the card would
    # state a resolution the model never saw.
    if tx and isinstance(data_cfg.get("image_size"), int):
        tx["image_size"] = data_cfg["image_size"]
    if tx:
        lines.extend(
            [
                f"## {tr_artifact('card.augmentation')}",
                "",
                f"| {tr_artifact('card.field')} | {tr_artifact('card.value')} |",
                MARKDOWN_TABLE_SEPARATOR,
            ]
        )
        for k, v in tx.items():
            lines.append(f"| {k} | `{v}` |")
        lines.append("")

    return lines


def _metric_lines(metrics: dict[str, Any], data: dict[str, Any]) -> list[str]:
    """Render final metrics and available bootstrap intervals."""
    lines: list[str] = []
    lines.append(f"## {tr_artifact('card.final_metrics')}")
    lines.append("")
    if metrics:
        cis: dict[str, Any] = data.get("metric_cis", {})
        has_cis = bool(cis)
        header = f"| {tr_artifact('card.metric')} | {tr_artifact('card.value')} |"
        if has_cis:
            header += f" {tr_artifact('card.ci_bootstrap')} |"
        lines.append(header)
        lines.append("|---|---|---|" if has_cis else MARKDOWN_TABLE_SEPARATOR)
        for k, v in metrics.items():
            value = _card_value(v)
            if not has_cis:
                lines.append(f"| {k} | {value} |")
                continue
            # run.json keys the CIs by bare metric name; the metrics block
            # prefixes the test-split ones with "test_".
            ci = cis.get(k[len("test_") :] if k.startswith("test_") else k)
            interval = f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]" if ci else "—"
            lines.append(f"| {k} | {value} | {interval} |")
        if has_cis:
            sample = next(iter(cis.values()))
            lines.append("")
            lines.append(
                tr_artifact(
                    "card.bootstrap_note",
                    n_samples=sample["n_samples"],
                    n_resamples=sample["n_resamples"],
                )
            )
    else:
        lines.append(tr_artifact("card.none_recorded"))
    lines.append("")

    return lines


def _history_lines(history: list[dict[str, Any]]) -> list[str]:
    """Render heterogeneous epoch records with stable column order."""
    lines: list[str] = []
    if history:
        # Columns come from the history itself rather than a fixed
        # classification shape: a segmentation epoch has val_miou and no
        # val_accuracy, a custom task streams whatever metrics it declared, and
        # a hardcoded header formatted every missing cell with ":.4f" — which
        # raised on the "?" fallback and made this endpoint 500 for every task
        # except classification.
        columns = _history_columns(history)

        lines.extend(
            [
                f"## {tr_artifact('card.history', n=len(history))}",
                "",
                f"| {tr_artifact('card.epoch')} | " + " | ".join(columns) + " |",
                "|---" * (len(columns) + 1) + "|",
            ]
        )
        for epoch in history:
            cells = [str(epoch.get("epoch", "?"))]
            cells.extend(_card_value(epoch.get(key)) for key in columns)
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    return lines


def _history_columns(history: list[dict[str, Any]]) -> list[str]:
    """Keep metric columns in first-observed order across all epochs."""
    columns: list[str] = []
    for epoch in history:
        for key in epoch:
            if key != "epoch" and key not in columns:
                columns.append(key)
    return columns


def _graphic_lines(artifacts: dict[str, Any]) -> list[str]:
    """List the recorded graphic artifacts."""
    lines: list[str] = []
    graphics = artifacts.get("graphics", [])
    if graphics:
        lines.append(f"## {tr_artifact('card.plots')}")
        lines.append("")
        for g in graphics:
            lines.append(f"- `{g}`")
        lines.append("")

    return lines


def _test_lines(data: dict[str, Any]) -> list[str]:
    """Render results from additional test datasets."""
    lines: list[str] = []
    tests = data.get("tests", [])
    if tests:
        lines.append(f"## {tr_artifact('card.tests', n=len(tests))}")
        lines.append("")
        for t in tests:
            lines.extend(
                [
                    f"### {t.get('label', t.get('test_id'))}",
                    f"- **{tr_artifact('card.test_base_dir')}:** `{t.get('base_dir')}`",
                    f"- **{tr_artifact('card.test_timestamp')}:** {t.get('timestamp')}",
                ]
            )
            m = t.get("metrics", {})
            if m:
                lines.append(
                    f"| {tr_artifact('card.metric')} | {tr_artifact('card.value')} |"
                )
                lines.append(MARKDOWN_TABLE_SEPARATOR)
                for k, v in m.items():
                    lines.append(f"| {k} | {_card_value(v)} |")
            lines.append("")

    return lines


def render_run_markdown(run_dir: Path, data: dict[str, Any]) -> str:
    """Format a run.json into a localized model card without mutating it."""
    config = data.get("config", {})
    lines = _identity_lines(run_dir, data)
    lines.extend(_configuration_lines(config, data))
    lines.extend(_preprocessing_lines(config))
    lines.extend(_augmentation_lines(config))
    lines.extend(_metric_lines(data.get("metrics", {}), data))
    lines.extend(_history_lines(data.get("history", [])))
    lines.extend(_graphic_lines(data.get("artifacts", {})))
    lines.extend(_test_lines(data))
    return "\n".join(lines)
