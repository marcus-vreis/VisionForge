"""LaTeX tables from experiment reports (ADR-061).

The last mile of a result is a table in a paper, and retyping numbers is where
transcription errors enter — the kind no reviewer can catch and no rerun
reproduces. Every advanced report (replicates, sweep, cross-validation) is
therefore written as a ``booktabs`` table next to its JSON/CSV, ready to
``\\input{}``.

Pure string building over the report dicts the API already produces: no torch,
no filesystem, testable against exact expected output.
"""

from __future__ import annotations

from typing import Any

from visionforge.utils.messages import tr_artifact

# Characters that would otherwise break compilation or silently render wrong.
_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape(text: Any) -> str:
    """Escape a value for LaTeX text mode (metric names carry underscores).

    Single pass, character by character: chained ``str.replace`` calls would
    re-escape the braces of an earlier replacement, turning ``a\\b`` into
    ``a\\textbackslash\\{\\}b``.
    """
    return "".join(_ESCAPES.get(char, char) for char in str(text))


def _num(value: Any, digits: int = 4) -> str:
    """Format a number, or an em dash when it is missing/not numeric."""
    if value is None:
        return "---"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return escape(value)
    if number != number:  # NaN
        return "---"
    return f"{number:.{digits}f}"


def _counts(*, stopped: int = 0, failed: int = 0, not_run: int = 0) -> str:
    """ "1 stopped, 3 not run" for the non-zero counts, in a fixed order."""
    parts = (
        ("latex.count_stopped", stopped),
        ("latex.count_failed", failed),
        ("latex.count_not_run", not_run),
    )
    return ", ".join(tr_artifact(key, n=n) for key, n in parts if n)


def _table(
    *,
    caption: str,
    label: str,
    column_spec: str,
    header: list[str],
    rows: list[list[str]],
    note: str = "",
) -> str:
    """Assemble one booktabs table environment."""
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{column_spec}}}",
        r"    \toprule",
        "    " + " & ".join(header) + r" \\",
        r"    \midrule",
    ]
    lines.extend("    " + " & ".join(row) + r" \\" for row in rows)
    lines.extend([r"    \bottomrule", r"  \end{tabular}"])
    if note:
        lines.append(f"  \\par\\smallskip\\footnotesize {note}")
    lines.extend([r"\end{table}", ""])
    return "\n".join(lines)


def replicates_to_latex(report: dict[str, Any], *, experiment: str = "") -> str:
    """Per-metric mean ± SD with both confidence intervals over N seeds."""
    aggregates: dict[str, Any] = report.get("aggregates") or {}
    metric = report.get("metric", "")
    seeds = report.get("seeds") or []
    rows: list[list[str]] = []
    for name, agg in aggregates.items():
        marker = r"$\star$" if name == metric else ""
        rows.append(
            [
                f"{escape(name)} {marker}".strip(),
                _num(agg.get("mean")),
                _num(agg.get("std")),
                f"[{_num(agg.get('ci95_low'))}, {_num(agg.get('ci95_high'))}]",
                f"[{_num(agg.get('boot95_low'))}, {_num(agg.get('boot95_high'))}]",
            ]
        )
    n = report.get("successful_replicates", len(seeds))
    # The seeds that produced the numbers above, not the ones asked for: after
    # a stop or a failure they differ, and the table must be reproducible from
    # what it says (ADR-111).
    trials: list[dict[str, Any]] = report.get("trials") or []
    finished = (
        [t.get("seed") for t in trials if t.get("status") == "success"]
        if trials
        else list(seeds)
    )
    seed_note = escape(", ".join(str(s) for s in finished))
    if len(finished) < len(seeds):
        statuses = [t.get("status") for t in trials]
        detail = _counts(
            stopped=statuses.count("stopped"),
            failed=statuses.count("failed"),
            not_run=len(seeds) - len(trials),
        )
        seed_note += " " + tr_artifact(
            "latex.seeds_requested",
            finished=len(finished),
            planned=len(seeds),
            detail=detail,
        )
    # Below ~5 seeds both intervals mislead in opposite directions (t far too
    # wide, percentile bootstrap far too narrow). Say so in the table itself:
    # a caveat that lives only in the docs never reaches the reader.
    caveat = (
        " " + tr_artifact("latex.replicates_caution")
        if isinstance(n, int) and n < 5
        else ""
    )
    return _table(
        caption=tr_artifact(
            "latex.replicates_caption",
            experiment=escape(experiment or tr_artifact("latex.experiment")),
            n=n,
        ),
        label="tab:replicates",
        column_spec="lrrcc",
        header=[
            tr_artifact("latex.metric"),
            tr_artifact("latex.mean"),
            tr_artifact("latex.sd"),
            tr_artifact("latex.ci_t"),
            tr_artifact("latex.ci_bootstrap"),
        ],
        rows=rows,
        note=tr_artifact("latex.replicates_note", seeds=seed_note, caveat=caveat),
    )


def sweep_to_latex(report: dict[str, Any], *, experiment: str = "") -> str:
    """Ranked sweep trials with the searched values and the ranking metric."""
    metric = report.get("metric", "")
    trials: list[dict[str, Any]] = [
        t for t in (report.get("trials") or []) if t.get("status") == "success"
    ]
    paths: list[str] = []
    for trial in trials:
        for path in trial.get("overrides") or {}:
            if path not in paths:
                paths.append(path)

    rows: list[list[str]] = []
    for rank, trial in enumerate(trials, start=1):
        overrides = trial.get("overrides") or {}
        rows.append(
            [
                str(rank),
                *[_num(overrides.get(p), digits=5) for p in paths],
                _num((trial.get("metrics") or {}).get(metric)),
                _num(trial.get("training_time_s"), digits=1),
            ]
        )
    return _table(
        caption=tr_artifact(
            "latex.sweep_caption",
            experiment=escape(experiment or tr_artifact("latex.experiment")),
            mode=escape(report.get("mode", "grid")),
            metric=escape(metric),
        ),
        label="tab:sweep",
        column_spec="l" + "r" * (len(paths) + 2),
        header=[
            "\\#",
            *[escape(p.split(".")[-1]) for p in paths],
            escape(metric),
            tr_artifact("latex.time_s"),
        ],
        rows=rows,
        note=tr_artifact(
            "latex.sweep_note",
            succeeded=len(trials),
            total=report.get("total_trials", len(trials)),
            stop_note=_sweep_stop_note(report),
        ),
    )


def _sweep_stop_note(report: dict[str, Any]) -> str:
    """For a stopped sweep: how far it got of what was planned (ADR-111)."""
    if not report.get("stopped"):
        return ""
    ran = report.get("total_trials", 0)
    planned = report.get("planned_trials", ran)
    cut = report.get("stopped_count", 0)
    key = "latex.sweep_stopped_cut" if cut else "latex.sweep_stopped"
    return " " + tr_artifact(key, ran=ran, planned=planned, cut=cut)


def cv_to_latex(report: dict[str, Any], *, experiment: str = "") -> str:
    """Per-fold metrics plus the mean ± SD aggregate.

    Only folds that ran to the end show numbers: a fold the stop cut, or one
    that failed, keeps its status and dashes, and the caption counts the folds
    the mean is over (ADR-111).
    """
    folds: list[dict[str, Any]] = report.get("fold_results") or []
    aggregate: dict[str, Any] = report.get("aggregate") or {}
    names = list(aggregate) or sorted(
        {k for f in folds for k in (f.get("metrics") or {})}
    )

    rows: list[list[str]] = []
    for fold in folds:
        number = str(int(fold.get("fold", 0)) + 1)
        status = fold.get("status", "success")
        if status != "success":
            rows.append([f"{number} ({escape(status)})", *["---" for _ in names]])
            continue
        metrics = fold.get("metrics") or {}
        rows.append([number, *[_num(metrics.get(name)) for name in names]])
    if aggregate:
        rows.append(
            [
                tr_artifact("latex.cv_mean_sd"),
                *[
                    f"{_num(aggregate[name].get('mean'))} $\\pm$ "
                    f"{_num(aggregate[name].get('std'))}"
                    for name in names
                ],
            ]
        )
    n_folds = report.get("n_folds", len(folds))
    statuses = [f.get("status", "success") for f in folds]
    finished = statuses.count("success")
    shown = escape(experiment or tr_artifact("latex.experiment"))
    if finished < n_folds:
        detail = _counts(
            stopped=statuses.count("stopped"),
            failed=statuses.count("failed"),
            not_run=n_folds - len(folds),
        )
        caption = tr_artifact(
            "latex.cv_caption_partial",
            experiment=shown,
            n_folds=n_folds,
            finished=finished,
            detail=detail,
        )
    else:
        caption = tr_artifact("latex.cv_caption", experiment=shown, n_folds=n_folds)
    return _table(
        caption=caption,
        label="tab:cv",
        column_spec="l" + "r" * len(names),
        # "Fold" is the same word in both languages, so it has no catalog entry.
        header=["Fold", *[escape(n) for n in names]],
        rows=rows,
    )


def comparison_to_latex(
    comparisons: list[dict[str, Any]], *, experiment: str = "", alpha: float = 0.05
) -> str:
    """Paired significance matrix: difference, CI, test, p, effect size."""
    rows: list[list[str]] = []
    for c in comparisons:
        star = r"$^{*}$" if c.get("significant") else ""
        rows.append(
            [
                tr_artifact(
                    "latex.versus",
                    a=escape(c.get("label_a")),
                    b=escape(c.get("label_b")),
                ),
                f"{_num(c.get('mean_difference'))}{star}",
                f"[{_num(c.get('ci_low'))}, {_num(c.get('ci_high'))}]",
                escape(c.get("test", "").replace("_", " ")),
                _num(c.get("p_value"), digits=4),
                _num(c.get("effect_size"), digits=2),
            ]
        )
    # A rank test on few pairs has a p-value floor; if it sits above alpha,
    # "not significant" carries no information about the effect and the table
    # must not let a reader infer otherwise.
    blocked = [c for c in comparisons if c.get("underpowered")]
    power_note = ""
    if blocked:
        floor = max(float(c.get("min_achievable_p") or 0.0) for c in blocked)
        power_note = " " + tr_artifact(
            "latex.comparison_underpowered", floor=floor, alpha=alpha
        )
    return _table(
        caption=tr_artifact(
            "latex.comparison_caption",
            experiment=escape(experiment or tr_artifact("latex.experiment")),
        ),
        label="tab:comparison",
        column_spec="lrcccr",
        header=[
            tr_artifact("latex.comparison"),
            tr_artifact("latex.difference"),
            tr_artifact("latex.ci"),
            tr_artifact("latex.test"),
            "$p$",
            "$d_z$",
        ],
        rows=rows,
        note=tr_artifact("latex.comparison_note", alpha=alpha, power_note=power_note),
    )


def report_to_latex(report: dict[str, Any], *, experiment: str = "") -> str | None:
    """Dispatch on report shape; ``None`` when there is no table to write.

    Detection order matters: replicate and CV reports both carry aggregates,
    and a sweep report is the only one with ranked trials.
    """
    if not isinstance(report, dict):
        return None
    if "aggregates" in report and "seeds" in report:
        return replicates_to_latex(report, experiment=experiment)
    if "fold_results" in report:
        return cv_to_latex(report, experiment=experiment)
    if "trials" in report and "best_trial" in report:
        return sweep_to_latex(report, experiment=experiment)
    if "comparisons" in report:
        return comparison_to_latex(report["comparisons"], experiment=experiment)
    return None


__all__ = [
    "comparison_to_latex",
    "cv_to_latex",
    "escape",
    "replicates_to_latex",
    "report_to_latex",
    "sweep_to_latex",
]
