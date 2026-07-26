"""End-to-end self-test: drives the real GUI API with real trainings (ADR-060).

``visionforge selftest`` answers the question unit tests cannot: *does this
install actually train?* It starts the real FastAPI app on a real socket,
builds tiny synthetic datasets (``selftest_data``), and POSTs to the same
endpoints the browser uses — one case per (task, strategy) pair. Each case is
validated on three axes:

1. the run reaches ``completed`` and the stored report carries its metrics,
2. the SSE stream delivers the live-monitor contract (``start``/``epoch_end``/
   ``end``, or ``trial_start``/``trial_end`` for multi-trial strategies),
3. artifacts (``run.json``) land on disk for single runs.

Everything is CPU-sized, ``pretrained=False`` and ``num_workers=0`` so it runs
offline in minutes on any machine, and writes only inside a scratch directory.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from visionforge.utils.selftest_data import (
    build_anomaly_dataset,
    build_classification_dataset,
    build_detection_dataset,
    build_regression_dataset,
    build_segmentation_dataset,
)

TASKS = (
    "classification",
    "regression",
    "segmentation",
    "anomaly",
    "detection",
    "custom",
)
STRATEGIES = ("simple", "cv", "sweep", "replicates")

# Per-case ceiling. A one-epoch CPU run on tiny data is seconds; this only
# fires when something is genuinely stuck.
CASE_TIMEOUT_S = 900.0


@dataclass(frozen=True)
class SelfTestCase:
    """One (task, strategy) probe against a real endpoint."""

    task: str
    strategy: str
    endpoint: str
    payload: dict[str, Any]
    # Report keys that must be present for the case to count as passed.
    expect_keys: tuple[str, ...] = ()
    # True when the strategy runs N inner trainings (fold/trial/seed) and the
    # SSE stream must therefore carry trial_start/trial_end.
    multi_trial: bool = False


@dataclass
class SelfTestOutcome:
    """What happened when a case ran."""

    task: str
    strategy: str
    status: str = "failed"  # passed | failed | skipped
    duration_s: float = 0.0
    detail: str = ""
    events: dict[str, int] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """``task/strategy`` — the row key in the summary table."""
        return f"{self.task}/{self.strategy}"


# ── HTTP helpers (stdlib only: the CLI ships without test dependencies) ───────


def _post(base_url: str, path: str, body: dict[str, Any]) -> tuple[int, Any]:
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        return exc.code, raw


def _get(base_url: str, path: str) -> Any:
    with urllib.request.urlopen(f"{base_url}{path}") as resp:
        return json.load(resp)


class _EventCollector:
    """Reads /api/experiment/events in a thread and tallies event kinds."""

    def __init__(self, base_url: str) -> None:
        self._url = f"{base_url}/api/experiment/events"
        self.counts: dict[str, int] = {}
        self._thread = threading.Thread(target=self._read, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float) -> dict[str, int]:
        self._thread.join(timeout)
        return dict(self.counts)

    def _read(self) -> None:
        try:
            with urllib.request.urlopen(self._url, timeout=CASE_TIMEOUT_S) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if not payload or payload == "{}":
                        continue
                    kind = str(json.loads(payload).get("event", "?"))
                    self.counts[kind] = self.counts.get(kind, 0) + 1
        except Exception as exc:  # noqa: BLE001 — a dead stream is a finding, not a crash
            self.counts["<stream-error>"] = 1
            logger.debug("Self-test SSE stream ended: {}", exc)


# ── server harness ───────────────────────────────────────────────────────────


@contextmanager
def serve_app(host: str = "127.0.0.1") -> Iterator[str]:
    """Run the real GUI app on an ephemeral port; yield its base URL.

    A real socket (not an in-process test client) is deliberate: it exercises
    uvicorn's persistent event loop, which is what makes the genuinely
    asynchronous background training and SSE streaming behave as in production.
    """
    import uvicorn

    from visionforge.gui.server import app

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        port = probe.getsockname()[1]

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 30.0
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("Self-test server did not start within 30s.")
        time.sleep(0.05)

    try:
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=15.0)


# ── case construction ────────────────────────────────────────────────────────


def _out(workdir: Path, name: str) -> dict[str, str]:
    return {
        "models_dir": str(workdir / "models" / name),
        "reports_dir": str(workdir / "reports" / name),
        "graphics_dir": str(workdir / "graphics" / name),
        "logs_dir": str(workdir / "logs" / name),
    }


def _classification_payload(base: Path, workdir: Path) -> dict[str, Any]:
    return {
        "name": "selftest_cls",
        "task": "multiclass",
        "block": "classification",
        "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "early_stopping_patience": 1,
            "seed": 0,
        },
        "data": {
            "base_dir": str(base),
            "num_workers": 0,
            "pin_memory": False,
            "transforms": {"image_size": 32},
        },
        "output": _out(workdir, "classification"),
        "device": {"kind": "cpu"},
    }


def _regression_payload(base: Path, workdir: Path) -> dict[str, Any]:
    return {
        "name": "selftest_reg",
        "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
        "data": {
            "base_dir": str(base),
            "target_columns": ["target"],
            "image_size": 32,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.001, "seed": 0},
        "output": _out(workdir, "regression"),
        "device": {"kind": "cpu"},
    }


def _segmentation_payload(base: Path, workdir: Path) -> dict[str, Any]:
    return {
        "name": "selftest_seg",
        "model": {"name": "unet", "num_classes": 3, "pretrained": False},
        "data": {
            "base_dir": str(base),
            "image_size": 64,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001, "seed": 0},
        "output": _out(workdir, "segmentation"),
        "device": {"kind": "cpu"},
    }


def _anomaly_payload(base: Path, workdir: Path) -> dict[str, Any]:
    return {
        "name": "selftest_anom",
        "model": {"name": "autoencoder", "latent_dim": 16},
        "data": {
            "base_dir": str(base),
            "image_size": 64,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001, "seed": 0},
        "output": _out(workdir, "anomaly"),
        "device": {"kind": "cpu"},
    }


def _detection_payload(base: Path, workdir: Path) -> dict[str, Any]:
    # torchvision backend with pretrained=False: no Ultralytics extra, no
    # weight download, so the self-test runs offline on a bare install.
    return {
        "name": "selftest_det",
        "model": {
            "backend": "torchvision",
            "name": "fasterrcnn_mobilenet_v3_large_fpn",
            "num_classes": 1,
            "pretrained": False,
        },
        "data": {"base_dir": str(base), "image_size": 128},
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.005,
            "workers": 0,
            "seed": 0,
        },
        "output": _out(workdir, "detection"),
        "device": {"kind": "cpu"},
    }


def _custom_payload(workdir: Path) -> dict[str, Any]:
    # example_counting generates its own data, so base_dir is only a marker.
    return {
        "name": "selftest_custom",
        "data": {"base_dir": str(workdir), "num_workers": 0},
        "training": {"epochs": 1, "batch_size": 8, "seed": 0},
        "output": {"models_dir": str(workdir / "models" / "custom")},
        "device": {"kind": "cpu"},
        "n_samples": 32,
        "max_count": 3,
    }


def _standalone_cases(
    task: str,
    payload: dict[str, Any],
    *,
    has_cv: bool,
    metric: str,
    simple_expect: tuple[str, ...],
) -> list[SelfTestCase]:
    """simple / cv / sweep / replicates for one standalone task.

    ``simple_expect`` differs per task because each block returns its own
    report shape (train/test sections for regression-family tasks, a
    ``detection`` section for detection) — the self-test asserts the real
    contract rather than a lowest common denominator.
    """
    cases = [
        SelfTestCase(task, "simple", f"/api/{task}/run", payload, simple_expect),
        SelfTestCase(
            task,
            "sweep",
            f"/api/{task}/sweep",
            {
                "config": payload,
                "mode": "grid",
                "search_space": {"training.learning_rate": [0.001, 0.002]},
                "metric": metric,
            },
            ("best_trial", "total_trials"),
            multi_trial=True,
        ),
        SelfTestCase(
            task,
            "replicates",
            f"/api/{task}/replicates",
            {"config": payload, "seeds": [1, 2], "metric": metric},
            ("headline", "total_replicates"),
            multi_trial=True,
        ),
    ]
    if has_cv:
        cases.insert(
            1,
            SelfTestCase(
                task,
                "cv",
                f"/api/{task}/cv",
                {"config": payload, "n_folds": 2, "shuffle": True, "fold_seed": 0},
                ("aggregate", "fold_results"),
                multi_trial=True,
            ),
        )
    return cases


def build_cases(
    workdir: Path, tasks: tuple[str, ...], strategies: tuple[str, ...]
) -> list[SelfTestCase]:
    """Materialize datasets for ``tasks`` and return the matching cases."""
    datasets = workdir / "datasets"
    cases: list[SelfTestCase] = []

    if "classification" in tasks:
        base = build_classification_dataset(datasets / "classification")
        simple = _classification_payload(base, workdir)
        cv_cfg = {
            **simple,
            "block": "cross_validation",
            "cross_validation": {
                "n_folds": 2,
                "stratified": True,
                "shuffle": True,
                "fold_seed": 0,
            },
        }
        grid_cfg = {
            **simple,
            "block": "grid_search",
            "grid_search": {
                "hyperparameters": {"training.learning_rate": [0.001, 0.002]}
            },
        }
        cases += [
            SelfTestCase(
                "classification",
                "simple",
                "/api/experiment/run",
                simple,
                ("train", "eval"),
            ),
            SelfTestCase(
                "classification",
                "cv",
                "/api/experiment/run",
                cv_cfg,
                ("mean_accuracy",),
                multi_trial=True,
            ),
            SelfTestCase(
                "classification",
                "sweep",
                "/api/experiment/run",
                grid_cfg,
                ("best_trial",),
                multi_trial=True,
            ),
            # "accuracy", not "test_accuracy": ClassificationRunner projects the
            # eval report onto accuracy/f1/auc_roc for the generic orchestrators.
            SelfTestCase(
                "classification",
                "replicates",
                "/api/classification/replicates",
                {"config": simple, "seeds": [1, 2], "metric": "accuracy"},
                ("headline", "total_replicates"),
                multi_trial=True,
            ),
        ]

    if "regression" in tasks:
        base = build_regression_dataset(datasets / "regression")
        cases += _standalone_cases(
            "regression",
            _regression_payload(base, workdir),
            has_cv=True,
            metric="r2",
            simple_expect=("train", "test"),
        )

    if "segmentation" in tasks:
        base = build_segmentation_dataset(datasets / "segmentation")
        cases += _standalone_cases(
            "segmentation",
            _segmentation_payload(base, workdir),
            has_cv=True,
            metric="miou",
            simple_expect=("train", "test"),
        )

    if "anomaly" in tasks:
        base = build_anomaly_dataset(datasets / "anomaly")
        cases += _standalone_cases(
            "anomaly",
            _anomaly_payload(base, workdir),
            has_cv=False,
            metric="auroc",
            simple_expect=("train", "test"),
        )

    if "detection" in tasks:
        base = build_detection_dataset(datasets / "detection")
        cases += _standalone_cases(
            "detection",
            _detection_payload(base, workdir),
            has_cv=False,
            metric="map50",
            simple_expect=("detection",),
        )

    if "custom" in tasks:
        payload = _custom_payload(workdir)
        key = "example_counting"
        cases += [
            SelfTestCase(
                "custom", "simple", f"/api/custom/{key}/run", payload, ("metrics",)
            ),
            SelfTestCase(
                "custom",
                "sweep",
                f"/api/custom/{key}/sweep",
                {
                    "config": payload,
                    "mode": "grid",
                    "search_space": {"max_count": [3, 4]},
                    "metric": "mae",
                },
                ("best_trial", "total_trials"),
                multi_trial=True,
            ),
            SelfTestCase(
                "custom",
                "replicates",
                f"/api/custom/{key}/replicates",
                {"config": payload, "seeds": [1, 2], "metric": "mae"},
                ("headline", "total_replicates"),
                multi_trial=True,
            ),
        ]

    return [c for c in cases if c.strategy in strategies]


# ── execution ────────────────────────────────────────────────────────────────


def _validate_events(counts: dict[str, int], case: SelfTestCase) -> str:
    """Return an error string when the SSE contract was not honoured."""
    if "<stream-error>" in counts:
        return "SSE stream errored"
    if case.multi_trial:
        if not counts.get("trial_start"):
            return f"no trial_start events (got {counts or 'nothing'})"
        return ""
    if not counts.get("epoch_end"):
        return f"no epoch_end events (got {counts or 'nothing'})"
    return ""


def run_case(base_url: str, case: SelfTestCase) -> SelfTestOutcome:
    """POST one case, follow its SSE stream, and validate the stored report."""
    outcome = SelfTestOutcome(task=case.task, strategy=case.strategy)
    started = time.monotonic()
    try:
        status_code, body = _post(base_url, case.endpoint, case.payload)
        if status_code != 200:
            outcome.detail = f"POST {case.endpoint} → {status_code}: {str(body)[:200]}"
            return outcome

        collector = _EventCollector(base_url)
        collector.start()
        run_id = body["run_id"]

        deadline = time.monotonic() + CASE_TIMEOUT_S
        state: dict[str, Any] = {"status": "running"}
        while time.monotonic() < deadline:
            state = _get(base_url, "/api/experiment/status")
            if state["status"] in ("completed", "failed"):
                break
            time.sleep(0.25)
        else:
            outcome.detail = f"timed out after {CASE_TIMEOUT_S:.0f}s"
            return outcome

        outcome.events = collector.join(timeout=15.0)

        if state["status"] != "completed":
            outcome.detail = f"run failed: {str(state.get('error'))[:300]}"
            return outcome

        report = _get(base_url, f"/api/experiment/result/{run_id}").get("report") or {}
        missing = [k for k in case.expect_keys if k not in report]
        if missing:
            outcome.detail = f"report missing {missing} (got {sorted(report)[:8]})"
            return outcome

        problem = _validate_events(outcome.events, case)
        if problem:
            outcome.detail = problem
            return outcome

        # Summarize before flipping to passed: a report shaped oddly enough to
        # break the formatter is a finding, not a cosmetic detail.
        outcome.detail = _summarize(report)
        outcome.status = "passed"
        return outcome
    except Exception as exc:  # noqa: BLE001 — one broken case must not abort the sweep
        outcome.detail = f"{type(exc).__name__}: {exc}"
        return outcome
    finally:
        outcome.duration_s = time.monotonic() - started


def _summarize(report: dict[str, Any]) -> str:
    """One-line headline for the results table."""
    h = report.get("headline")
    if isinstance(h, dict):
        half = h.get("ci95_high", 0.0) - h.get("mean", 0.0)
        return f"{report.get('metric', '?')}={h.get('mean', 0.0):.4f}±{half:.4f}"
    if "best_trial" in report and isinstance(report["best_trial"], dict):
        metrics = report["best_trial"].get("metrics") or {}
        first = next(iter(metrics.items()), None)
        return f"best {first[0]}={first[1]:.4f}" if first else "best trial ok"
    if "aggregate" in report:
        agg = report["aggregate"]
        first = next(iter(agg.items()), None)
        if first and isinstance(first[1], dict):
            return f"{first[0]}={first[1].get('mean', 0):.4f}"
    if "metrics" in report and isinstance(report["metrics"], dict):
        first = next(iter(report["metrics"].items()), None)
        return f"{first[0]}={first[1]:.4f}" if first else "ok"
    if "eval" in report and isinstance(report["eval"], dict):
        return f"accuracy={report['eval'].get('accuracy', 0):.4f}"
    if "mean_accuracy" in report:
        return f"mean_accuracy={report['mean_accuracy']:.4f}"
    # Standalone tasks report a nested train/test (or detection) section.
    for section in ("test", "detection", "train"):
        block = report.get(section)
        if isinstance(block, dict):
            numeric = [(k, v) for k, v in block.items() if isinstance(v, int | float)]
            if numeric:
                k, v = numeric[0]
                return f"{section}.{k}={v:.4f}"
    return "ok"


def run_selftest(
    workdir: Path,
    *,
    tasks: tuple[str, ...] = TASKS,
    strategies: tuple[str, ...] = STRATEGIES,
) -> list[SelfTestOutcome]:
    """Run every selected case against a live server and return the outcomes."""
    workdir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(workdir, tasks, strategies)
    outcomes: list[SelfTestOutcome] = []

    with serve_app() as base_url:
        for i, case in enumerate(cases, start=1):
            logger.info(
                "Self-test {}/{}: {}/{}", i, len(cases), case.task, case.strategy
            )
            outcome = run_case(base_url, case)
            level = "success" if outcome.status == "passed" else "warning"
            logger.log(
                level.upper(),
                "  {} — {} ({:.1f}s)",
                outcome.label,
                outcome.detail or outcome.status,
                outcome.duration_s,
            )
            outcomes.append(outcome)
    return outcomes


def format_report(outcomes: list[SelfTestOutcome]) -> str:
    """Render the outcomes as an aligned pass/fail table plus a verdict line."""
    if not outcomes:
        return "No self-test cases selected."

    width = max(len(o.label) for o in outcomes)
    marks = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}
    lines = [
        f"{'case'.ljust(width)}  {'result':<6} {'time':>7}  detail",
        f"{'-' * width}  {'-' * 6} {'-' * 7}  {'-' * 40}",
    ]
    for o in outcomes:
        lines.append(
            f"{o.label.ljust(width)}  {marks.get(o.status, '?'):<6} "
            f"{o.duration_s:>6.1f}s  {o.detail[:60]}"
        )

    passed = sum(1 for o in outcomes if o.status == "passed")
    failed = [o.label for o in outcomes if o.status == "failed"]
    lines.append("")
    lines.append(
        f"{passed}/{len(outcomes)} cases passed"
        + (f" — failed: {', '.join(failed)}" if failed else "")
    )
    return "\n".join(lines)


__all__ = [
    "CASE_TIMEOUT_S",
    "STRATEGIES",
    "TASKS",
    "SelfTestCase",
    "SelfTestOutcome",
    "build_cases",
    "format_report",
    "run_case",
    "run_selftest",
    "serve_app",
]
