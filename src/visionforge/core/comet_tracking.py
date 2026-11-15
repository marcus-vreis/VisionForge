"""Optional Comet export of one run, private by default (ADR-200).

Same contract as the W&B exporter (ADR-172): every call is a best-effort
boundary that records a failure in ``comet.json`` beside the run and keeps
going. Before the first upload the project is created as, or confirmed to be,
private; a public project is refused, never changed. Everything the SDK would
collect on its own (code, git, console, packages, host, system metrics,
framework hooks, its own error reports) is switched off.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from loguru import logger

from visionforge.core.tracking_privacy import (
    TRACKER_METRIC_KEYS,
    public_parameters,
    tracker_metrics,
)
from visionforge.utils.comet_names import is_valid_comet_name

_LOCK = RLock()
_SIDECAR = "comet.json"
_STATUSES = {"FINISHED", "KILLED", "FAILED"}
# Read by comet_ml from the environment. Auto-logging patches the import of
# torch, keras and other frameworks; error tracking sends Sentry reports with
# the Python path and the installed packages; internal reporting sends the
# SDK's tracebacks to Comet. All three stay off.
IMPORT_SWITCHES = {
    "COMET_DISABLE_AUTO_LOGGING": "1",
    "COMET_ERROR_TRACKING_ENABLE": "false",
    "COMET_INTERNAL_REPORTING": "false",
}
# Comet keeps one live experiment per process and ends the previous one when a
# new one starts: a second export fails and waits for the History's retry.
_OPEN: set[str] = set()


class CometTokenMissingError(ValueError):
    """The profile has the export on but no saved key."""


class CometPrivateProjectError(ValueError):
    """The destination project is public."""


class CometRequestError(RuntimeError):
    """A Comet call failed; the message never carries the response or key."""


class CometBusyError(RuntimeError):
    """Another Comet export of this process is still open."""


@dataclass(frozen=True)
class CometTarget:
    """Where a profile's runs go: ``workspace`` empty means the key's default."""

    project: str
    workspace: str = ""
    token: str | None = None


def import_comet() -> Any:
    """``comet_ml``, imported with its auto-logging and error reports off."""
    os.environ.update(IMPORT_SWITCHES)
    import comet_ml

    return comet_ml


def ensure_private_project(api: Any, workspace: str, project: str) -> None:
    """Create ``workspace/project`` as private, or confirm the existing one is.

    Raises:
        CometPrivateProjectError: the project is public, or the server did not
            report it as private after creating it.
    """
    found = api.get_project(workspace, project)
    if found is None:
        api.create_project(workspace, project, public=False)
        found = api.get_project(workspace, project)
    if not isinstance(found, dict) or found.get("isPublic") is not False:
        raise CometPrivateProjectError("The Comet project is not private")


def _no_notification(*_args: Any, **_kwargs: Any) -> None:
    """Stand-in for ``send_notification``, whose title holds the OS user name."""


class CometTracker:
    """Export one local run to Comet without changing its training outcome."""

    def __init__(self, profile: str, run_dir: Path, target: CometTarget) -> None:
        self._profile = profile
        self._run_dir = run_dir.absolute()
        self._target = target
        self._identity = hashlib.sha256(
            f"{profile}\0{self._run_dir.resolve()}".encode()
        ).hexdigest()
        self._experiment: Any = None
        self._state = "running"
        self._error: str | None = None
        self._status = "RUNNING"
        self._url: str | None = None
        self._workspace = target.workspace
        # Epochs Comet confirmed (flushed); a retry sends only the later ones.
        self._synced = 0
        self._logged = 0

    def start(self, config: dict[str, Any]) -> None:
        """Open the private experiment and record the public parameters."""
        with _LOCK:
            try:
                self._start(config)
            except Exception as exc:  # noqa: BLE001
                # Optional export boundary.
                self._fail(exc)
                self._close()

    def epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Log the finite scalars of one epoch at step ``epoch``."""
        with _LOCK:
            if self._experiment is None:
                return
            try:
                self._epoch(epoch, metrics)
            except Exception as exc:  # noqa: BLE001
                # Optional export boundary.
                self._fail(exc)

    def finish(self, status: str) -> None:
        """Write the final public metrics and close the experiment on every path."""
        with _LOCK:
            try:
                if status not in _STATUSES:
                    raise ValueError("Invalid terminal status")
                self._status = status
                if self._experiment is not None:
                    self._finish()
            except Exception as exc:  # noqa: BLE001
                # Optional export boundary.
                self._fail(exc)
            finally:
                self._close()
                self._save_safely()

    def retry(self, record: dict[str, Any]) -> None:
        """Reopen the same experiment and send the epochs Comet did not confirm."""
        with _LOCK:
            self._error = None
            self._state = "running"
            try:
                self._start(record.get("config", {}))
                status = self._status
                if status == "RUNNING":
                    status = "KILLED" if record.get("stopped") else "FINISHED"
                for row in record.get("history", []):
                    metrics = {k: v for k, v in row.items() if k != "epoch"}
                    self._epoch(row["epoch"], metrics)
                self.finish(status)
            except Exception as exc:  # noqa: BLE001
                # Retry is best-effort too.
                self._fail(exc)
                self._close()

    def _start(self, config: dict[str, Any]) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", self._profile):
            raise ValueError("Invalid profile")
        if not self._run_dir.is_dir() or self._run_dir.is_symlink():
            raise ValueError("Invalid run directory")
        self._load_sidecar()
        token = (self._target.token or "").strip()
        if not token:
            raise CometTokenMissingError("No Comet key in this profile")
        if not is_valid_comet_name(self._target.project):
            raise ValueError("Invalid Comet project")
        if _OPEN - {self._identity}:
            raise CometBusyError("Another Comet export is open")
        comet_ml = import_comet()
        # Only this key: a saved ``comet login`` or COMET_API_KEY is never read.
        api = comet_ml.API(api_key=token, cache=False)
        workspace = self._target.workspace or api.get_default_workspace()
        if not isinstance(workspace, str) or not is_valid_comet_name(workspace):
            raise CometRequestError("The key has no Comet workspace")
        self._workspace = workspace
        ensure_private_project(api, workspace, self._target.project)
        experiment = comet_ml.start(
            api_key=token,
            workspace=workspace,
            project_name=self._target.project,
            experiment_key=self._identity[:32],
            mode="get_or_create",
            online=True,
            experiment_config=self._experiment_config(comet_ml),
        )
        # Kept before anything else can fail, so every path ends it.
        self._experiment = experiment
        _OPEN.add(self._identity)
        experiment.send_notification = _no_notification
        if not getattr(experiment, "alive", False):
            raise CometRequestError("The Comet experiment did not start")
        experiment.log_parameters(public_parameters(config))
        url = getattr(experiment, "url", None)
        self._url = url if isinstance(url, str) and url.startswith("https://") else None
        self._save()

    def _experiment_config(self, comet_ml: Any) -> Any:
        # Nothing about this machine leaves it, and no ambient COMET_* setting
        # can turn a collector back on: each one is an explicit argument.
        return comet_ml.ExperimentConfig(
            disabled=False,
            log_code=False,
            log_graph=False,
            parse_args=False,
            display_summary_level=0,
            log_git_metadata=False,
            log_git_patch=False,
            log_env_details=False,
            log_env_gpu=False,
            log_env_host=False,
            log_env_cpu=False,
            log_env_network=False,
            log_env_disk=False,
            auto_output_logging=False,
            auto_param_logging=False,
            auto_metric_logging=False,
            auto_log_co2=False,
            auto_histogram_tensorboard_logging=False,
            auto_histogram_weight_logging=False,
            auto_histogram_gradient_logging=False,
            auto_histogram_activation_logging=False,
            name=self._run_dir.name,
        )

    def _epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("Invalid epoch")
        if epoch <= self._synced:
            return  # Comet confirmed it already: a retry adds the missing epochs.
        values = {
            TRACKER_METRIC_KEYS.get(name, name): float(value)
            for name, value in metrics.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        }
        if values:
            self._experiment.log_metrics(values, step=epoch, epoch=epoch)
            self._logged = max(self._logged, epoch)

    def _finish(self) -> None:
        final = self._final_metrics()
        if final:
            self._experiment.log_metrics(final)
        self._experiment.log_other("visionforge_status", self._status)
        if not self._experiment.flush():
            raise CometRequestError("Comet did not confirm the upload")
        self._synced = max(self._synced, self._logged)
        self._state = "finished"

    def _final_metrics(self) -> dict[str, int | float]:
        path = self._run_dir / "run.json"
        if path.is_symlink() or not path.is_file():
            return {}
        return tracker_metrics(json.loads(path.read_text("utf-8")))

    def _close(self) -> None:
        experiment, self._experiment = self._experiment, None
        _OPEN.discard(self._identity)
        if experiment is None:
            return
        try:
            experiment.end()
        except Exception as exc:  # noqa: BLE001
            # Cleanup cannot mask training.
            self._fail(exc)

    def _load_sidecar(self) -> None:
        sidecar = self._run_dir / _SIDECAR
        if not (sidecar.exists() or sidecar.is_symlink()):
            return
        if sidecar.is_symlink():
            raise ValueError("Unsafe Comet sidecar")
        saved = json.loads(sidecar.read_text("utf-8"))
        if saved.get("identity") != self._identity:
            raise ValueError("Comet record belongs to a different local run")
        status = saved.get("status", "RUNNING")
        if status not in _STATUSES | {"RUNNING"}:
            raise ValueError("Invalid saved status")
        synced = saved.get("synced_epoch", 0)
        if not isinstance(synced, int) or isinstance(synced, bool) or synced < 0:
            raise ValueError("Invalid saved epoch")
        self._status = status
        self._synced = self._logged = synced

    def _save(self) -> None:
        path = self._run_dir / _SIDECAR
        if path.is_symlink():
            raise ValueError("Unsafe Comet sidecar")
        payload = {
            "state": self._state,
            "error": self._error,
            "identity": self._identity,
            "status": self._status,
            "url": self._url,
            "project": self._target.project,
            "workspace": self._workspace or None,
            "synced_epoch": self._synced,
        }
        descriptor, temporary = tempfile.mkstemp(prefix=".comet-", dir=self._run_dir)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _save_safely(self) -> None:
        try:
            self._save()
        except Exception as exc:  # noqa: BLE001
            # Full disk cannot fail training.
            logger.warning("Comet status could not be saved ({})", type(exc).__name__)

    def _fail(self, exc: Exception) -> None:
        self._state = "disabled" if isinstance(exc, ImportError) else "failed"
        self._error = type(exc).__name__
        logger.warning("Comet export unavailable ({})", self._error)
        self._save_safely()
