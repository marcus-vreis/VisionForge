"""Optional local MLflow export with replay-safe epoch records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any

from loguru import logger

_LOCK = RLock()
_METRICS = {
    "train_loss": "loss/train",
    "val_loss": "loss/val",
    "train_accuracy": "accuracy/train",
    "val_accuracy": "accuracy/val",
    "lr": "lr",
}
_STATUSES = {"FINISHED", "KILLED", "FAILED"}
_SIDECAR = "tracking.json"
# Keys never exported: secrets, and anything holding a local path (*_dir too,
# e.g. data.base_dir carries the user's folder and name).
_PRIVATE = re.compile(
    r"(?:token|password|secret|credential|path|uri|url)|(?:(?:^|_)dirs?$)", re.I
)


class MlflowTracker:
    """Export one local run to SQLite without changing its training outcome."""

    def __init__(self, store: Path, profile: str, run_dir: Path) -> None:
        self._store = store.absolute() / "mlflow.db"
        self._profile = profile
        self._run_dir = run_dir.absolute()
        self._identity = hashlib.sha256(
            f"{profile}\0{self._run_dir.resolve()}".encode()
        ).hexdigest()
        self._client: Any = None
        self._run_id: str | None = None
        self._saved_run_id: str | None = None
        self._state = "running"
        self._error: str | None = None
        self._status = "RUNNING"

    def start(self, config: dict[str, Any]) -> None:
        """Initialise an optional client and record non-sensitive scalar parameters."""
        with _LOCK:
            try:
                self._start(config)
            except Exception as exc:  # noqa: BLE001
                # Optional export boundary.
                self._fail(exc)

    def epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Export finite scalars once per metric and epoch, rejecting changed values."""
        with _LOCK:
            if self._error is not None or self._client is None:
                return
            try:
                self._epoch(epoch, metrics)
            except Exception as exc:  # noqa: BLE001
                # Optional export boundary.
                self._fail(exc)

    def finish(self, status: str) -> None:
        """Upload the final run.json and close the external run on every exit path."""
        with _LOCK:
            try:
                if status not in _STATUSES:
                    raise ValueError("Invalid terminal status")
                self._status = status
                if self._error is None and self._client is not None:
                    record = self._contained_file("run.json")
                    self._client.log_artifact(self._run_id, str(record))
                    self._state = "finished"
            except Exception as exc:  # noqa: BLE001
                # Optional export boundary.
                self._fail(exc)
            finally:
                if self._client is not None and self._run_id is not None:
                    try:
                        self._client.set_terminated(self._run_id, status=self._status)
                    except Exception as exc:  # noqa: BLE001
                        # Cleanup cannot mask training.
                        self._fail(exc)
                self._save_safely()

    def retry(self, record: dict[str, Any]) -> None:
        """Reconcile canonical history and re-upload the local record to the same run."""
        with _LOCK:
            self._error = None
            self._state = "running"
            try:
                saved = json.loads(self._contained_file(_SIDECAR).read_text("utf-8"))
                if saved.get("identity") != self._identity:
                    raise ValueError("Tracking record belongs to a different local run")
                self._start(record.get("config", {}))
                status = self._status
                if status == "RUNNING":
                    status = "KILLED" if record.get("stopped") else "FINISHED"
                for row in record.get("history", []):
                    metrics = {
                        key: value for key, value in row.items() if key != "epoch"
                    }
                    self._epoch(row["epoch"], metrics)
                self.finish(status)
            except Exception as exc:  # noqa: BLE001
                # Retry is best-effort too.
                self._fail(exc)

    def _start(self, config: dict[str, Any]) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", self._profile):
            raise ValueError("Invalid profile")
        if not self._run_dir.is_dir() or self._run_dir.is_symlink():
            raise ValueError("Invalid run directory")
        saved_run_id = self._load_sidecar()
        from mlflow import MlflowClient

        self._store.parent.mkdir(parents=True, exist_ok=True)
        self._client = MlflowClient(tracking_uri=f"sqlite:///{self._store.as_posix()}")
        experiment_id = self._experiment_id()
        self._run_id = (
            self._verified_run(saved_run_id, experiment_id)
            if saved_run_id
            else self._found_or_created_run(experiment_id)
        )
        self._save()
        for key, value in self._parameters(config).items():
            self._client.log_param(self._run_id, key, value)

    def _load_sidecar(self) -> str | None:
        """Restore the saved run ID and status; ``None`` when nothing was saved."""
        sidecar = self._run_dir / _SIDECAR
        if not (sidecar.exists() or sidecar.is_symlink()):
            return None
        saved = json.loads(self._contained_file(_SIDECAR).read_text("utf-8"))
        if saved.get("identity") != self._identity:
            raise ValueError("Tracking record belongs to a different local run")
        saved_run_id = saved.get("run_id")
        if saved_run_id is not None and not isinstance(saved_run_id, str):
            raise ValueError("Invalid saved run ID")
        self._saved_run_id = saved_run_id
        self._status = saved.get("status", "RUNNING")
        if self._status not in _STATUSES | {"RUNNING"}:
            raise ValueError("Invalid saved status")
        return saved_run_id

    def _experiment_id(self) -> str:
        name = f"visionforge-{self._profile}"
        experiment = self._client.get_experiment_by_name(name)
        if experiment:
            return str(experiment.experiment_id)
        artifacts = (self._store.parent / "artifacts").as_uri()
        return str(self._client.create_experiment(name, artifact_location=artifacts))

    def _verified_run(self, run_id: str, experiment_id: str) -> str:
        existing = self._client.get_run(run_id)
        if (
            existing.data.tags.get("vf.local_run") != self._identity
            or existing.info.experiment_id != experiment_id
        ):
            raise ValueError("External run identity does not match")
        return run_id

    def _found_or_created_run(self, experiment_id: str) -> str:
        matches = self._client.search_runs(
            [experiment_id],
            filter_string=f"tags.`vf.local_run` = '{self._identity}'",
        )
        if len(matches) > 1:
            raise ValueError("Ambiguous external run identity")
        if matches:
            return str(matches[0].info.run_id)
        tags = {"vf.local_run": self._identity, "vf.profile": self._profile}
        return str(self._client.create_run(experiment_id, tags=tags).info.run_id)

    def _epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("Invalid epoch")
        from mlflow.entities import Metric

        pending = []
        for name, value in metrics.items():
            key = _METRICS.get(name, name)
            if self._is_new_value(key, value, epoch):
                # Stable identity also covers a remote commit before local acknowledgement.
                pending.append(Metric(key, float(value), epoch, epoch))
        if pending:
            self._client.log_batch(self._run_id, metrics=pending, synchronous=True)

    def _is_new_value(self, key: str, value: Any, epoch: int) -> bool:
        """True when a finite scalar is not logged yet; refuses a changed value."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isfinite(value):
            return False
        if not re.fullmatch(r"[\w./ -]+", key, flags=re.ASCII):
            raise ValueError("Invalid metric key")
        history = self._client.get_metric_history(self._run_id, key)
        previous = {item.value for item in history if item.step == epoch}
        if previous and previous != {float(value)}:
            raise ValueError("Conflicting epoch value")
        return not previous

    def _contained_file(self, name: str) -> Path:
        path = self._run_dir / name
        if path.is_symlink() or path.resolve().parent != self._run_dir.resolve():
            raise ValueError("Artifact escapes run directory")
        if not path.is_file():
            raise FileNotFoundError(name)
        return path

    @staticmethod
    def _parameters(config: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        _collect_parameters(config, "", result)
        return result

    def _save(self) -> None:
        path = self._run_dir / _SIDECAR
        if path.is_symlink():
            raise ValueError("Unsafe tracking sidecar")
        payload = {
            "state": self._state,
            "error": self._error,
            "run_id": self._run_id or self._saved_run_id,
            "identity": self._identity,
            "status": self._status,
        }
        descriptor, temporary = tempfile.mkstemp(prefix=".tracking-", dir=self._run_dir)
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
            logger.warning("MLflow status could not be saved ({})", type(exc).__name__)

    def _fail(self, exc: Exception) -> None:
        self._state = "disabled" if isinstance(exc, ImportError) else "failed"
        self._error = type(exc).__name__
        logger.warning("MLflow export unavailable ({})", self._error)
        self._save_safely()


def _collect_parameters(
    data: dict[str, Any], prefix: str, result: dict[str, str]
) -> None:
    """Flatten scalar config values into ``result``, skipping private keys."""
    for name, value in sorted(data.items()):
        if not isinstance(name, str) or _PRIVATE.search(name):
            continue
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Ambiguous parameter name")
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            _collect_parameters(value, key, result)
        elif value is None or isinstance(value, (str, bool, int, float)):
            encoded = json.dumps(value, ensure_ascii=True, allow_nan=False)
            if len(key) <= 250 and len(encoded) <= 250:
                result[key] = encoded
