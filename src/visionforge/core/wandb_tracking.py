"""Optional Weights & Biases export of one run, private by default (ADR-172).

The export never decides the training: every call is a best-effort boundary that
records a failure in ``wandb.json`` beside the run and keeps going. Before the
first upload the project is created as, or confirmed to be, ``PRIVATE``; a
project with any other visibility is refused, never changed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from loguru import logger

from visionforge.core.tracking_privacy import (
    TRACKER_METRIC_KEYS,
    public_metrics,
    public_parameters,
)

_LOCK = RLock()
_SIDECAR = "wandb.json"
_STATUSES = {"FINISHED", "KILLED", "FAILED"}
_NAME = re.compile(r"[\w.-]{1,128}", re.ASCII)
_PRIVATE = "PRIVATE"
_TIMEOUT_S = 30

_VIEWER = "query Viewer { viewer { entity } }"
_PROJECT = (
    "query Project($name: String!, $entity: String!) "
    "{ project(name: $name, entityName: $entity) { access } }"
)
_CREATE = (
    "mutation CreateProject($input: UpsertModelInput!) "
    "{ upsertModel(input: $input) { project { access } } }"
)

GraphQL = Callable[[str, dict[str, Any]], dict[str, Any]]


class WandbTokenMissingError(ValueError):
    """The profile has the export on but no saved token."""


class WandbPrivateProjectError(ValueError):
    """The destination project exists and is not private."""


class WandbRequestError(RuntimeError):
    """A W&B API call failed; the message never carries the response or token."""


@dataclass(frozen=True)
class WandbTarget:
    """Where a profile's runs go: ``entity`` empty means the token's own."""

    project: str
    entity: str = ""
    token: str | None = None


def is_valid_wandb_name(value: str) -> bool:
    """True for a project or entity name W&B accepts in a URL path."""
    return bool(_NAME.fullmatch(value)) and value not in {".", ".."}


def ensure_private_project(graphql: GraphQL, entity: str, project: str) -> None:
    """Create ``entity/project`` as private, or confirm the existing one is.

    Raises:
        WandbPrivateProjectError: the project exists with another visibility,
            or the server created it with one.
    """
    found = graphql(_PROJECT, {"name": project, "entity": entity}).get("project")
    if found is None:
        created = graphql(
            _CREATE,
            {"input": {"name": project, "entityName": entity, "access": _PRIVATE}},
        )
        found = (created.get("upsertModel") or {}).get("project") or {}
    if found.get("access") != _PRIVATE:
        raise WandbPrivateProjectError("The W&B project is not private")


def http_graphql(base_url: str, token: str) -> GraphQL:
    """A GraphQL caller for ``base_url`` with only this token, no ambient login."""
    credential = base64.b64encode(f"api:{token}".encode()).decode("ascii")
    endpoint = f"{base_url.rstrip('/')}/graphql"
    if not endpoint.startswith("https://"):
        raise WandbRequestError("The W&B address must use HTTPS")

    def call(query: str, variables: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps({"query": query, "variables": variables}).encode()
        request = urllib.request.Request(  # noqa: S310
            # The scheme is checked above: only https reaches urlopen.
            endpoint,
            data=body,
            headers={
                "Authorization": f"Basic {credential}",
                "Content-Type": "application/json",
                "User-Agent": "visionforge",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:  # noqa: S310
                # Same https endpoint as above.
                payload = json.loads(response.read())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise WandbRequestError(type(exc).__name__) from None
        if not isinstance(payload, dict) or payload.get("errors"):
            raise WandbRequestError("The W&B API refused the request")
        data = payload.get("data")
        return data if isinstance(data, dict) else {}

    return call


class WandbTracker:
    """Export one local run to W&B without changing its training outcome."""

    # The real check in the scratchpad runs offline; the GUI never changes it.
    _MODE = "online"

    def __init__(
        self,
        store: Path,
        profile: str,
        run_dir: Path,
        target: WandbTarget,
        graphql: GraphQL | None = None,
    ) -> None:
        self._store = store.absolute() / profile
        self._profile = profile
        self._run_dir = run_dir.absolute()
        self._target = target
        self._graphql = graphql
        self._identity = hashlib.sha256(
            f"{profile}\0{self._run_dir.resolve()}".encode()
        ).hexdigest()
        self._run: Any = None
        self._state = "running"
        self._error: str | None = None
        self._status = "RUNNING"
        self._url: str | None = None
        self._entity = target.entity

    def start(self, config: dict[str, Any]) -> None:
        """Open the private run and record the public parameters."""
        with _LOCK:
            try:
                self._start(config)
            except Exception as exc:  # noqa: BLE001
                # Optional export boundary.
                self._fail(exc)

    def epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Log the finite scalars of one epoch at step ``epoch``."""
        with _LOCK:
            if self._run is None:
                return
            try:
                self._epoch(epoch, metrics)
            except Exception as exc:  # noqa: BLE001
                # Optional export boundary.
                self._fail(exc)

    def finish(self, status: str) -> None:
        """Write the final public metrics and close the run on every exit path."""
        with _LOCK:
            try:
                if status not in _STATUSES:
                    raise ValueError("Invalid terminal status")
                self._status = status
                if self._run is not None:
                    self._run.summary.update(self._final_metrics())
                    self._state = "finished"
            except Exception as exc:  # noqa: BLE001
                # Optional export boundary.
                self._fail(exc)
            finally:
                self._close()
                self._save_safely()

    def retry(self, record: dict[str, Any]) -> None:
        """Resume the same W&B run and send the saved history again."""
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
            raise WandbTokenMissingError("No W&B token in this profile")
        if not is_valid_wandb_name(self._target.project):
            raise ValueError("Invalid W&B project")
        import wandb

        base_url = wandb.Settings.model_fields["base_url"].default
        graphql = self._graphql or http_graphql(base_url, token)
        if self._MODE == "online":
            self._entity = self._target.entity or self._viewer_entity(graphql)
            ensure_private_project(graphql, self._entity, self._target.project)
        self._store.mkdir(parents=True, exist_ok=True)
        self._run = wandb.init(
            entity=self._entity or None,
            project=self._target.project,
            id=self._identity[:32],
            name=self._run_dir.name,
            dir=str(self._store),
            resume="allow",
            reinit="create_new",
            config=public_parameters(config),
            settings=self._settings(wandb, base_url, token),
        )
        url = getattr(self._run, "url", None)
        self._url = url if isinstance(url, str) and url.startswith("https://") else None
        self._save()

    def _settings(self, wandb: Any, base_url: str, token: str) -> Any:
        # Nothing about this machine leaves it: no git, code, console, system
        # stats, host metadata or package list, and no reading of a saved login.
        return wandb.Settings(
            api_key=token,
            base_url=base_url,
            mode=self._MODE,
            console="off",
            disable_git=True,
            disable_code=True,
            save_code=False,
            silent=True,
            x_disable_meta=True,
            x_disable_machine_info=True,
            x_disable_stats=True,
            x_save_requirements=False,
        )

    @staticmethod
    def _viewer_entity(graphql: GraphQL) -> str:
        entity = (graphql(_VIEWER, {}).get("viewer") or {}).get("entity")
        if not isinstance(entity, str) or not is_valid_wandb_name(entity):
            raise WandbRequestError("The token has no W&B entity")
        return entity

    def _epoch(self, epoch: int, metrics: dict[str, Any]) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("Invalid epoch")
        values = {
            TRACKER_METRIC_KEYS.get(name, name): float(value)
            for name, value in metrics.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        }
        if values:
            # A resumed run drops steps it already has: a retry adds only
            # the missing epochs.
            self._run.log(values, step=epoch)

    def _final_metrics(self) -> dict[str, int | float]:
        path = self._run_dir / "run.json"
        if path.is_symlink() or not path.is_file():
            return {}
        record = json.loads(path.read_text("utf-8"))
        return public_metrics(
            record.get("metrics") if isinstance(record, dict) else None
        )

    def _close(self) -> None:
        run, self._run = self._run, None
        if run is None:
            return
        try:
            run.finish(exit_code=0 if self._status == "FINISHED" else 1)
        except Exception as exc:  # noqa: BLE001
            # Cleanup cannot mask training.
            self._fail(exc)

    def _load_sidecar(self) -> None:
        sidecar = self._run_dir / _SIDECAR
        if not (sidecar.exists() or sidecar.is_symlink()):
            return
        if sidecar.is_symlink():
            raise ValueError("Unsafe W&B sidecar")
        saved = json.loads(sidecar.read_text("utf-8"))
        if saved.get("identity") != self._identity:
            raise ValueError("W&B record belongs to a different local run")
        status = saved.get("status", "RUNNING")
        if status not in _STATUSES | {"RUNNING"}:
            raise ValueError("Invalid saved status")
        self._status = status

    def _save(self) -> None:
        path = self._run_dir / _SIDECAR
        if path.is_symlink():
            raise ValueError("Unsafe W&B sidecar")
        payload = {
            "state": self._state,
            "error": self._error,
            "identity": self._identity,
            "status": self._status,
            "url": self._url,
            "project": self._target.project,
            "entity": self._entity or None,
        }
        descriptor, temporary = tempfile.mkstemp(prefix=".wandb-", dir=self._run_dir)
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
            logger.warning("W&B status could not be saved ({})", type(exc).__name__)

    def _fail(self, exc: Exception) -> None:
        self._state = "disabled" if isinstance(exc, ImportError) else "failed"
        self._error = type(exc).__name__
        logger.warning("W&B export unavailable ({})", self._error)
        self._save_safely()
