"""Switch off the usage telemetry of the third-party SDKs VisionForge bundles.

The README promises that nothing leaves this computer unless you ask for it
(ADR-120, ADR-170). Some SDKs we depend on report usage on their own, so this
module turns that off, once, before any of them is imported.

``apply_opt_outs`` is idempotent and never overrides a value the user already
set: a user who wants a SDK's telemetry on keeps it on.

``NETWORK_SDKS`` lists every dependency that can reach the network, and says
how it is covered. ``tests/utils/test_third_party_privacy.py`` fails when
``pyproject.toml`` gains a known network SDK that has no entry here.
"""

from __future__ import annotations

import os

# Environment switches honoured by the SDKs, set only when the user did not.
_ENV_OPT_OUTS: dict[str, str] = {
    "HF_HUB_DISABLE_TELEMETRY": "1",  # huggingface_hub download telemetry
    "DO_NOT_TRACK": "1",  # community convention, read by several SDKs
    "MLFLOW_DISABLE_TELEMETRY": "true",  # MLflow usage data (3.2+)
    "DVC_NO_ANALYTICS": "1",  # DVC; dvc_export.py also sets it per subprocess
}

# Dependency name -> how it is covered. Not YOLO_OFFLINE: that would also stop
# the weights download the user asks for.
NETWORK_SDKS: dict[str, str] = {
    "ultralytics": "events.enabled=False in-process (events for train/val/export)",
    "huggingface-hub": "HF_HUB_DISABLE_TELEMETRY",
    "datasets": "HF_HUB_DISABLE_TELEMETRY (uses huggingface_hub)",
    "mlflow": "MLFLOW_DISABLE_TELEMETRY",
    "dvc": "DVC_NO_ANALYTICS",
    "wandb": "opt-in only: sends nothing until the user asks for it",
    "roboflow": "dataset download the user asks for; no telemetry",
    "kaggle": "dataset download the user asks for; no telemetry",
}


def apply_env_opt_outs() -> None:
    """Set only the environment switches; imports no SDK.

    The GUI server calls this on import: it no longer imports the trainers, so
    it must not pay for ultralytics (and torch) just to switch telemetry off.
    """
    for name, value in _ENV_OPT_OUTS.items():
        os.environ.setdefault(name, value)


def apply_opt_outs() -> None:
    """Turn off third-party telemetry; keep any value the user already set."""
    apply_env_opt_outs()
    _disable_ultralytics_sync()


def _disable_ultralytics_sync() -> None:
    """Stop ultralytics' usage events in this process only.

    Its ``settings.update({"sync": False})`` would write the user's global
    ultralytics settings file and switch the events off for every other project
    on the machine too. The events go through one ``Events`` instance whose
    ``enabled`` flag is read on each call, so turning that off here keeps the
    choice inside VisionForge.
    """
    try:
        from ultralytics.utils.events import events

        events.enabled = False
    except Exception:  # noqa: BLE001 - absent or broken SDK must not stop the app
        return
