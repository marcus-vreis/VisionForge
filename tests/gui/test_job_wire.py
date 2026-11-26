"""The JSON crossing of a strategy's call between the server and its child (ADR-252)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from visionforge.core.cancellation import CancellationToken
from visionforge.core.comparison import ComparisonTrial
from visionforge.gui.api import job_wire
from visionforge.gui.api.job_call import RawConfig, call_spec, kfold_report, trials_of
from visionforge.gui.api.job_worker import _load


def _across(value: Any) -> Any:
    """``value`` written, sent as one JSON line and read back, with no argument tags."""
    line = json.dumps(job_wire.dump_value(value))
    return job_wire.load_value(json.loads(line))


class _Config(BaseModel):
    name: str = "c"
    out: Path = Path("out")


class _Runner:
    config_type = _Config
    _cancel_token: CancellationToken | None = None
    _progress_callback: Any = None

    def run(self, cfg: Any) -> None:
        return None


@dataclass
class _Row:
    seed: int
    metrics: dict[str, float] = field(default_factory=dict)
    run_dir: Path | None = None


def _orchestrator(*args: Any, **kwargs: Any) -> None:
    return None


def test_plain_values_cross_unchanged_and_nan_and_infinity_survive() -> None:
    value = {"a": [1, 2.5, "x", None, True], "b": {"c": float("inf")}}
    assert _across(value) == value
    back = _across({"loss": float("nan"), "low": -math.inf})
    assert math.isnan(back["loss"])
    assert back["low"] == -math.inf


def test_paths_tuples_and_int_keys_come_back_as_they_were(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    back = _across({"run": Path("runs/a"), "pair": (1, "b"), "per_seed": {42: 0.9}})
    # A relative path is made absolute where it is written: the server's folder.
    assert back == {
        "run": tmp_path / "runs" / "a",
        "pair": (1, "b"),
        "per_seed": {42: 0.9},
    }
    assert isinstance(back["run"], Path)


def test_a_dict_from_a_request_that_looks_like_a_tag_stays_data() -> None:
    sneaky = {"$model": {"type": "os:system", "data": {}}, "$runner": "os:getcwd"}
    assert _across(sneaky) == sneaky
    assert _across({"$path": "x"}) == {"$path": "x"}


def test_a_dataclass_crosses_as_its_fields_and_is_rebuilt_by_the_decoder() -> None:
    trial = ComparisonTrial("resnet18", "success", {"acc": float("nan")}, 1.5, "", "r")
    rows = _across([trial])
    assert rows[0]["model_arch"] == "resnet18"
    (back,) = trials_of(ComparisonTrial)(rows)
    assert isinstance(back, ComparisonTrial)
    assert (back.model_arch, back.run_dir, back.training_time_s) == (
        "resnet18",
        "r",
        1.5,
    )
    assert math.isnan(back.metrics["acc"])
    row = _across(_Row(1, {"r2": 0.5}, Path("/abs/run")))
    assert row["run_dir"] == Path("/abs/run").absolute()


def test_a_kfold_report_is_rebuilt_with_its_folds() -> None:
    from visionforge.blocks._cv_common import CrossValidationReport, FoldResult

    report = CrossValidationReport(
        n_folds=2,
        metric="r2",
        folds=[FoldResult(0, "success", 8, 2, {"r2": 0.5}, run_dir="f0")],
        aggregate={"r2": {"mean": 0.5, "std": None, "n": 1}},
        select_on="last",
    )
    back = kfold_report(_across(report))
    assert back == report
    assert isinstance(back.folds[0], FoldResult)


def test_a_numpy_scalar_becomes_its_number_and_an_array_is_refused() -> None:
    assert _across({"acc": np.float32(0.25), "n": np.int64(3)}) == {"acc": 0.25, "n": 3}
    with pytest.raises(TypeError, match="ndarray"):
        job_wire.dump_value(np.zeros(2))


def test_an_unknown_object_or_tag_is_refused() -> None:
    with pytest.raises(TypeError, match="object"):
        job_wire.dump_value(object())
    with pytest.raises(ValueError, match=r"\$nope"):
        job_wire.load_value({"$nope": 1})


def test_the_call_crosses_with_its_runner_config_token_and_pump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    server_token = CancellationToken()
    runner = _Runner()
    runner._cancel_token = server_token  # as _run_group_job hands it over

    def progress(event: dict[str, Any]) -> None:
        return None

    spec = call_spec(
        _orchestrator,
        (
            runner,
            RawConfig({"name": "c", "out": "out"}, runner),
            _Config(out=Path("cfg")),
        ),
        {"cancel_token": server_token, "progress_callback": progress, "alpha": 0.05},
        progress,
    )
    # The spec is one JSON line; the names in it are read off the server's objects.
    spec = json.loads(json.dumps(spec))
    assert spec["kind"] == "call"
    assert spec["function"] == f"{__name__}:_orchestrator"

    child_token = CancellationToken()
    events: list[dict[str, Any]] = []
    read = job_wire.child_arguments(_load, events.append, child_token)
    args = job_wire.load_value(spec["args"], read)
    kwargs = job_wire.load_value(spec["kwargs"], read)

    child_runner, config_dict, config = args
    assert isinstance(child_runner, _Runner) and child_runner is not runner
    assert child_runner._cancel_token is child_token
    # The dict is validated and every path pinned to the server's folder.
    assert config_dict == {"name": "c", "out": str(tmp_path / "out")}
    assert config == _Config(out=tmp_path / "cfg")
    assert kwargs["cancel_token"] is child_token
    assert kwargs["alpha"] == 0.05
    kwargs["progress_callback"]({"event": "epoch_end"})
    assert events == [{"event": "epoch_end"}]


def test_what_cannot_be_rebuilt_in_a_child_is_refused_before_it_starts() -> None:
    stateful = _Runner()
    stateful.task = "custom"  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="task"):
        call_spec(_orchestrator, (stateful,), {}, None)
    with pytest.raises(TypeError, match="not importable"):
        call_spec(lambda: None, (), {}, None)
    with pytest.raises(TypeError, match="function"):
        call_spec(_orchestrator, (), {"callback": _orchestrator}, None)
