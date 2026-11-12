from __future__ import annotations

import json
import re
from pathlib import Path

import export_runs
import pytest
from export_runs import public_record, run_name, trial_run_dirs

RUNS = Path(__file__).resolve().parent / "runs"
# A drive path (C:\ or C:/), a UNC share, or a POSIX home/account folder.
ABSOLUTE_PATH = re.compile(r"\b[A-Za-z]:[\\/]|\\\\\\\\|/(?:home|Users|root)/")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _exported() -> list[Path]:
    return sorted(RUNS.glob("*.json"))


class TestPublishedRuns:
    def test_every_trial_of_the_grid_is_published(self) -> None:
        names = {run_name(d) for d in trial_run_dirs()}
        assert len(names) == 270
        assert {p.stem for p in _exported()} == names

    def test_each_record_matches_its_trial_in_the_report(self) -> None:
        results = Path(__file__).resolve().parent / "results"
        for report_path in sorted(results.glob("*__*.json")):
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for variant in report["variants"].values():
                for trial in variant["trials"]:
                    path = RUNS / f"{run_name(trial['run_dir'])}.json"
                    record = json.loads(path.read_text(encoding="utf-8"))
                    accuracy = record["metrics"]["test_accuracy"]
                    assert accuracy == trial["metrics"]["accuracy"], path.name
                    assert record["config"]["training"]["seed"] == trial["seed"]

    def test_no_local_path_username_or_email_in_any_file(self) -> None:
        offenders = []
        for path in _exported():
            text = path.read_text(encoding="utf-8")
            if ABSOLUTE_PATH.search(text) or EMAIL.search(text):
                offenders.append(path.name)
            elif "marcus" in text.lower():
                offenders.append(path.name)
        assert offenders == []

    def test_published_runs_stay_small(self) -> None:
        total = sum(p.stat().st_size for p in _exported())
        assert total <= 3_000_000


class TestPublicRecord:
    def test_paths_are_reduced_to_final_names(self) -> None:
        data = {
            "run_dir": "C:\\Users\\someone\\proj\\outputs\\models\\x\\1",
            "dataset_fingerprint": {"root": "C:\\Users\\someone\\data\\Beans"},
            "artifacts": {
                "model": "outputs\\models\\x\\1\\best_model.pth",
                "graphics": ["outputs\\models\\x\\1\\loss.png"],
                "report": None,
            },
            "metrics": {"test_accuracy": 0.5},
        }
        record = public_record(data)
        assert record["run_dir"] == "."
        assert record["dataset_fingerprint"]["root"] == "Beans"
        assert record["artifacts"] == {
            "model": "best_model.pth",
            "graphics": ["loss.png"],
            "report": None,
        }
        assert record["metrics"] == {"test_accuracy": 0.5}

    def test_run_name_is_the_experiment_folder(self) -> None:
        assert run_name("outputs\\models\\seedvar_a_s0\\20261015_1") == "seedvar_a_s0"


class TestMain:
    @pytest.fixture
    def grid(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        results = tmp_path / "results"
        results.mkdir()
        trials = [
            {"seed": 0, "run_dir": "outputs\\models\\exp_s0\\1"},
            {"seed": 1, "run_dir": "outputs\\models\\exp_s1\\2"},
        ]
        report = {"variants": {"baseline": {"trials": trials}}}
        (results / "a__b.json").write_text(json.dumps(report), encoding="utf-8")
        run = tmp_path / "outputs" / "models" / "exp_s0" / "1"
        run.mkdir(parents=True)
        record = {"run_dir": str(run), "artifacts": {}, "metrics": {"x": 1}}
        (run / "run.json").write_text(json.dumps(record), encoding="utf-8")
        monkeypatch.setattr(export_runs, "RESULTS", results)
        monkeypatch.setattr(export_runs, "RUNS", tmp_path / "runs")
        return tmp_path

    def test_exports_found_runs_and_flags_missing(self, grid: Path) -> None:
        code = export_runs.main(["--root", str(grid)])
        assert code == 1
        written = json.loads((grid / "runs" / "exp_s0.json").read_text("utf-8"))
        assert written == {"run_dir": ".", "artifacts": {}, "metrics": {"x": 1}}
        assert not (grid / "runs" / "exp_s1.json").exists()
