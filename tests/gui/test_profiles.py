"""Profiles for a shared server (ADR-114): a profile is an output folder.

Three layers are covered. The folder logic (slug rules, create, list, resolve)
is tested on ``gui/api/profiles.py`` directly. The HTTP contract (the
``X-VF-Profile`` header, what a bad one answers) and the History actions run
through the real app. And every submission route is proven to hand its executor
a config whose output folders sit under the profile, which is what keeps one
person's runs out of another's History.

Every test runs with the working directory moved to ``tmp_path``: the default
profile is ``outputs/models`` and the others ``outputs/profiles/<slug>``, both
relative to where the GUI was started, so nothing here touches the repo.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from visionforge.gui.api.profiles import (
    OUTPUT_SUBDIRS,
    InvalidProfileError,
    ProfileCreateError,
    ProfileExistsError,
    ProfileNameError,
    UnknownProfileError,
    create_profile,
    default_profile,
    is_valid_slug,
    list_profiles,
    normalize_display_name,
    resolve_profile,
    slugify,
)
from visionforge.gui.api.schemas import RunResponse
from visionforge.utils.config import OutputConfig

from .conftest import occupy_queue, release_queue
from .test_routes_custom_orchestrators import _payload as _toy_payload
from .test_routes_custom_orchestrators import _register_toy
from .test_routes_resume import _classification_full
from .test_routes_resume import _run as _stopped_run

HEADER = "X-VF-Profile"
_TS = "20260522_100000_000000"


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """The app and routes module, running from an empty working directory."""
    import visionforge.gui.api.routes as routes_mod
    from visionforge.gui.server import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(routes_mod, "_MODELS_DIR", Path("outputs/models"))
    monkeypatch.setattr(routes_mod, "_PROFILES_DIR", Path("outputs/profiles"))
    routes_mod._current_run = None
    yield app, routes_mod
    release_queue(routes_mod)


@pytest.fixture
def client(env) -> TestClient:  # type: ignore[no-untyped-def]
    app, _ = env
    return TestClient(app, raise_server_exceptions=True, client=("127.0.0.1", 50000))


def _make_profile(client: TestClient, name: str) -> dict[str, Any]:
    resp = client.post("/api/profiles", json={"name": name})
    assert resp.status_code == 201, resp.text
    created: dict[str, Any] = resp.json()
    return created


def _write_run(models: Path, experiment: str = "exp", ts: str = _TS) -> Path:
    run_dir = models / experiment / ts
    run_dir.mkdir(parents=True)
    data = {
        "id": f"{experiment}_{ts}",
        "experiment": experiment,
        "timestamp": "2026-05-22T10:00:00",
        "status": "completed",
        "run_dir": str(run_dir.resolve()),
        "config": {},
        "metrics": {"total_epochs": 1},
        "history": [],
        "artifacts": {},
        "tests": [],
    }
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")
    return run_dir


def _link_dir(link: Path, target: Path) -> None:
    """Make ``link`` a directory link to ``target``, or skip the test.

    A symlink needs a privilege a Windows account often lacks; a junction
    (``mklink /J``) does not, and ``Path.resolve`` follows both the same way.
    """
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        made = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
        )
        if made.returncode == 0:
            return
    pytest.skip("this account cannot create directory links")


def _ids(resp: Any) -> set[str]:
    assert resp.status_code == 200, resp.text
    return {run["run_id"] for run in resp.json()}


# ── the folder logic ─────────────────────────────────────────────────────────


class TestSlug:
    @pytest.mark.parametrize(
        ("name", "slug"),
        [
            ("Ana", "ana"),
            ("João Silva", "joao-silva"),
            ("  Ana   Maria  ", "ana-maria"),
            ("ana_maria", "ana_maria"),
            ("Ünïcödé", "unicode"),
            ("---a---", "a"),
            ("a/b\\c", "a-b-c"),
            ("../x", "x"),
            ("Lab 2 (GPU)", "lab-2-gpu"),
            ("!!!", ""),
            ("李雷", ""),
            ("", ""),
        ],
    )
    def test_derived_from_the_display_name(self, name: str, slug: str) -> None:
        assert slugify(name) == slug

    def test_is_capped_without_a_dangling_dash(self) -> None:
        slug = slugify("a" * 39 + " b")

        assert slug == "a" * 39
        assert len(slugify("x" * 80)) == 40

    @pytest.mark.parametrize("slug", ["ana", "a", "ana-2", "ana_2", "0", "x" * 40])
    def test_accepts_the_allowed_alphabet(self, slug: str) -> None:
        assert is_valid_slug(slug)

    @pytest.mark.parametrize(
        "slug",
        [
            "",
            "Ana",
            "../x",
            "..",
            ".",
            "a/b",
            "a\\b",
            "a.b",
            "a b",
            "-a",
            "_a",
            "x" * 41,
            "joão",
            "default",
            "con",
            "nul",
            "com1",
            "lpt9",
        ],
    )
    def test_refuses_everything_else(self, slug: str) -> None:
        assert not is_valid_slug(slug)

    def test_a_derived_slug_is_always_valid_or_empty(self) -> None:
        for name in ("João", "A b c", "x" * 99, "-_-a", "Ünï", "ok_1-2"):
            slug = slugify(name)
            assert slug == "" or is_valid_slug(slug)

    def test_the_display_name_is_tidied(self) -> None:
        assert normalize_display_name("  Ana \t\n Maria\x00 ") == "Ana Maria"
        assert len(normalize_display_name("n" * 200)) == 60


class TestFolders:
    def test_creating_makes_the_folder_and_its_output_subdirs(
        self, tmp_path: Path
    ) -> None:
        profiles = tmp_path / "profiles"

        profile = create_profile("João Silva", profiles, tmp_path / "models")

        folder = profiles / "joao-silva"
        for sub in ("models", "graphics", "logs", "reports"):
            assert (folder / sub).is_dir()
        meta = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
        assert meta["name"] == "João Silva"
        assert meta["slug"] == "joao-silva"
        assert (profile.slug, profile.name) == ("joao-silva", "João Silva")
        assert profile.models_dir == profiles / "joao-silva" / "models"

    def test_the_same_slug_twice_is_refused_and_names_the_first(
        self, tmp_path: Path
    ) -> None:
        create_profile("João", tmp_path / "p", tmp_path / "m")

        with pytest.raises(ProfileExistsError, match="João"):
            create_profile("Joao", tmp_path / "p", tmp_path / "m")

    @pytest.mark.parametrize(
        "name",
        # "COM¹" and "LPT³" fold to com1 and lpt3 (NFKD), so they are caught too.
        ["!!!", "李雷", "default", "Default", "NUL", "COM0", "lpt0", "COM¹", "LPT³"],
    )
    def test_a_name_with_no_usable_slug_is_refused(
        self, tmp_path: Path, name: str
    ) -> None:
        with pytest.raises(ProfileNameError):
            create_profile(name, tmp_path / "p", tmp_path / "m")

        assert not (tmp_path / "p").exists() or not any((tmp_path / "p").iterdir())

    def test_listing_puts_the_default_first_then_names_in_order(
        self, tmp_path: Path
    ) -> None:
        for name in ("zeca", "Bia", "ana"):
            create_profile(name, tmp_path / "p", tmp_path / "m")

        listed = list_profiles(tmp_path / "p", tmp_path / "m")

        assert [p.slug for p in listed] == ["default", "ana", "bia", "zeca"]
        assert listed[0].is_default
        assert not any(p.is_default for p in listed[1:])

    def test_a_hand_made_folder_is_a_profile_named_by_its_slug(
        self, tmp_path: Path
    ) -> None:
        profiles = tmp_path / "p"
        (profiles / "lab-3").mkdir(parents=True)
        (profiles / "Not A Slug").mkdir()
        (profiles / "stray.txt").write_text("x", encoding="utf-8")

        listed = list_profiles(profiles, tmp_path / "m")

        assert [(p.slug, p.name) for p in listed] == [
            ("default", "default"),
            ("lab-3", "lab-3"),
        ]

    def test_a_broken_profile_json_falls_back_to_the_slug(self, tmp_path: Path) -> None:
        profiles = tmp_path / "p"
        (profiles / "ana").mkdir(parents=True)
        (profiles / "ana" / "profile.json").write_text("{nope", encoding="utf-8")

        assert resolve_profile("ana", profiles, tmp_path / "m").name == "ana"

    def test_no_profiles_folder_lists_only_the_default(self, tmp_path: Path) -> None:
        listed = list_profiles(tmp_path / "missing", tmp_path / "m")

        assert [p.slug for p in listed] == ["default"]

    def test_a_folder_the_os_refuses_is_a_clear_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_mkdir = Path.mkdir

        def deny(self: Path, *args: Any, **kwargs: Any) -> None:
            if self.name == "ana":
                raise PermissionError(13, "Acesso negado")
            real_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", deny)

        with pytest.raises(ProfileCreateError, match="Não foi possível criar"):
            create_profile("Ana", tmp_path / "p", tmp_path / "m")

    def test_a_profiles_root_that_is_a_file_is_not_reported_as_taken(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "p").write_text("not a folder", encoding="utf-8")

        with pytest.raises(ProfileCreateError):
            create_profile("Ana", tmp_path / "p", tmp_path / "m")

    def test_a_full_disk_leaves_no_half_made_profile_to_trip_over(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        profiles = tmp_path / "p"

        with monkeypatch.context() as patched:

            def full(self: Path, *args: Any, **kwargs: Any) -> int:
                raise OSError(28, "No space left on device")

            patched.setattr(Path, "write_text", full)
            with pytest.raises(ProfileCreateError, match="No space left"):
                create_profile("Ana", profiles, tmp_path / "m")

        assert not (profiles / "ana").exists()
        # The same name works once the disk has room; it was not left "taken".
        assert create_profile("Ana", profiles, tmp_path / "m").slug == "ana"


class TestResolve:
    def test_no_header_and_default_are_the_legacy_layout(self, tmp_path: Path) -> None:
        legacy = tmp_path / "outputs" / "models"

        for header in (None, "", "  ", "default"):
            profile = resolve_profile(header, tmp_path / "p", legacy)
            assert profile.is_default
            assert profile.models_dir == legacy
            assert profile.output_paths() is None

    def test_a_named_profile_lives_under_the_profiles_folder(
        self, tmp_path: Path
    ) -> None:
        create_profile("Ana", tmp_path / "p", tmp_path / "m")

        profile = resolve_profile("ana", tmp_path / "p", tmp_path / "m")

        assert not profile.is_default
        assert profile.models_dir == tmp_path / "p" / "ana" / "models"
        assert profile.name == "Ana"

    @pytest.mark.parametrize(
        "header",
        ["../x", "..", "a/b", "a\\b", "Ana", "x" * 41, "con", "-a", "a.b", "%2e%2e"],
    )
    def test_an_unsafe_slug_is_invalid(self, tmp_path: Path, header: str) -> None:
        with pytest.raises(InvalidProfileError):
            resolve_profile(header, tmp_path / "p", tmp_path / "m")

    def test_a_safe_slug_without_a_folder_is_unknown(self, tmp_path: Path) -> None:
        (tmp_path / "p").mkdir()

        with pytest.raises(UnknownProfileError):
            resolve_profile("ghost", tmp_path / "p", tmp_path / "m")

    def test_a_link_out_of_the_profiles_folder_is_refused(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (tmp_path / "p").mkdir()
        _link_dir(tmp_path / "p" / "evil", outside)

        with pytest.raises(InvalidProfileError):
            resolve_profile("evil", tmp_path / "p", tmp_path / "m")
        assert [p.slug for p in list_profiles(tmp_path / "p", tmp_path / "m")] == [
            "default"
        ]

    def test_a_models_folder_linked_elsewhere_is_refused(self, tmp_path: Path) -> None:
        create_profile("Ana", tmp_path / "p", tmp_path / "m")
        models = tmp_path / "p" / "ana" / "models"
        models.rmdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        _link_dir(models, outside)

        with pytest.raises(InvalidProfileError):
            resolve_profile("ana", tmp_path / "p", tmp_path / "m")

    @pytest.mark.parametrize("sub", OUTPUT_SUBDIRS)
    def test_any_output_folder_linked_elsewhere_is_refused(
        self, tmp_path: Path, sub: str
    ) -> None:
        create_profile("Ana", tmp_path / "p", tmp_path / "m")
        linked = tmp_path / "p" / "ana" / sub
        linked.rmdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        _link_dir(linked, outside)

        with pytest.raises(InvalidProfileError, match=sub):
            resolve_profile("ana", tmp_path / "p", tmp_path / "m")
        assert [p.slug for p in list_profiles(tmp_path / "p", tmp_path / "m")] == [
            "default"
        ]


class _Cfg(BaseModel):
    name: str = "x"
    output: OutputConfig = OutputConfig()


class TestScoping:
    def test_the_default_profile_leaves_a_config_alone(self, tmp_path: Path) -> None:
        profile = default_profile(tmp_path / "m")
        raw = {"name": "x", "output": {"models_dir": "elsewhere"}}
        model = _Cfg()

        assert profile.scope_dict(raw) is raw
        assert profile.scope_model(model) is model

    def test_a_named_profile_forces_all_four_output_folders(
        self, tmp_path: Path
    ) -> None:
        create_profile("Ana", tmp_path / "p", tmp_path / "m")
        profile = resolve_profile("ana", tmp_path / "p", tmp_path / "m")
        raw: dict[str, Any] = {
            "name": "x",
            "output": {"models_dir": "elsewhere", "extra": 1},
        }

        scoped = profile.scope_dict(raw)

        root = (tmp_path / "p" / "ana").as_posix()
        assert scoped["output"] == {
            "extra": 1,
            "models_dir": f"{root}/models",
            "graphics_dir": f"{root}/graphics",
            "logs_dir": f"{root}/logs",
            "reports_dir": f"{root}/reports",
        }
        assert raw["output"]["models_dir"] == "elsewhere"  # the input is not mutated

    def test_a_config_without_output_gets_one(self, tmp_path: Path) -> None:
        create_profile("Ana", tmp_path / "p", tmp_path / "m")
        profile = resolve_profile("ana", tmp_path / "p", tmp_path / "m")

        assert set(profile.scope_dict({"name": "x"})["output"]) == {
            "models_dir",
            "graphics_dir",
            "logs_dir",
            "reports_dir",
        }

    def test_a_validated_model_is_copied_not_mutated(self, tmp_path: Path) -> None:
        create_profile("Ana", tmp_path / "p", tmp_path / "m")
        profile = resolve_profile("ana", tmp_path / "p", tmp_path / "m")
        original = _Cfg()

        scoped = profile.scope_model(original)

        assert scoped.output.models_dir == tmp_path / "p" / "ana" / "models"
        assert scoped.output.reports_dir == tmp_path / "p" / "ana" / "reports"
        assert original.output.models_dir == Path("outputs/models")


# ── the HTTP contract ────────────────────────────────────────────────────────


class TestProfilesApi:
    def test_a_fresh_install_lists_only_the_default(self, client: TestClient) -> None:
        body = client.get("/api/profiles").json()

        assert body["profiles"] == [
            {"slug": "default", "name": "default", "is_default": True}
        ]

    def test_creating_then_listing(self, client: TestClient, tmp_path: Path) -> None:
        created = _make_profile(client, "João Silva")

        assert created == {
            "slug": "joao-silva",
            "name": "João Silva",
            "is_default": False,
        }
        assert (tmp_path / "outputs" / "profiles" / "joao-silva" / "models").is_dir()
        slugs = [p["slug"] for p in client.get("/api/profiles").json()["profiles"]]
        assert slugs == ["default", "joao-silva"]

    def test_a_taken_name_is_a_409(self, client: TestClient) -> None:
        _make_profile(client, "Ana")

        resp = client.post("/api/profiles", json={"name": "ana"})

        assert resp.status_code == 409
        assert "Ana" in resp.json()["detail"]

    @pytest.mark.parametrize("name", ["!!!", "default", "con"])
    def test_a_name_that_cannot_be_a_folder_is_a_422(
        self, client: TestClient, name: str
    ) -> None:
        assert client.post("/api/profiles", json={"name": name}).status_code == 422

    def test_a_folder_the_os_refuses_is_a_500_with_a_message(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        (tmp_path / "outputs").mkdir()
        (tmp_path / "outputs" / "profiles").write_text("x", encoding="utf-8")

        resp = client.post("/api/profiles", json={"name": "Ana"})

        assert resp.status_code == 500
        assert "Não foi possível criar" in resp.json()["detail"]

    @pytest.mark.parametrize("body", [{}, {"name": ""}, {"name": "n" * 61}])
    def test_a_missing_or_overlong_name_is_a_422(
        self, client: TestClient, body: dict[str, Any]
    ) -> None:
        assert client.post("/api/profiles", json=body).status_code == 422


class TestHeaderValidation:
    @pytest.mark.parametrize(
        "slug", ["../x", "..", "Ana", "a/b", "a\\b", "x" * 41, "con", "-a", "a.b"]
    )
    def test_an_unsafe_slug_is_a_400(self, client: TestClient, slug: str) -> None:
        resp = client.get("/api/runs", headers={HEADER: slug})

        assert resp.status_code == 400

    def test_an_unknown_profile_is_a_404(self, client: TestClient) -> None:
        resp = client.get("/api/runs", headers={HEADER: "ghost"})

        assert resp.status_code == 404
        assert "ghost" in resp.json()["detail"]

    def test_no_header_and_default_are_the_default_profile(
        self, client: TestClient
    ) -> None:
        assert client.get("/api/runs").status_code == 200
        assert client.get("/api/runs", headers={HEADER: "default"}).status_code == 200

    def test_a_refused_header_does_not_create_anything(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        client.get("/api/runs", headers={HEADER: "../x"})
        client.get("/api/runs", headers={HEADER: "ghost"})

        assert not (tmp_path / "outputs" / "profiles").exists()
        assert not (tmp_path / "x").exists()

    def test_routes_that_ignore_profiles_ignore_a_stale_header(
        self, client: TestClient
    ) -> None:
        """A profile folder deleted by hand must not brick the whole page."""
        resp = client.get("/api/system/info", headers={HEADER: "ghost"})

        assert resp.status_code == 200
        assert client.get("/api/profiles", headers={HEADER: "ghost"}).status_code == 200


# ── History is per profile ───────────────────────────────────────────────────


class TestHistoryIsPerProfile:
    def test_the_default_profile_reads_the_legacy_models_folder(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        legacy = _write_run(tmp_path / "outputs" / "models")

        for headers in ({}, {HEADER: "default"}):
            resp = client.get("/api/runs", headers=headers)
            assert [r["run_id"] for r in resp.json()] == [legacy.name]
        assert client.get(f"/api/runs/{legacy.name}").status_code == 200

    def test_a_profile_lists_only_its_own_runs(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        _make_profile(client, "Ana")
        _make_profile(client, "Bob")
        outputs = tmp_path / "outputs"
        legacy = _write_run(outputs / "models", ts="20260522_100000_000001")
        ana = _write_run(
            outputs / "profiles" / "ana" / "models", ts="20260522_100000_000002"
        )
        bob = _write_run(
            outputs / "profiles" / "bob" / "models", ts="20260522_100000_000003"
        )

        assert _ids(client.get("/api/runs")) == {legacy.name}
        assert _ids(client.get("/api/runs", headers={HEADER: "ana"})) == {ana.name}
        assert _ids(client.get("/api/runs", headers={HEADER: "bob"})) == {bob.name}

    def test_the_detail_of_another_profiles_run_is_a_404(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        _make_profile(client, "Ana")
        _make_profile(client, "Bob")
        ana = _write_run(tmp_path / "outputs" / "profiles" / "ana" / "models")

        assert (
            client.get(f"/api/runs/{ana.name}", headers={HEADER: "ana"}).status_code
            == 200
        )
        assert (
            client.get(f"/api/runs/{ana.name}", headers={HEADER: "bob"}).status_code
            == 404
        )
        assert client.get(f"/api/runs/{ana.name}").status_code == 404

    def test_a_run_is_found_by_its_recorded_id_inside_the_profile_only(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        _make_profile(client, "Ana")
        _write_run(tmp_path / "outputs" / "profiles" / "ana" / "models")

        by_id = f"exp_{_TS}"

        assert (
            client.get(f"/api/runs/{by_id}", headers={HEADER: "ana"}).status_code == 200
        )
        assert client.get(f"/api/runs/{by_id}").status_code == 404

    def test_delete_stays_inside_the_profile(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        _make_profile(client, "Ana")
        _make_profile(client, "Bob")
        outputs = tmp_path / "outputs"
        ana = _write_run(outputs / "profiles" / "ana" / "models")
        legacy = _write_run(
            outputs / "models", experiment="legacy", ts="20260522_100000_000009"
        )

        # Another profile and the default one cannot delete ana's run.
        assert (
            client.delete(f"/api/runs/{ana.name}", headers={HEADER: "bob"}).status_code
            == 404
        )
        assert client.delete(f"/api/runs/{ana.name}").status_code == 404
        assert ana.is_dir()
        # And ana cannot reach the legacy run, even by its exact id.
        assert (
            client.delete(
                f"/api/runs/{legacy.name}", headers={HEADER: "ana"}
            ).status_code
            == 404
        )
        assert legacy.is_dir()

        resp = client.delete(f"/api/runs/{ana.name}", headers={HEADER: "ana"})

        assert resp.status_code == 200
        assert not ana.exists()
        assert legacy.is_dir()

    def test_reveal_opens_the_profile_folder_and_only_for_its_owner(
        self,
        client: TestClient,
        env,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,  # type: ignore[no-untyped-def]
    ) -> None:
        _, routes_mod = env
        _make_profile(client, "Ana")
        _make_profile(client, "Bob")
        ana = _write_run(tmp_path / "outputs" / "profiles" / "ana" / "models")
        opened: list[Path] = []
        monkeypatch.setattr(routes_mod, "open_folder", opened.append)

        refused = client.post(f"/api/runs/{ana.name}/reveal", headers={HEADER: "bob"})
        assert refused.status_code == 404
        assert opened == []

        resp = client.post(f"/api/runs/{ana.name}/reveal", headers={HEADER: "ana"})

        assert resp.status_code == 200
        assert [p.resolve() for p in opened] == [ana.resolve()]

    def test_resume_is_found_in_its_own_profile_and_scoped_to_it(
        self,
        client: TestClient,
        env,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,  # type: ignore[no-untyped-def]
    ) -> None:
        _, routes_mod = env
        _make_profile(client, "Ana")
        _make_profile(client, "Bob")
        stored = _classification_full(tmp_path)
        run_dir, _ = _stopped_run(
            tmp_path / "outputs" / "profiles" / "ana" / "models" / "e",
            stored,
            resume_file=True,
        )
        submitted: dict[str, Any] = {}

        def fake_submit(job_id, label, task, strategy, start, **kwargs):  # type: ignore[no-untyped-def]
            submitted.update(start=start, **kwargs)
            return RunResponse(run_id=job_id)

        seen: dict[str, Any] = {}

        async def fake_execute(config, job_id, *, resume_dir):  # type: ignore[no-untyped-def]
            seen.update(config=config, resume_dir=resume_dir)

        monkeypatch.setattr(routes_mod, "_submit_job", fake_submit)
        monkeypatch.setattr(routes_mod, "_execute_experiment", fake_execute)

        assert (
            client.post(
                f"/api/runs/{run_dir.name}/resume", headers={HEADER: "bob"}
            ).status_code
            == 404
        )
        assert client.post(f"/api/runs/{run_dir.name}/resume").status_code == 404
        assert not submitted

        resp = client.post(f"/api/runs/{run_dir.name}/resume", headers={HEADER: "ana"})

        assert resp.status_code == 200, resp.text
        assert submitted["profile"].slug == "ana"
        asyncio.run(submitted["start"]())
        assert seen["resume_dir"].resolve() == run_dir.resolve()
        out = seen["config"].output
        assert out.models_dir == Path("outputs/profiles/ana/models")
        assert out.reports_dir == Path("outputs/profiles/ana/reports")

    @pytest.mark.parametrize(
        ("method", "suffix", "body"),
        [
            ("get", "/export_md", None),
            ("post", "/test", {"data_dir": "x"}),
            ("post", "/gradcam", {"input_dir": "x"}),
            ("post", "/batch_predict", {"input_dir": "x"}),
            ("post", "/export_onnx", {}),
            ("post", "/resume", None),
            ("post", "/reveal", None),
            ("delete", "", None),
            ("get", "", None),
        ],
    )
    def test_every_run_action_is_a_404_for_another_profile(
        self,
        client: TestClient,
        tmp_path: Path,
        method: str,
        suffix: str,
        body: dict[str, Any] | None,
    ) -> None:
        _make_profile(client, "Ana")
        _make_profile(client, "Bob")
        ana = _write_run(tmp_path / "outputs" / "profiles" / "ana" / "models")
        url = f"/api/runs/{ana.name}{suffix}"
        kwargs: dict[str, Any] = {"headers": {HEADER: "bob"}}
        if body is not None:
            kwargs["json"] = body

        resp = getattr(client, method)(url, **kwargs)

        assert resp.status_code == 404, resp.text
        assert "not found" in resp.json()["detail"]
        assert (ana / "run.json").is_file()

    @pytest.mark.parametrize(
        ("method", "suffix", "body"),
        [
            ("get", "/export_md", None),
            ("post", "/test", {"data_dir": "x"}),
            ("post", "/gradcam", {"input_dir": "x"}),
            ("post", "/batch_predict", {"input_dir": "x"}),
            ("post", "/export_onnx", {}),
        ],
    )
    def test_the_owner_gets_past_the_lookup(
        self,
        client: TestClient,
        tmp_path: Path,
        method: str,
        suffix: str,
        body: dict[str, Any] | None,
    ) -> None:
        """The same request as above is not a 404 for the profile that owns the run."""
        _make_profile(client, "Ana")
        ana = _write_run(tmp_path / "outputs" / "profiles" / "ana" / "models")
        kwargs: dict[str, Any] = {"headers": {HEADER: "ana"}}
        if body is not None:
            kwargs["json"] = body

        resp = getattr(client, method)(f"/api/runs/{ana.name}{suffix}", **kwargs)

        assert resp.status_code != 404, resp.text

    def test_a_run_whose_folder_is_a_link_out_of_the_root_is_not_found(
        self,
        client: TestClient,
        env,
        tmp_path: Path,  # type: ignore[no-untyped-def]
    ) -> None:
        _, routes_mod = env
        _make_profile(client, "Ana")
        secret = _write_run(tmp_path / "secret" / "models", experiment="loot")
        models = tmp_path / "outputs" / "profiles" / "ana" / "models"
        _link_dir(models / "loot", secret.parent)
        profile = routes_mod.resolve_profile(
            "ana", routes_mod._PROFILES_DIR, routes_mod._MODELS_DIR
        )

        assert routes_mod._find_run_dir(secret.name, profile) is None
        assert routes_mod._find_run_dir("..", profile) is None
        assert secret.is_dir()

    def test_the_default_profile_follows_a_linked_run_folder(
        self,
        client: TestClient,
        env,
        tmp_path: Path,  # type: ignore[no-untyped-def]
    ) -> None:
        """Old runs moved to another drive and linked back stay reachable.

        The History lists them, so every action on them has to find them too:
        the link-escape check belongs to the named profiles only (ADR-114).
        """
        _, routes_mod = env
        moved = _write_run(tmp_path / "other-drive" / "models", experiment="old")
        models = tmp_path / "outputs" / "models"
        models.mkdir(parents=True)
        _link_dir(models / "old", moved.parent)
        if moved.name not in _ids(client.get("/api/runs")):
            pytest.skip("this Python's glob does not enter that kind of link")

        for headers in ({}, {HEADER: "default"}):
            assert (
                client.get(f"/api/runs/{moved.name}", headers=headers).status_code
                == 200
            )
            assert (
                client.get(
                    f"/api/runs/{moved.name}/export_md", headers=headers
                ).status_code
                != 404
            )
        assert routes_mod._find_run_dir(moved.name) is not None

    def test_a_named_profile_still_refuses_a_linked_run_folder_over_http(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        _make_profile(client, "Ana")
        secret = _write_run(tmp_path / "secret" / "models", experiment="loot")
        models = tmp_path / "outputs" / "profiles" / "ana" / "models"
        _link_dir(models / "loot", secret.parent)

        resp = client.get(f"/api/runs/{secret.name}", headers={HEADER: "ana"})

        assert resp.status_code == 404
        assert secret.is_dir()


# ── submissions land in the profile ──────────────────────────────────────────


def _finish(client: TestClient) -> dict[str, Any]:
    status: dict[str, Any] = {"status": "running"}
    for _ in range(600):
        status = client.get("/api/experiment/status").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    return status


class TestARealRunLandsInItsProfile:
    """The researcher's whole path, with nothing mocked: submit, train, list."""

    @pytest.fixture
    def live(self, env, tmp_path: Path):  # type: ignore[no-untyped-def]
        from visionforge.tasks import clear_task_registry

        app, routes_mod = env
        clear_task_registry()
        _register_toy()
        try:
            # One event loop for the whole test: the job is a background task of
            # the request that queued it.
            with TestClient(app, raise_server_exceptions=True) as client:
                _make_profile(client, "Ana")
                _make_profile(client, "Bob")
                yield client
        finally:
            routes_mod._RUN_QUEUE.reset()
            routes_mod._current_run = None
            routes_mod._active_cancel_token = None
            clear_task_registry()

    @staticmethod
    def _payload(tmp_path: Path) -> dict[str, Any]:
        cfg = _toy_payload(tmp_path)
        cfg.pop("output")  # the profile decides where this goes
        return cfg

    def test_a_run_submitted_with_a_profile_writes_under_it(
        self, live: TestClient, tmp_path: Path
    ) -> None:
        resp = live.post(
            "/api/custom/toyorch/run",
            json=self._payload(tmp_path),
            headers={HEADER: "ana"},
        )
        assert resp.status_code == 200, resp.text
        assert _finish(live)["status"] == "completed"

        ana_models = tmp_path / "outputs" / "profiles" / "ana" / "models"
        assert list(ana_models.rglob("run.json"))
        assert not (tmp_path / "outputs" / "models").exists()
        # History: the owner sees it, another profile and the default do not.
        assert len(live.get("/api/runs", headers={HEADER: "ana"}).json()) == 1
        assert live.get("/api/runs", headers={HEADER: "bob"}).json() == []
        assert live.get("/api/runs").json() == []

    def test_a_run_without_a_profile_still_writes_the_legacy_folder(
        self, live: TestClient, tmp_path: Path
    ) -> None:
        resp = live.post("/api/custom/toyorch/run", json=self._payload(tmp_path))
        assert resp.status_code == 200, resp.text
        assert _finish(live)["status"] == "completed"

        assert list((tmp_path / "outputs" / "models").rglob("run.json"))
        assert not list((tmp_path / "outputs" / "profiles").rglob("run.json"))
        assert len(live.get("/api/runs").json()) == 1
        assert live.get("/api/runs", headers={HEADER: "ana"}).json() == []

    def test_replicates_and_their_group_land_in_the_profile(
        self, live: TestClient, tmp_path: Path
    ) -> None:
        resp = live.post(
            "/api/custom/toyorch/replicates",
            json={"config": self._payload(tmp_path), "seeds": [7, 8]},
            headers={HEADER: "bob"},
        )
        assert resp.status_code == 200, resp.text
        assert _finish(live)["status"] == "completed"

        bob = tmp_path / "outputs" / "profiles" / "bob"
        assert list((bob / "reports").rglob("replicates_summary.json"))
        runs = live.get("/api/runs", headers={HEADER: "bob"}).json()
        assert len(runs) == 3  # two seeds and the group that ties them
        assert live.get("/api/runs", headers={HEADER: "ana"}).json() == []
        assert live.get("/api/runs").json() == []
        assert not (tmp_path / "outputs" / "reports").exists()


class TestQueueShowsTheProfile:
    def test_a_waiting_job_carries_its_profiles_display_name(
        self,
        client: TestClient,
        env,
        tmp_path: Path,  # type: ignore[no-untyped-def]
    ) -> None:
        _, routes_mod = env
        _make_profile(client, "João Silva")
        base = tmp_path / "ds"
        base.mkdir()
        occupy_queue(routes_mod)
        body = {
            "name": "reg",
            "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
            "data": {"base_dir": str(base), "target_columns": ["target"]},
            "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.001},
        }

        owned = client.post(
            "/api/regression/run", json=body, headers={HEADER: "joao-silva"}
        )
        anonymous = client.post("/api/regression/run", json=body)

        assert owned.json()["status"] == "queued"
        assert anonymous.json()["status"] == "queued"
        pending = client.get("/api/queue").json()["pending"]
        assert [(j["profile"], j["profile_name"]) for j in pending] == [
            ("joao-silva", "João Silva"),
            ("default", "default"),
        ]


class TestEverySubmissionIsScoped:
    """Each starter hands its executor output folders inside the profile."""

    @staticmethod
    def _cfg(kind: str, tmp_path: Path) -> dict[str, Any]:
        base = tmp_path / "ds"
        for split in ("train", "val"):
            (base / "images" / split).mkdir(parents=True, exist_ok=True)
        training = {"epochs": 1, "batch_size": 4, "learning_rate": 0.001}
        configs: dict[str, dict[str, Any]] = {
            "classification": _classification_full(base),
            "detection": {
                "name": "det",
                "model": {
                    "backend": "ultralytics",
                    "name": "yolo11n",
                    "num_classes": 2,
                },
                "data": {"base_dir": str(base), "image_size": 640},
                "training": training,
            },
            "regression": {
                "name": "reg",
                "model": {"name": "resnet18", "num_targets": 1, "pretrained": False},
                "data": {"base_dir": str(base), "target_columns": ["target"]},
                "training": training,
            },
            "segmentation": {
                "name": "seg",
                "model": {"name": "unet", "num_classes": 3, "pretrained": False},
                "data": {"base_dir": str(base)},
                "training": training,
            },
            "anomaly": {
                "name": "ano",
                "model": {"name": "autoencoder", "latent_dim": 16},
                "data": {"base_dir": str(base)},
                "training": training,
            },
            "custom": {
                "name": "toy",
                "data": {"base_dir": str(base)},
                "training": {"epochs": 1, "batch_size": 4},
                "device": {"kind": "cpu"},
            },
        }
        return configs[kind]

    @staticmethod
    def _body(shape: str, cfg: dict[str, Any]) -> dict[str, Any]:
        extra: dict[str, dict[str, Any]] = {
            "run": {},
            "compare": {"model_names": ["resnet18", "resnet34"]},
            "sweep": {"search_space": {"training.learning_rate": [0.001, 0.01]}},
            "cv": {"n_folds": 2},
            "replicates": {"seeds": [1, 2]},
            "replicated-comparison": {
                "variants": {
                    "a": {"training.epochs": 1},
                    "b": {"training.epochs": 2},
                },
                "seeds": [1, 2],
            },
        }
        return cfg if shape == "run" else {"config": cfg, **extra[shape]}

    CASES = [
        ("/api/experiment/run", "classification", "run"),
        ("/api/detection/run", "detection", "run"),
        ("/api/regression/run", "regression", "run"),
        ("/api/segmentation/run", "segmentation", "run"),
        ("/api/anomaly/run", "anomaly", "run"),
        ("/api/custom/toyorch/run", "custom", "run"),
        *[
            (f"/api/{task}/{route}", task, route)
            for task in ("detection", "regression", "segmentation", "anomaly")
            for route in ("compare", "sweep", "replicates", "replicated-comparison")
        ],
        ("/api/regression/cv", "regression", "cv"),
        ("/api/segmentation/cv", "segmentation", "cv"),
        ("/api/classification/replicates", "classification", "replicates"),
        (
            "/api/classification/replicated-comparison",
            "classification",
            "replicated-comparison",
        ),
        ("/api/custom/toyorch/sweep", "custom", "sweep"),
        ("/api/custom/toyorch/replicates", "custom", "replicates"),
        (
            "/api/custom/toyorch/replicated-comparison",
            "custom",
            "replicated-comparison",
        ),
    ]

    @pytest.fixture
    def spy(self, client: TestClient, env, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
        """Replace the queue and every executor; record the output folders seen."""
        from visionforge.tasks import clear_task_registry

        _, routes_mod = env
        clear_task_registry()
        _register_toy()
        _make_profile(client, "Ana")
        jobs: list[tuple[Any, Any]] = []
        outputs: list[dict[str, str]] = []

        def paths(output: Any) -> dict[str, str]:
            raw = output if isinstance(output, dict) else output.model_dump()
            return {k: Path(str(v)).as_posix() for k, v in raw.items()}

        def fake_submit(
            run_id, label, task, strategy, start, stop_at="auto", profile=None
        ):  # type: ignore[no-untyped-def]
            jobs.append((profile, start))
            return RunResponse(run_id=run_id)

        def single(position: int):  # type: ignore[no-untyped-def]
            async def fake(*args: Any, **kwargs: Any) -> None:
                holder = args[position]
                outputs.append(
                    paths(
                        holder["output"] if isinstance(holder, dict) else holder.output
                    )
                )

            return fake

        async def fake_cv(config, req, run_id, cv_fn, task_label):  # type: ignore[no-untyped-def]
            outputs.append(paths(config.output))
            outputs.append(paths(req.config["output"]))

        monkeypatch.setattr(routes_mod, "_submit_job", fake_submit)
        for name in (
            "_execute_experiment",
            "_execute_detection",
            "_execute_regression",
            "_execute_segmentation",
            "_execute_anomaly",
        ):
            monkeypatch.setattr(routes_mod, name, single(0))
        monkeypatch.setattr(routes_mod, "_execute_custom_task", single(1))
        monkeypatch.setattr(routes_mod, "_execute_comparison", single(1))
        monkeypatch.setattr(routes_mod, "_execute_sweep", single(1))
        monkeypatch.setattr(routes_mod, "_execute_replicated_comparison", single(1))
        monkeypatch.setattr(routes_mod, "_execute_replicates", single(1))
        monkeypatch.setattr(routes_mod, "_execute_task_cv", fake_cv)
        yield jobs, outputs
        clear_task_registry()

    @pytest.mark.parametrize(("path", "kind", "shape"), CASES)
    def test_a_named_profile_owns_every_output_folder(
        self,
        client: TestClient,
        spy: tuple[list[Any], list[dict[str, str]]],
        tmp_path: Path,
        path: str,
        kind: str,
        shape: str,
    ) -> None:
        jobs, outputs = spy
        body = self._body(shape, self._cfg(kind, tmp_path))

        resp = client.post(path, json=body, headers={HEADER: "ana"})

        assert resp.status_code == 200, resp.text
        assert len(jobs) == 1
        profile, start = jobs[0]
        assert profile.slug == "ana"
        asyncio.run(start())
        assert outputs
        for out in outputs:
            assert out == {
                "models_dir": "outputs/profiles/ana/models",
                "graphics_dir": "outputs/profiles/ana/graphics",
                "logs_dir": "outputs/profiles/ana/logs",
                "reports_dir": "outputs/profiles/ana/reports",
            }

    @pytest.mark.parametrize(("path", "kind", "shape"), CASES)
    def test_the_default_profile_does_not_touch_the_config(
        self,
        client: TestClient,
        spy: tuple[list[Any], list[dict[str, str]]],
        tmp_path: Path,
        path: str,
        kind: str,
        shape: str,
    ) -> None:
        jobs, outputs = spy
        cfg = self._cfg(kind, tmp_path)
        cfg["output"] = {"models_dir": "my/own/models"}

        resp = client.post(path, json=self._body(shape, cfg))

        assert resp.status_code == 200, resp.text
        profile, start = jobs[0]
        assert profile.is_default
        asyncio.run(start())
        assert outputs
        assert all(out["models_dir"] == "my/own/models" for out in outputs)

    def test_a_browser_supplied_output_cannot_leave_the_profile(
        self,
        client: TestClient,
        spy: tuple[list[Any], list[dict[str, str]]],
        tmp_path: Path,
    ) -> None:
        jobs, outputs = spy
        cfg = self._cfg("regression", tmp_path)
        cfg["output"] = {"models_dir": "../../../elsewhere", "reports_dir": "/abs/path"}

        client.post("/api/regression/run", json=cfg, headers={HEADER: "ana"})
        asyncio.run(jobs[0][1]())

        assert outputs[0]["models_dir"] == "outputs/profiles/ana/models"
        assert outputs[0]["reports_dir"] == "outputs/profiles/ana/reports"

    @pytest.mark.parametrize(
        ("path", "kind", "shape"),
        [c for c in CASES if c[2] in ("sweep", "replicated-comparison")],
    )
    @pytest.mark.parametrize("profile", [None, "ana"])
    @pytest.mark.parametrize(
        "target", ["output.models_dir", "output.reports_dir", "output"]
    )
    def test_a_sweep_or_comparison_cannot_vary_the_output_folder(
        self,
        client: TestClient,
        spy: tuple[list[Any], list[dict[str, str]]],
        tmp_path: Path,
        path: str,
        kind: str,
        shape: str,
        profile: str | None,
        target: str,
    ) -> None:
        """Applied per trial after the scoping, it would undo the profile."""
        jobs, _ = spy
        body = self._body(shape, self._cfg(kind, tmp_path))
        if shape == "sweep":
            body["search_space"] = {
                "training.learning_rate": [0.001, 0.01],
                target: ["elsewhere", "other"],
            }
        else:
            body["variants"] = {"a": {}, "b": {target: "elsewhere"}}
        headers = {HEADER: profile} if profile else {}

        resp = client.post(path, json=body, headers=headers)

        assert resp.status_code == 422, resp.text
        assert "pasta de saída" in resp.json()["detail"]
        assert target in resp.json()["detail"]
        assert not jobs

    def test_a_random_search_space_cannot_vary_the_output_folder_either(
        self,
        client: TestClient,
        spy: tuple[list[Any], list[dict[str, str]]],
        tmp_path: Path,
    ) -> None:
        jobs, _ = spy
        body = self._body("sweep", self._cfg("regression", tmp_path))
        body["mode"] = "random"
        body["search_space"] = {
            "output.models_dir": {"type": "choice", "options": ["a", "b"]}
        }

        resp = client.post("/api/regression/sweep", json=body, headers={HEADER: "ana"})

        assert resp.status_code == 422, resp.text
        assert not jobs
