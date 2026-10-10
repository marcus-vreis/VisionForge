"""The server words its messages in the language of the page (ADR-116).

The browser sends ``X-VF-Lang: pt|en`` on every call; a request without it, or
with something unreadable, is Portuguese, which is what these messages always
were. One test per family of migrated messages asserts the English text with the
English header and the Portuguese text without it, through the real app, so the
router-level dependency that binds the language is what is being tested rather
than a helper called by hand.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from visionforge.core.training_health import collapsed_predictions
from visionforge.gui.api.schemas import RunResponse
from visionforge.utils.messages import current_lang, set_lang

from .conftest import occupy_queue, release_queue
from .test_routes_resume import _classification_full
from .test_routes_resume import _run as _stopped_run

EN = {"X-VF-Lang": "en"}
PROFILE = "X-VF-Profile"


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


class TestTheHeaderPicksTheLanguage:
    URL = "/api/runs/nope"

    def test_english_and_portuguese(self, client: TestClient) -> None:
        en = client.get(self.URL, headers=EN)
        pt = client.get(self.URL, headers={"X-VF-Lang": "pt"})

        assert en.status_code == pt.status_code == 404
        assert en.json()["detail"] == "Run 'nope' not found."
        assert pt.json()["detail"] == "Run 'nope' não encontrado."

    def test_no_header_is_portuguese(self, client: TestClient) -> None:
        assert client.get(self.URL).json()["detail"] == "Run 'nope' não encontrado."

    @pytest.mark.parametrize("garbage", ["klingon", "", "e", "en;q=0.9", "english"])
    def test_an_unreadable_header_is_portuguese(
        self, client: TestClient, garbage: str
    ) -> None:
        resp = client.get(self.URL, headers={"X-VF-Lang": garbage})

        assert resp.json()["detail"] == "Run 'nope' não encontrado."

    def test_a_region_suffix_still_reads_as_the_language(
        self, client: TestClient
    ) -> None:
        resp = client.get(self.URL, headers={"X-VF-Lang": "en-US"})

        assert resp.json()["detail"] == "Run 'nope' not found."

    def test_one_request_never_inherits_the_language_of_the_one_before(
        self, client: TestClient
    ) -> None:
        first = client.get(self.URL, headers=EN)
        second = client.get(self.URL)
        third = client.get(self.URL, headers=EN)

        assert first.json()["detail"].endswith("not found.")
        assert second.json()["detail"].endswith("não encontrado.")
        assert third.json()["detail"].endswith("not found.")


class TestRunsAndTheQueue:
    def test_a_missing_run(self, client: TestClient) -> None:
        for method, url in (
            ("get", "/api/runs/x"),
            ("delete", "/api/runs/x"),
            ("post", "/api/runs/x/test"),
            ("get", "/api/runs/x/export_md"),
        ):
            en = getattr(client, method)(url, headers=EN, **self._body(url))
            pt = getattr(client, method)(url, **self._body(url))

            assert (en.status_code, pt.status_code) == (404, 404), url
            assert en.json()["detail"] == "Run 'x' not found."
            assert pt.json()["detail"] == "Run 'x' não encontrado."

    @staticmethod
    def _body(url: str) -> dict[str, Any]:
        return {"json": {"data_dir": "x"}} if url.endswith("/test") else {}

    def test_an_id_that_is_not_queued(self, client: TestClient) -> None:
        en = client.delete("/api/queue/ghost", headers=EN)
        pt = client.delete("/api/queue/ghost")

        assert en.status_code == pt.status_code == 404
        assert en.json()["detail"] == (
            "No queued run with that id — it may have already started."
        )
        assert pt.json()["detail"] == (
            "Nenhum run na fila com esse id — talvez ele já tenha começado."
        )

    def test_a_running_job_that_cannot_be_stopped(
        self, client: TestClient, env
    ) -> None:
        _, routes_mod = env
        occupy_queue(routes_mod, "busy_run")
        routes_mod._RUN_QUEUE._active.stop_at = None

        en = client.delete("/api/queue/busy_run", headers=EN)
        pt = client.delete("/api/queue/busy_run")

        assert en.status_code == pt.status_code == 409
        assert en.json()["detail"].startswith("This run does not check for stop")
        assert en.json()["detail"].endswith("Nothing was interrupted.")
        assert pt.json()["detail"].startswith("Esta execução não verifica pedidos")
        assert pt.json()["detail"].endswith("Nada foi interrompido.")

    def test_a_result_that_is_not_ready(self, client: TestClient, env) -> None:
        _, routes_mod = env
        routes_mod._current_run = {
            "run_id": "r1",
            "status": "running",
            "error": None,
            "report": None,
            "run_dir": None,
        }

        en = client.get("/api/experiment/result/r1", headers=EN)
        pt = client.get("/api/experiment/result/r1")

        assert en.status_code == pt.status_code == 409
        assert en.json()["detail"] == "Experiment is still running."
        assert pt.json()["detail"] == "O experimento ainda está em execução."

    def test_a_failed_run_wraps_its_error_in_the_page_language(
        self, client: TestClient, env
    ) -> None:
        _, routes_mod = env
        routes_mod._current_run = {
            "run_id": "r2",
            "status": "failed",
            "error": "OSError: disk full",
            "report": None,
            "run_dir": None,
        }

        en = client.get("/api/experiment/result/r2", headers=EN)
        pt = client.get("/api/experiment/result/r2")

        assert en.status_code == pt.status_code == 500
        assert en.json()["detail"] == "Experiment failed: OSError: disk full"
        assert pt.json()["detail"] == "O experimento falhou: OSError: disk full"

    def test_a_run_in_a_running_job_cannot_be_deleted(
        self, client: TestClient, env, tmp_path: Path
    ) -> None:
        _, routes_mod = env
        run_dir = tmp_path / "outputs" / "models" / "e" / "20260522_100000_000000"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text("{}", encoding="utf-8")
        routes_mod._current_run = {"run_id": run_dir.name, "status": "running"}

        en = client.delete(f"/api/runs/{run_dir.name}", headers=EN)
        pt = client.delete(f"/api/runs/{run_dir.name}")

        assert en.status_code == pt.status_code == 409
        assert en.json()["detail"] == "Cannot delete a run that is currently executing."
        assert (
            pt.json()["detail"] == "Não é possível excluir um run que está em execução."
        )

    def test_a_replicate_group_has_nothing_to_act_on(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "outputs" / "models" / "g" / "20260522_100000_000000"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            '{"timestamp": "2026-05-22T10:00:00", "group": {"kind": "replicates"}}',
            encoding="utf-8",
        )

        en = client.post(f"/api/runs/{run_dir.name}/resume", headers=EN)
        pt = client.post(f"/api/runs/{run_dir.name}/resume")

        assert en.status_code == pt.status_code == 400
        assert en.json()["detail"].startswith("This run is a set of replicates")
        assert pt.json()["detail"].startswith("Esta execução é um conjunto de réplicas")


class TestRevealRefusals:
    def _run(self, tmp_path: Path) -> str:
        run_dir = tmp_path / "outputs" / "models" / "e" / "20260522_100000_000000"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text("{}", encoding="utf-8")
        return run_dir.name

    def test_a_page_from_another_site(self, client: TestClient, tmp_path: Path) -> None:
        run_id = self._run(tmp_path)
        url = f"/api/runs/{run_id}/reveal"
        evil = {"Origin": "https://evil.example"}

        en = client.post(url, headers={**evil, **EN})
        pt = client.post(url, headers=evil)

        assert en.status_code == pt.status_code == 403
        assert en.json()["detail"] == (
            "The folder can only be opened from VisionForge's own page."
        )
        assert pt.json()["detail"] == (
            "A pasta só pode ser aberta pela página do próprio VisionForge."
        )

    def test_a_client_on_another_machine(self, env, tmp_path: Path) -> None:
        app, _ = env
        remote = TestClient(app, client=("203.0.113.9", 50000))
        run_id = self._run(tmp_path)
        url = f"/api/runs/{run_id}/reveal"

        en = remote.post(url, headers=EN)
        pt = remote.post(url)

        assert en.status_code == pt.status_code == 403
        assert en.json()["detail"] == (
            "The folder can only be opened from the machine running the server."
        )
        assert pt.json()["detail"] == (
            "A pasta só pode ser aberta a partir da máquina que executa o servidor."
        )


class TestProfileErrors:
    def test_a_slug_that_is_not_safe(self, client: TestClient) -> None:
        en = client.get("/api/runs", headers={**EN, PROFILE: "Not A Slug"})
        pt = client.get("/api/runs", headers={PROFILE: "Not A Slug"})

        assert en.status_code == pt.status_code == 400
        assert en.json()["detail"].startswith(
            "'Not A Slug' is not a valid profile name"
        )
        assert pt.json()["detail"].startswith(
            "'Not A Slug' não é um nome de perfil válido"
        )

    def test_a_profile_that_does_not_exist(self, client: TestClient) -> None:
        en = client.get("/api/runs", headers={**EN, PROFILE: "ghost"})
        pt = client.get("/api/runs", headers={PROFILE: "ghost"})

        assert en.status_code == pt.status_code == 404
        assert en.json()["detail"] == "Profile 'ghost' does not exist."
        assert pt.json()["detail"] == "O perfil 'ghost' não existe."

    def test_a_name_with_nothing_usable_in_it(self, client: TestClient) -> None:
        en = client.post("/api/profiles", json={"name": "!!!"}, headers=EN)
        pt = client.post("/api/profiles", json={"name": "!!!"})

        assert en.status_code == pt.status_code == 422
        assert (
            en.json()["detail"] == "The name needs at least one letter (a-z) or digit."
        )
        assert pt.json()["detail"] == (
            "O nome precisa ter ao menos uma letra (a-z) ou um número."
        )

    def test_a_reserved_name(self, client: TestClient) -> None:
        en = client.post("/api/profiles", json={"name": "con"}, headers=EN)
        pt = client.post("/api/profiles", json={"name": "con"})

        assert en.status_code == pt.status_code == 422
        assert en.json()["detail"] == "'con' is a reserved name. Choose another."
        assert pt.json()["detail"] == "'con' é um nome reservado. Escolha outro."

    def test_a_name_that_is_taken(self, client: TestClient) -> None:
        assert client.post("/api/profiles", json={"name": "Ana"}).status_code == 201

        en = client.post("/api/profiles", json={"name": "Ana"}, headers=EN)
        pt = client.post("/api/profiles", json={"name": "Ana"})

        assert en.status_code == pt.status_code == 409
        assert en.json()["detail"] == "A profile 'Ana' already exists (folder 'ana')."
        assert pt.json()["detail"] == "Já existe um perfil 'Ana' (pasta 'ana')."

    def test_a_sweep_cannot_move_a_profiles_output_folder(
        self, client: TestClient
    ) -> None:
        body = {"config": {}, "search_space": {"output.models_dir": ["a", "b"]}}

        en = client.post("/api/regression/sweep", json=body, headers=EN)
        pt = client.post("/api/regression/sweep", json=body)

        assert en.status_code == pt.status_code == 422
        assert en.json()["detail"] == (
            "The output folder is set by the profile and cannot be changed in a "
            "sweep: remove 'output.models_dir'."
        )
        assert pt.json()["detail"] == (
            "A pasta de saída é definida pelo perfil e não pode ser alterada numa "
            "varredura: remova 'output.models_dir'."
        )


class TestDatasetScanMessages:
    def test_split_detection_without_a_path(self, client: TestClient) -> None:
        en = client.post("/api/dataset/detect", json={"base_dir": ""}, headers=EN)
        pt = client.post("/api/dataset/detect", json={"base_dir": ""})

        assert en.json()["message"] == "Enter the path of the dataset's base directory."
        assert pt.json()["message"] == "Informe o caminho do diretório base do dataset."

    def test_split_detection_names_the_roles_in_the_page_language(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        (tmp_path / "train").mkdir()
        (tmp_path / "val").mkdir()
        body = {"base_dir": str(tmp_path)}

        en = client.post("/api/dataset/detect", json=body, headers=EN).json()
        pt = client.post("/api/dataset/detect", json=body).json()

        assert en["message"] == (
            "Partially detected. Missing: test. Select the remaining folders by hand."
        )
        assert pt["message"] == (
            "Detectado parcialmente. Faltando: teste. Selecione manualmente as "
            "pastas restantes."
        )

    def test_split_detection_success(self, client: TestClient, tmp_path: Path) -> None:
        for name in ("train", "val", "test"):
            (tmp_path / name).mkdir()
        body = {"base_dir": str(tmp_path)}

        en = client.post("/api/dataset/detect", json=body, headers=EN).json()
        pt = client.post("/api/dataset/detect", json=body).json()

        assert en["detected"] is pt["detected"] is True
        assert en["message"] == (
            "Splits detected: training='train', validation='val', test='test'."
        )
        assert pt["message"] == (
            "Splits detectados: treino='train', validação='val', teste='test'."
        )

    @pytest.mark.parametrize(
        "path",
        [
            "/api/dataset/stats",
            "/api/detection/dataset/stats",
            "/api/segmentation/dataset/stats",
            "/api/anomaly/dataset/stats",
            "/api/regression/dataset/stats",
        ],
    )
    def test_a_missing_base_directory_in_every_stats_endpoint(
        self, client: TestClient, tmp_path: Path, path: str
    ) -> None:
        body = {"base_dir": str(tmp_path / "nowhere")}

        en = client.post(path, json=body, headers=EN)
        pt = client.post(path, json=body)

        assert en.status_code == pt.status_code == 200, en.text
        assert en.json()["message"] == f"Base directory not found: {body['base_dir']}"
        assert (
            pt.json()["message"] == f"Diretório base não encontrado: {body['base_dir']}"
        )

    def test_a_split_that_is_not_there(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        body = {"base_dir": str(tmp_path), "split": "val"}

        en = client.post("/api/dataset/samples", json=body, headers=EN).json()
        pt = client.post("/api/dataset/samples", json=body).json()

        assert en["message"] == f"Split 'val' not found in {tmp_path}."
        assert pt["message"] == f"Split 'val' não encontrado em {tmp_path}."

    def test_a_dataset_with_no_split_at_all(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        body = {"base_dir": str(tmp_path)}

        en = client.post("/api/segmentation/dataset/stats", json=body, headers=EN)
        pt = client.post("/api/segmentation/dataset/stats", json=body)

        assert (
            en.json()["message"] == "No split found (expected <split>/{images,masks})."
        )
        assert pt.json()["message"] == (
            "Nenhum split encontrado (esperado <split>/{imagens,máscaras})."
        )

    def test_an_anomaly_dataset_without_normal_training_images(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        body = {"base_dir": str(tmp_path)}

        en = client.post("/api/anomaly/dataset/stats", json=body, headers=EN).json()
        pt = client.post("/api/anomaly/dataset/stats", json=body).json()

        assert en["message"].startswith("Normal training folder not found: ")
        assert pt["message"].startswith("Pasta de treino normal não encontrada: ")


class TestFolderPickers:
    def test_the_container_hint(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VISIONFORGE_CONTAINER", "1")

        en = client.post("/api/dataset/pick", headers=EN).json()
        pt = client.post("/api/dataset/pick").json()

        assert en["cancelled"] is pt["cancelled"] is True
        assert en["message"].startswith("The native picker does not open inside")
        assert "/work/datasets/my-dataset" in en["message"]
        assert pt["message"].startswith("O seletor nativo não abre dentro do container")
        assert "/work/datasets/meu-dataset" in pt["message"]

    def test_a_dialog_that_fails_to_open(self, client: TestClient) -> None:
        with patch("tkinter.Tk", side_effect=RuntimeError("no display")):
            en = client.post("/api/dataset/pick", headers=EN).json()
            pt = client.post("/api/dataset/pick").json()

        assert en["message"] == "Could not open the picker: no display"
        assert pt["message"] == "Falha ao abrir o seletor: no display"

    def test_a_dismissed_dialog_keeps_the_sentence_the_page_replaces(
        self, client: TestClient
    ) -> None:
        # The page swaps this exact sentence for its own dictionary text
        # (lib/picker-feedback.ts), so it must not depend on the header.
        with (
            patch("tkinter.Tk"),
            patch("tkinter.filedialog.askdirectory", return_value=""),
        ):
            en = client.post("/api/dataset/pick", headers=EN).json()
            pt = client.post("/api/dataset/pick").json()

        assert en["message"] == pt["message"] == "Cancelado."

    @pytest.mark.parametrize(
        ("url", "dialog", "en_title", "pt_title"),
        [
            (
                "/api/dataset/pick",
                "askdirectory",
                "Select the dataset's base directory",
                "Selecione o diretório base do dataset",
            ),
            (
                "/api/checkpoint/pick",
                "askopenfilename",
                "Select a checkpoint (.pth or .pt)",
                "Selecione um checkpoint (.pth ou .pt)",
            ),
            (
                "/api/detection/dataset/pick_yaml",
                "askopenfilename",
                "Select the dataset's data.yaml",
                "Selecione o data.yaml do dataset",
            ),
        ],
    )
    def test_the_title_of_the_native_dialog(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        url: str,
        dialog: str,
        en_title: str,
        pt_title: str,
    ) -> None:
        monkeypatch.delenv("VISIONFORGE_CONTAINER", raising=False)
        with patch("tkinter.Tk"), patch(f"tkinter.filedialog.{dialog}") as picker:
            picker.return_value = ""
            client.post(url, headers=EN)
            client.post(url)

        titles = [call.kwargs["title"] for call in picker.call_args_list]
        assert titles == [en_title, pt_title]


class TestTestingARunOnAnotherFolder:
    def _stopped_before_any_checkpoint(self, tmp_path: Path) -> str:
        run_dir = tmp_path / "outputs" / "models" / "e" / "20260522_100000_000000"
        run_dir.mkdir(parents=True)
        config = _classification_full(tmp_path)
        import json

        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "timestamp": "2026-05-22T10:00:00",
                    "status": "stopped",
                    "config": {**config, "task": "regression"},
                    "artifacts": {},
                }
            ),
            encoding="utf-8",
        )
        return run_dir.name

    def test_a_path_that_does_not_exist(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        run_id = self._stopped_before_any_checkpoint(tmp_path)
        body = {"data_dir": str(tmp_path / "nowhere")}

        en = client.post(f"/api/runs/{run_id}/test", json=body, headers=EN)
        pt = client.post(f"/api/runs/{run_id}/test", json=body)

        assert en.status_code == pt.status_code == 400, en.text
        assert en.json()["detail"].startswith("Path not found: ")
        assert pt.json()["detail"].startswith("Caminho não encontrado: ")

    def test_regression_wants_a_manifest_not_a_folder(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        run_id = self._stopped_before_any_checkpoint(tmp_path)
        body = {"data_dir": str(tmp_path)}

        en = client.post(f"/api/runs/{run_id}/test", json=body, headers=EN)
        pt = client.post(f"/api/runs/{run_id}/test", json=body)

        assert en.status_code == pt.status_code == 400, en.text
        assert en.json()["detail"].startswith("Regression is scored on a manifest")
        assert pt.json()["detail"].startswith("Regressão avalia um manifesto")

    def test_a_run_with_no_checkpoint(self, client: TestClient, tmp_path: Path) -> None:
        run_id = self._stopped_before_any_checkpoint(tmp_path)
        manifest = tmp_path / "test.csv"
        manifest.write_text("image,y\n", encoding="utf-8")
        body = {"data_dir": str(manifest)}

        en = client.post(f"/api/runs/{run_id}/test", json=body, headers=EN)
        pt = client.post(f"/api/runs/{run_id}/test", json=body)

        assert en.status_code == pt.status_code == 400, en.text
        assert en.json()["detail"] == (
            f"Run '{run_id}' has no usable checkpoint (artifacts.model: None)."
        )
        assert pt.json()["detail"] == (
            f"Run '{run_id}' não tem um checkpoint utilizável (artifacts.model: None)."
        )


class TestMessagesWrittenAfterTheRequestIsOver:
    """A queued job starts long after its request, in another task and a thread."""

    def test_a_submission_remembers_the_language_it_was_made_in(
        self, client: TestClient, env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, routes_mod = env
        monkeypatch.setattr(routes_mod, "_MODELS_DIR", tmp_path / "models")
        run_dir, _ = _stopped_run(
            tmp_path / "models" / "e", _classification_full(tmp_path), resume_file=True
        )
        seen: dict[str, Any] = {}

        def fake_submit(job_id, label, task, strategy, start, **_kwargs):  # type: ignore[no-untyped-def]
            seen.update(label=label, lang=current_lang())
            return RunResponse(run_id=job_id)

        monkeypatch.setattr(routes_mod, "_submit_job", fake_submit)

        assert (
            client.post(f"/api/runs/{run_dir.name}/resume", headers=EN).status_code
            == 200
        )
        assert seen["lang"] == "en"
        assert seen["label"].endswith("(resuming)")

        assert client.post(f"/api/runs/{run_dir.name}/resume").status_code == 200
        assert seen["lang"] == "pt"
        assert seen["label"].endswith("(retomando)")

    def test_a_job_writes_its_warnings_in_the_language_of_whoever_queued_it(
        self, env
    ) -> None:
        _, routes_mod = env
        written: dict[str, str] = {}

        def make_start(name: str) -> Callable[[], Awaitable[None]]:
            async def start() -> None:
                # The trainer runs in a thread (asyncio.to_thread), so the
                # language has to survive that hop too.
                def in_the_training_thread() -> str:
                    warning = collapsed_predictions([0, 0, 0], n_classes=2)
                    assert warning is not None
                    return warning.message

                written[name] = await asyncio.to_thread(in_the_training_thread)

            return start

        async def scenario() -> None:
            set_lang("en")
            routes_mod._submit_job(
                "job-en", "en", "classification", "simple", make_start("en")
            )
            set_lang("pt")
            routes_mod._submit_job(
                "job-pt", "pt", "classification", "simple", make_start("pt")
            )
            queue = routes_mod._RUN_QUEUE
            while queue.is_busy() or queue.pending_count():
                await asyncio.sleep(0.01)

        asyncio.run(scenario())

        assert "predicted the same class" in written["en"]
        assert "O modelo previu a mesma classe" in written["pt"]
        assert current_lang() == "pt"
