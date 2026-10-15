"""The message catalog and the language a message is written in (ADR-116).

Three layers: the catalog is internally consistent (every key in both languages
with the same placeholders, every key used and every use defined), the language
resolves the way the header documents, and the messages that core code writes
(health warnings, the paging-file hint) follow it.
"""

from __future__ import annotations

import ast
import asyncio
import string
from pathlib import Path

import pytest
import torch

import visionforge
from visionforge.core.loader_lifecycle import describe_worker_spawn_failure
from visionforge.core.training_health import (
    collapsed_predictions,
    collapsed_segmentation,
    constant_predictions,
    frozen_random_backbone,
    no_detections,
    stagnant_loss,
)
from visionforge.utils.messages import (
    CATALOG,
    DEFAULT_LANG,
    Lang,
    bound_lang,
    current_lang,
    lang_from_env,
    message,
    parse_lang,
    resolve_lang,
    set_lang,
    tr,
    tr_artifact,
    using_lang,
)

_SRC = Path(visionforge.__file__).parent
_LANGS: tuple[Lang, ...] = ("pt", "en")


def _placeholders(template: str) -> set[str]:
    names = set()
    for _, field, _, _ in string.Formatter().parse(template):
        if field is not None:
            names.add(field.split(".", 1)[0].split("[", 1)[0])
    return names


def _source_files() -> list[Path]:
    return [
        p
        for p in _SRC.rglob("*.py")
        if "static" not in p.parts and p.name != "messages.py"
    ]


def _string_constants(path: Path) -> list[ast.Constant]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


class TestCatalog:
    @pytest.mark.parametrize("key", sorted(CATALOG))
    def test_every_key_has_both_languages(self, key: str) -> None:
        assert set(CATALOG[key]) == set(_LANGS)
        assert CATALOG[key]["pt"].strip()
        assert CATALOG[key]["en"].strip()

    @pytest.mark.parametrize("key", sorted(CATALOG))
    def test_the_languages_take_the_same_parameters(self, key: str) -> None:
        pt, en = (_placeholders(CATALOG[key][lang]) for lang in _LANGS)

        assert pt == en, f"{key}: pt takes {sorted(pt)}, en takes {sorted(en)}"

    @pytest.mark.parametrize("key", sorted(CATALOG))
    def test_english_is_not_a_copy_of_the_portuguese(self, key: str) -> None:
        assert CATALOG[key]["en"] != CATALOG[key]["pt"]

    @pytest.mark.parametrize("lang", _LANGS)
    @pytest.mark.parametrize("key", sorted(CATALOG))
    def test_every_text_formats_with_its_own_parameters(
        self, key: str, lang: Lang
    ) -> None:
        params = dict.fromkeys(_placeholders(CATALOG[key][lang]), 1.5)

        text = message(lang, key, **params)

        # A doubled brace left in the template would come out doubled.
        assert text
        assert "{{" not in text
        assert "}}" not in text

    def test_every_call_names_a_key_the_catalog_defines(self) -> None:
        unknown: list[str] = []
        for path in _source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in {"tr", "tr_artifact"}
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value not in CATALOG
                ):
                    unknown.append(f"{path.name}:{node.lineno} {node.args[0].value!r}")

        assert unknown == []

    def test_every_key_is_used_somewhere(self) -> None:
        # A key reaches the call as a literal, or as the argument of a helper that
        # forwards it (``_refuse_output_paths``), so any literal in the source counts.
        literals = {
            node.value for path in _source_files() for node in _string_constants(path)
        }

        assert sorted(set(CATALOG) - literals) == []

    def test_a_typo_in_a_key_fails_loudly(self) -> None:
        with pytest.raises(KeyError):
            tr("health.no_such_thing")


class TestResolveLang:
    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("pt", "pt"),
            ("en", "en"),
            ("EN", "en"),
            (" en ", "en"),
            ("en-US", "en"),
            ("en_GB", "en"),
            ("pt-BR", "pt"),
            (None, "pt"),
            ("", "pt"),
            ("   ", "pt"),
            ("fr", "pt"),
            ("klingon", "pt"),
            ("english", "pt"),
            ("en;q=0.9", "pt"),
            ("\x00\x01", "pt"),
        ],
    )
    def test_header_values(self, header: str | None, expected: str) -> None:
        assert resolve_lang(header) == expected

    def test_absent_means_the_language_the_program_always_used(self) -> None:
        assert DEFAULT_LANG == "pt"
        assert current_lang() == "pt"


class TestParseAndEnvironment:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("pt", "pt"),
            ("EN", "en"),
            (" en-US ", "en"),
            ("pt_BR", "pt"),
            (None, None),
            ("", None),
            ("fr", None),
            ("english", None),
        ],
    )
    def test_a_value_names_a_language_or_none(
        self, value: str | None, expected: str | None
    ) -> None:
        assert parse_lang(value) == expected

    def test_resolve_falls_back_to_portuguese_where_parse_gives_none(self) -> None:
        assert parse_lang("fr") is None
        assert resolve_lang("fr") == "pt"

    @pytest.mark.parametrize(
        ("environ", "expected"),
        [
            ({"VISIONFORGE_LANG": "pt"}, "pt"),
            ({"VISIONFORGE_LANG": "en"}, "en"),
            ({"VISIONFORGE_LANG": "EN_us"}, "en"),
            ({"VISIONFORGE_LANG": ""}, None),
            ({"VISIONFORGE_LANG": "klingon"}, None),
            ({}, None),
            ({"LANG": "en_US.UTF-8"}, None),
        ],
    )
    def test_the_variable_selects_a_language_or_none(
        self, environ: dict[str, str], expected: str | None
    ) -> None:
        assert lang_from_env(environ) == expected


class TestArtifactLanguage:
    """A saved file is English unless somebody chose a language for the run."""

    def test_nobody_chose_means_english_for_a_file_and_portuguese_for_a_message(
        self,
    ) -> None:
        assert bound_lang() is None
        assert tr_artifact("plot.title_roc") == "ROC Curve"
        assert tr("plot.title_roc") == "Curva ROC"
        assert current_lang() == "pt"

    def test_choosing_portuguese_is_not_the_same_as_not_choosing(self) -> None:
        with using_lang("pt"):
            assert bound_lang() == "pt"
            assert tr_artifact("plot.title_roc") == "Curva ROC"

    def test_choosing_english(self) -> None:
        with using_lang("en"):
            assert tr_artifact("plot.title_roc") == "ROC Curve"
            assert tr("plot.title_roc") == "ROC Curve"

    def test_the_choice_ends_with_the_block(self) -> None:
        with using_lang("pt"):
            pass

        assert bound_lang() is None

    def test_parameters_are_filled_in(self) -> None:
        assert tr_artifact("plot.title_per_class", metric="IoU") == "IoU per class"


class TestCurrentLang:
    def test_using_lang_restores_the_previous_language(self) -> None:
        with using_lang("en"):
            assert current_lang() == "en"
            with using_lang("pt"):
                assert current_lang() == "pt"
            assert current_lang() == "en"

        assert current_lang() == "pt"

    def test_using_lang_restores_after_an_error(self) -> None:
        with pytest.raises(RuntimeError), using_lang("en"):
            raise RuntimeError("boom")

        assert current_lang() == "pt"

    def test_tr_follows_the_language_and_fills_parameters(self) -> None:
        with using_lang("en"):
            assert tr("scan.split_missing", split="val") == "Split 'val' not found."
        assert tr("scan.split_missing", split="val") == "Split 'val' não encontrado."

    def test_literal_braces_survive(self) -> None:
        assert tr("scan.no_seg_split").endswith("<split>/{imagens,máscaras}).")

    def test_a_worker_thread_inherits_the_language(self) -> None:
        async def scenario() -> str:
            set_lang("en")
            return await asyncio.to_thread(lambda: current_lang())

        assert asyncio.run(scenario()) == "en"

    def test_a_task_does_not_leak_its_language_to_the_next(self) -> None:
        async def first() -> None:
            set_lang("en")

        async def second() -> str:
            return current_lang()

        async def scenario() -> str:
            await asyncio.create_task(first())
            seen: str = await asyncio.create_task(second())
            return seen

        assert asyncio.run(scenario()) == "pt"


class TestHealthWarningsFollowTheLanguage:
    """The same warning, in the language of whoever started the run."""

    def _each(self):  # type: ignore[no-untyped-def]
        return [
            collapsed_predictions([0, 0, 0], n_classes=2),
            stagnant_loss([1.4, 1.39, 1.395]),
            constant_predictions([31.4] * 20),
            frozen_random_backbone(
                pretrained=False, mode="feature_extraction", frozen_params=31_400_000
            ),
            collapsed_segmentation([0.9, 0.0, 0.0], present_classes=3),
            no_detections(0.0, 5),
        ]

    def test_portuguese_is_the_default_and_is_what_it_always_said(self) -> None:
        warnings = self._each()

        assert all(w is not None for w in warnings)
        assert "mesma classe (0)" in warnings[0].message  # type: ignore[union-attr]
        assert "praticamente não caiu" in warnings[1].message  # type: ignore[union-attr]
        assert "mesmo valor" in warnings[2].message  # type: ignore[union-attr]
        assert "única classe" in warnings[4].message  # type: ignore[union-attr]
        assert "mAP@50 igual a zero" in warnings[5].message  # type: ignore[union-attr]

    def test_english_words_every_warning_and_keeps_the_codes(self) -> None:
        pt_codes = [w.code for w in self._each() if w]
        with using_lang("en"):
            warnings = self._each()

        assert [w.code for w in warnings if w] == pt_codes
        assert (
            "predicted the same class (0) for all 3 validation" in warnings[0].message
        )  # type: ignore[union-attr]
        assert "barely moved (1.4000 → 1.3900 over 3 epochs)" in warnings[1].message  # type: ignore[union-attr]
        assert "practically the same value (31.4000)" in warnings[2].message  # type: ignore[union-attr]
        assert "single class on every pixel" in warnings[4].message  # type: ignore[union-attr]
        assert "finished 5 epoch(s) with mAP@50" in warnings[5].message  # type: ignore[union-attr]
        for w in warnings:
            assert w is not None
            assert "não" not in w.message
            assert "modelo" not in w.message

    def test_a_custom_label_is_used_as_given(self) -> None:
        with using_lang("en"):
            warning = constant_predictions([2.0] * 10, label="temperature")

        assert warning is not None
        assert "the same temperature (2.0000)" in warning.message

    def test_the_frozen_count_is_grouped_the_way_the_language_writes_it(self) -> None:
        pt = frozen_random_backbone(
            pretrained=False, mode="feature_extraction", frozen_params=31_400_000
        )
        with using_lang("en"):
            en = frozen_random_backbone(
                pretrained=False, mode="feature_extraction", frozen_params=31_400_000
            )

        assert pt is not None
        assert en is not None
        assert "31.400.000 pesos" in pt.message
        assert "froze 31,400,000 weights" in en.message
        # The Portuguese sentence's own comma used to be turned into a period too.
        assert "pré-treinados, ou use" in pt.message


class TestPagingFileHint:
    @staticmethod
    def _winerror_1455() -> OSError:
        exc = OSError("Error loading shm.dll")
        exc.winerror = 1455  # type: ignore[attr-defined]
        return exc

    def test_portuguese_by_default(self) -> None:
        hint = describe_worker_spawn_failure(self._winerror_1455())

        assert hint is not None
        assert "espaço de paginação" in hint

    def test_english_under_the_english_interface(self) -> None:
        with using_lang("en"):
            hint = describe_worker_spawn_failure(self._winerror_1455())

        assert hint is not None
        assert "ran out of paging file space" in hint
        assert "data.num_workers" in hint
        assert "paginação" not in hint


class TestErrorsRaisedBelowTheRoutes:
    """The exceptions a route relays as its ``detail`` are worded per language too.

    Each case raises the message where it is written, below the route layer, so the
    language has to reach the dataset providers, the Grad-CAM and ONNX helpers and
    the task registry (ADR-116, third slice).
    """

    @staticmethod
    def _raise(case: str, tmp_path: Path) -> str:
        from visionforge.core.gradcam import resolve_target_layer
        from visionforge.gui.api.dataset_download import (
            _parse_roboflow_dataset,
            _require_images,
            download_roboflow,
            download_torchvision,
        )
        from visionforge.gui.api.torch_gradcam import build_gradcam
        from visionforge.gui.api.torch_onnx_export import _checkpoint
        from visionforge.tasks.manage import delete_task
        from visionforge.tasks.registry import get_task
        from visionforge.utils.credentials import save_credential

        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
        calls = {
            "torchvision": lambda: download_torchvision("nope", tmp_path),
            "empty": lambda: _require_images(tmp_path, "Roboflow", "ws/proj"),
            "roboflow_format": lambda: _parse_roboflow_dataset("only-one-part"),
            "roboflow_key": lambda: download_roboflow("ws/proj", tmp_path, api_key=""),
            "onnx": lambda: _checkpoint({}, "exp_1"),
            "gradcam_task": lambda: build_gradcam(
                {"config": {"task": "detection"}}, target_index=None, checkpoint="x"
            ),
            "gradcam_layer": lambda: resolve_target_layer(torch.nn.Linear(2, 2)),
            "confirm": lambda: delete_task("toy", "oops", tmp_path),
            "no_file": lambda: delete_task("ghost", "ghost", tmp_path),
            "not_registered": lambda: get_task("ghost", user_tasks_dir=tmp_path),
            "blank_key": lambda: save_credential("roboflow", "   "),
        }
        with pytest.raises((ValueError, FileNotFoundError, KeyError)) as exc:
            calls[case]()
        return str(exc.value.args[0])

    @pytest.mark.parametrize(
        ("case", "pt", "en"),
        [
            (
                "torchvision",
                "Dataset do torchvision desconhecido",
                "Unknown torchvision",
            ),
            ("empty", "nenhuma imagem chegou", "no images landed"),
            (
                "roboflow_format",
                "precisa ser 'workspace/project'",
                "must be 'workspace",
            ),
            ("roboflow_key", "precisa de uma api_key", "requires an api_key"),
            ("onnx", "não tem checkpoint registrado", "has no checkpoint recorded"),
            ("gradcam_task", "não está disponível para a tarefa", "is not available"),
            ("gradcam_layer", "não tem uma camada nn.Conv2d", "has no nn.Conv2d"),
            ("confirm", "repetir exatamente a chave", "must repeat the task key"),
            ("no_file", "Nenhum arquivo de tarefa", "No task file found"),
            ("not_registered", "não está registrada", "is not registered"),
            ("blank_key", "não pode ficar vazio", "cannot be empty"),
        ],
    )
    def test_portuguese_by_default_and_english_when_asked(
        self, case: str, pt: str, en: str, tmp_path: Path
    ) -> None:
        assert pt in self._raise(case, tmp_path)
        with using_lang("en"):
            assert en in self._raise(case, tmp_path)

    def test_the_stray_extensions_follow_the_sentence(self, tmp_path: Path) -> None:
        with using_lang("en"):
            text = self._raise("empty", tmp_path)

        assert text.endswith("has images. Found files with these extensions: .txt.")

    def test_no_registered_task_is_named_in_the_language(self, tmp_path: Path) -> None:
        from visionforge.tasks.registry import clear_task_registry

        clear_task_registry()
        assert "(disponíveis: nenhuma)" in self._raise("not_registered", tmp_path)
        with using_lang("en"):
            assert "(available: none)" in self._raise("not_registered", tmp_path)
