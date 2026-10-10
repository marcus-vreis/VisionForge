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
    current_lang,
    message,
    resolve_lang,
    set_lang,
    tr,
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
                    and node.func.id == "tr"
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
