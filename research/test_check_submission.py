"""Gate de submissao do paper: so falha com --strict, e aponta arquivo:linha."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "paper" / "check_submission.py"
STRICT = "--strict"
MD_PLACEHOLDER = "date: <submission date, e.g. 9 October 2026>"
TEX_PLACEHOLDER = r"\todo{x}"
TEX_DEFINITION = r"\newcommand{\todo}[1]{#1}"
TEX_ESCAPED_PERCENT = r"50\% "
NL = chr(10)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_submission", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load()


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_strict_com_placeholder_falha_e_aponta_arquivo_e_linha(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    md = _write(tmp_path, "paper.md", f"---\ntitle: t\n{MD_PLACEHOLDER}\n---\n")
    tex = _write(tmp_path, "main.tex", f"a\nb\n{TEX_PLACEHOLDER}\n")
    code = check.main([STRICT, str(md), str(tex)])
    captured = capsys.readouterr()
    assert code == 1
    assert f"{md.as_posix()}:3:" in captured.out
    assert f"{tex.as_posix()}:3:" in captured.out
    assert "--strict" in captured.err


def test_sem_strict_lista_mas_nao_falha(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    md = _write(tmp_path, "paper.md", MD_PLACEHOLDER + "\n")
    tex = _write(tmp_path, "main.tex", TEX_PLACEHOLDER + "\n")
    code = check.main([str(md), str(tex)])
    captured = capsys.readouterr()
    assert code == 0
    assert "2 pendencia(s)" in captured.out


def test_strict_em_arquivos_limpos_passa(tmp_path: Path) -> None:
    md = _write(
        tmp_path, "paper.md", "date: 9 October 2026\nVeja <https://example.org/x>.\n"
    )
    tex = _write(
        tmp_path, "main.tex", TEX_DEFINITION + NL + "% " + TEX_PLACEHOLDER + NL
    )
    assert check.main([STRICT, str(md), str(tex)]) == 0


def test_comentario_html_e_codigo_markdown_sao_ignorados(tmp_path: Path) -> None:
    text = (
        "<!-- preencha <cada campo aqui> -->\nuse `<a b>` e TODO_ok\n```\n<x y>\n```\n"
    )
    md = _write(tmp_path, "paper.md", text)
    assert check.find_pending(md) == []


def test_placeholder_multilinha_aponta_a_primeira_linha(tmp_path: Path) -> None:
    md = _write(tmp_path, "paper.md", "ok\n\n<Estudo: numero\nde participantes.> fim\n")
    assert check.find_pending(md) == [(3, "<Estudo: numero")]


def test_palavras_todo_e_tbd_no_markdown_sao_pendencia(tmp_path: Path) -> None:
    md = _write(tmp_path, "paper.md", "ok\nTBD aqui\nFIXME\n")
    assert [line for line, _ in check.find_pending(md)] == [2, 3]


def test_tex_ignora_porcentagem_escapada_e_acha_todo_apos_ela(tmp_path: Path) -> None:
    tex = _write(tmp_path, "main.tex", TEX_ESCAPED_PERCENT + TEX_PLACEHOLDER + NL)
    assert check.find_pending(tex) == [(1, TEX_PLACEHOLDER)]


def test_caminho_fora_da_raiz_e_exibido_como_recebido(tmp_path: Path) -> None:
    md = _write(tmp_path, "paper.md", "ok\n")
    assert check._display(md) == md.as_posix()


def test_arquivos_reais_nao_bloqueiam_sem_strict() -> None:
    assert check.main([]) == 0
