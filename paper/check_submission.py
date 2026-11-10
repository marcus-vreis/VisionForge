"""Lista placeholders e TODOs pendentes no paper antes da submissao.

Sem ``--strict`` so imprime as pendencias e sai com 0 (o desenvolvimento normal
continua livre). Com ``--strict`` sai com 1 se houver qualquer pendencia, uma por
linha no formato ``arquivo:linha: texto``. So le os arquivos; nao edita nada.

Uso: ``python paper/check_submission.py [--strict] [arquivo ...]``
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILES = (
    "paper/paper.md",
    "research/seed_variance/paper/main.tex",
)

# <texto com espaco, ate 5 linhas>: autolinks (<https://...>) nao tem espaco.
_ANGLE = re.compile(r"<[A-Za-z][^<>]*\s[^<>]*>")
_MD_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_MD_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
_MD_WORD = re.compile(r"\b(?:TODO|FIXME|TBD)\b")
_TEX_TODO = re.compile(r"\\todo\s*\{")
_TEX_COMMENT = re.compile(r"(?<!\\)%.*")


def _blank(match: re.Match[str]) -> str:
    """Troca o trecho por espacos, mantendo as quebras de linha (numeracao)."""
    return re.sub(r"[^\n]", " ", match.group(0))


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _scan_markdown(text: str) -> list[tuple[int, str]]:
    clean = _MD_CODE.sub(_blank, _MD_COMMENT.sub(_blank, text))
    found = [
        (_line_of(clean, m.start()), m.group(0).split("\n")[0].strip())
        for pattern in (_ANGLE, _MD_WORD)
        for m in pattern.finditer(clean)
    ]
    return sorted(found)


def _scan_tex(text: str) -> list[tuple[int, str]]:
    clean = _TEX_COMMENT.sub(_blank, text)
    return [
        (_line_of(clean, m.start()), clean[m.start() :].split("\n")[0].strip())
        for m in _TEX_TODO.finditer(clean)
    ]


def find_pending(path: Path) -> list[tuple[int, str]]:
    """Pendencias de um arquivo como ``(linha, trecho)``."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".tex":
        return _scan_tex(text)
    return _scan_markdown(text)


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="padrao: paper.md e main.tex")
    parser.add_argument(
        "--strict", action="store_true", help="sai 1 se houver pendencia"
    )
    args = parser.parse_args(argv)
    paths = [Path(f) for f in args.files] or [ROOT / f for f in DEFAULT_FILES]
    total = 0
    for path in paths:
        for line, snippet in find_pending(path):
            total += 1
            print(f"{_display(path)}:{line}: {snippet}")
    print(f"{total} pendencia(s) para a submissao.")
    if args.strict and total:
        print("falha: --strict nao aceita pendencias.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
