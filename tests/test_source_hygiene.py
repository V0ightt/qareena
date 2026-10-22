"""Guard against invisible Unicode in source ("Trojan Source", CVE-2021-42574).

Bidi overrides and zero-width characters can make code read differently from how it runs.
Arabic text is fine in tests and docs; *invisible format characters* (category Cf) are never
fine in source and must be written as escapes or ``chr()`` calls.
"""

import tokenize
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted(
    p
    for pattern in (
        "src/**/*.py",
        "src/**/*.html",
        "src/**/*.j2",
        "src/**/*.css",
        "src/**/*.js",
        "tests/**/*.py",
        "config/**/*.yaml",
        "tools/**/*.py",
    )
    for p in ROOT.glob(pattern)
)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_invisible_format_characters(path):
    text = path.read_text(encoding="utf-8")
    bad = [
        (line_no, f"U+{ord(ch):04X} {unicodedata.name(ch, '?')}")
        for line_no, line in enumerate(text.splitlines(), 1)
        for ch in line
        if unicodedata.category(ch) == "Cf"
    ]
    assert not bad, f"invisible characters in {path.name}: {bad[:5]}"


def _code_tokens_with_non_ascii(path: Path) -> list[tuple[int, str]]:
    """Non-ASCII tokens that are code: comments and docstrings may show Arabic examples."""
    bad, prev = [], tokenize.NEWLINE
    with path.open(encoding="utf-8") as fh:
        for tok in tokenize.generate_tokens(fh.readline):
            is_doc = tok.type == tokenize.STRING and prev in (
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.NL,
            )
            if tok.type not in (tokenize.COMMENT, tokenize.NL):
                prev = tok.type
            if tok.type in (tokenize.COMMENT, tokenize.NL) or is_doc:
                continue
            if not tok.string.isascii():
                bad.append((tok.start[0], tok.string[:30]))
    return bad


def test_library_text_tables_are_ascii():
    # Code-point tables in the text package must be reviewable escapes, never glyph literals.
    for path in (ROOT / "src" / "qareena" / "text").glob("*.py"):
        bad = _code_tokens_with_non_ascii(path)
        assert not bad, f"{path.name}: non-ASCII code tokens {bad[:3]}"
