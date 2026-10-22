"""Ground a VLM-extracted value in the independent OCR reader's lines.

A value is *grounded* when the OCR engine, reading the same photo on its own, also produced
it. The matched line supplies the evidence box shown to the reviewer. The number of
matching lines is itself a signal: 0 suggests a misread or hallucination, more than 1 means
the value is ambiguous on the page (e.g. father and grandfather share a name), and the
reviewer must look at the box we chose.

Matching happens at the spelling layer (and digits-only for numeric kinds), so the readers
may differ in invisible marks or alef forms but not in letters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz

from qareena.config import FieldKind
from qareena.readers.ocr import OcrLine
from qareena.text import arabic

MATCH = 0.90  # partial-ratio similarity (0..1) at which an OCR line counts as containing the value


@dataclass(frozen=True)
class Grounding:
    matches: int  # OCR lines containing the value at >= MATCH similarity
    similarity: float  # best similarity over all lines, 0..1
    line_index: int | None
    box: tuple[float, float, float, float] | None  # pixels, from the best line
    ocr_score: float | None  # the OCR engine's own confidence for that line
    exact: bool  # the value occurs verbatim (after folding) in the best line


def _fold(text: str, kind: FieldKind) -> str:
    if kind in (FieldKind.DIGITS, FieldKind.DATE):
        return "".join(ch for ch in arabic.normalize_digits(text) if ch.isdigit())
    if kind in (FieldKind.CODE, FieldKind.MRZ_LINE):
        return "".join(arabic.strict(text).split()).upper()
    return arabic.spelling(text).replace(" ", "")


def ground(value: str | None, kind: FieldKind, lines: Sequence[OcrLine]) -> Grounding:
    if not value:
        return Grounding(0, 0.0, None, None, None, False)
    target = _fold(value, kind)
    if not target:
        return Grounding(0, 0.0, None, None, None, False)
    best_i, best_sim, n = None, 0.0, 0
    for i, line in enumerate(lines):
        hay = _fold(line.text, kind)
        if not hay:
            continue
        if target in hay:
            sim = 1.0
        elif len(hay) < max(2, len(target) // 2):
            continue  # a short fragment cannot contain the value
        else:
            sim = fuzz.partial_ratio(target, hay) / 100.0
        if sim >= MATCH:
            n += 1
        if sim > best_sim:
            best_i, best_sim = i, sim
    if best_i is None:
        return Grounding(0, 0.0, None, None, None, False)
    line = lines[best_i]
    return Grounding(n, best_sim, best_i, line.bbox, line.score, target in _fold(line.text, kind))
