"""Compare Iraqi personal names across documents.

The question is never "how similar are these strings" but "what kind of difference is
this", because the answer decides what a reviewer must do:

* SAME         identical after invisible-only normalisation
* VARIANT      identical after same-letter spelling folds (عبدالله / عبد الله, ة/ه, أ/ا)
* SOUND_ALIKE  identical only after Iraqi sound folds (گاطع / قاطع) -> a human checks
* PARTIAL      one name is the other with parts missing (no laqab, no grandfather)
* CONFLICT     an aligned part differs (حسين / حسن) -> a human decides

Fuzzy similarity is reported for display only. It is never used to decide a class:
حسين and حسن are 80% similar and are different people's grandfathers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from rapidfuzz.distance import Levenshtein

from qareena.text import arabic

_ABD = "\u0639\u0628\u062f"  # عبد  — joins with the next token: عبد الله -> عبدالله
_ABU = "\u0627\u0628\u0648"  # ابو  (after spelling fold of أبو)
_AL_DIN = "\u0627\u0644\u062f\u064a\u0646"  # الدين — joins with the previous token: نور الدين
_AL = "\u0627\u0644"  # ال prefix


class Match(IntEnum):
    """Ordered from best to worst; the overall class is the worst part."""

    SAME = 0
    VARIANT = 1
    SOUND_ALIKE = 2
    PARTIAL = 3
    CONFLICT = 4


@dataclass(frozen=True)
class PartResult:
    index: int
    left: str | None
    right: str | None
    match: Match


@dataclass(frozen=True)
class NameComparison:
    match: Match
    parts: tuple[PartResult, ...]
    missing: tuple[str, ...] = field(default=())  # e.g. ("laqab",) or ("grandfather",)
    similarity: float = 0.0  # display only, 0..1

    @property
    def reasons(self) -> tuple[str, ...]:
        out = [f"NAME_{self.match.name}"]
        out += [f"MISSING_{m.upper()}" for m in self.missing]
        return tuple(out)


def tokens(full_name: str) -> list[str]:
    """Split a full name into name parts, joining Arabic compound names.

    ``عبد الرحمن`` and ``أبو بكر`` become one part; ``نور الدين`` becomes one part.
    Joining happens on the spelling-folded form so ``أبو`` and ``ابو`` behave the same.
    """
    raw = arabic.strict(full_name).split(" ")
    raw = [t for t in raw if t]
    out: list[str] = []
    i = 0
    while i < len(raw):
        tok = raw[i]
        folded = arabic.spelling(tok)
        if folded in (_ABD, _ABU) and i + 1 < len(raw):
            out.append(f"{tok} {raw[i + 1]}")  # keep the space: it is a spelling difference
            i += 2
            continue
        if folded == _AL_DIN and out:
            out[-1] = f"{out[-1]} {tok}"
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def compare_part(a: str, b: str) -> Match:
    if arabic.strict(a) == arabic.strict(b):
        return Match.SAME
    if _fold_spelling(a) == _fold_spelling(b):
        return Match.VARIANT
    if _fold_sound(a) == _fold_sound(b):
        return Match.SOUND_ALIKE
    return Match.CONFLICT


def compare_laqab(a: str, b: str) -> Match:
    """Like compare_part, but the definite article is optional: الربيعي = ربيعي (VARIANT)."""
    m = compare_part(a, b)
    if m != Match.CONFLICT:
        return m
    a2, b2 = _drop_al(a), _drop_al(b)
    if (a2, b2) != (a, b):
        inner = compare_part(a2, b2)
        if inner == Match.SAME:
            return Match.VARIANT
        return inner
    return m


def compare_names(
    left: str | list[str],
    right: str | list[str],
    *,
    left_has_laqab: bool | None = None,
    right_has_laqab: bool | None = None,
) -> NameComparison:
    """Compare two names given as full strings or as ordered part lists.

    Parts are ordered: given, father, grandfather, then laqab (optional). When one side is
    shorter, two alignments are tried — a prefix (missing trailing parts) and
    "given + father + laqab" (missing grandfather) — and the better one is kept.
    ``*_has_laqab`` tells whether the last part of a 4-part list is a laqab; for strings
    it is inferred: a 4th part is treated as the laqab.
    """
    lp = tokens(left) if isinstance(left, str) else [p for p in left if p]
    rp = tokens(right) if isinstance(right, str) else [p for p in right if p]
    sim = _similarity(" ".join(lp), " ".join(rp))
    if not lp or not rp:
        return NameComparison(Match.CONFLICT, (), similarity=sim)

    short, long_, swapped = (lp, rp, False) if len(lp) <= len(rp) else (rp, lp, True)
    candidates = [_align_prefix(short, long_)]
    if len(long_) == 4 and len(short) == 3:
        candidates.append(_align_skip_grandfather(short, long_))
    best = min(candidates, key=lambda c: (c.match, len(c.missing)))
    if swapped:
        best = NameComparison(
            best.match,
            tuple(PartResult(p.index, p.right, p.left, p.match) for p in best.parts),
            best.missing,
        )
    return NameComparison(best.match, best.parts, best.missing, sim)


# --- internals ------------------------------------------------------------------------

_PART_NAMES = ("given", "father", "grandfather", "laqab")


def _align_prefix(short: list[str], long_: list[str]) -> NameComparison:
    parts = []
    for i, (a, b) in enumerate(zip(short, long_, strict=False)):
        cmp = compare_laqab if i == 3 else compare_part
        parts.append(PartResult(i, a, b, cmp(a, b)))
    missing = tuple(_PART_NAMES[i] if i < 4 else f"part{i + 1}" for i in range(len(short), len(long_)))
    return _combine(parts, missing)


def _align_skip_grandfather(short: list[str], long_: list[str]) -> NameComparison:
    pairs = [(0, short[0], long_[0]), (1, short[1], long_[1]), (3, short[2], long_[3])]
    parts = [PartResult(i, a, b, (compare_laqab if i == 3 else compare_part)(a, b)) for i, a, b in pairs]
    return _combine(parts, ("grandfather",))


def _combine(parts: list[PartResult], missing: tuple[str, ...]) -> NameComparison:
    worst = max((p.match for p in parts), default=Match.CONFLICT)
    if worst < Match.PARTIAL and missing:
        worst = Match.PARTIAL
    return NameComparison(worst, tuple(parts), missing)


def _fold_spelling(s: str) -> str:
    return arabic.spelling(s).replace(" ", "")


def _fold_sound(s: str) -> str:
    return arabic.sound(s).replace(" ", "")


def _drop_al(s: str) -> str:
    folded = arabic.spelling(s)
    return folded[len(_AL) :] if folded.startswith(_AL) and len(folded) > len(_AL) + 1 else s


def _similarity(a: str, b: str) -> float:
    return float(Levenshtein.normalized_similarity(arabic.spelling(a), arabic.spelling(b)))
