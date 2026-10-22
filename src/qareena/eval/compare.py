"""Score one extracted field against ground truth, by field kind.

"Correct" means equal after the *strict* layer only (invisible differences), plus kind-specific
canonical forms: dates are parsed with the document's printed format, codes and digits lose
spaces and case, MRZ lines lose whitespace. Spelling variants are NOT counted as correct here:
the reader must transcribe what is printed.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from qareena.config import FieldKind, FieldSpec
from qareena.text import arabic

_NUM = re.compile(r"\d+")


@dataclass(frozen=True)
class FieldScore:
    expected: str | None  # canonical form, None = field absent or unreadable (expect null)
    predicted: str | None
    correct: bool
    cer: float | None  # character error rate vs expected; None when expected is None
    false_fill: bool  # a value was returned where null was expected
    false_null: bool  # null was returned where a value was expected


def canonical(value: object, spec: FieldSpec) -> str | None:
    if value is None:
        return None
    if spec.kind == FieldKind.DATE and isinstance(value, dt.date):
        return value.isoformat()
    text = arabic.strict(str(value))
    if not text:
        return None
    if spec.kind in (FieldKind.DIGITS, FieldKind.CODE):
        return text.replace(" ", "").upper()
    if spec.kind == FieldKind.MRZ_LINE:
        return "".join(text.split()).upper()
    if spec.kind == FieldKind.DATE:
        parsed = parse_date(text, spec.date_format or "%Y/%m/%d")
        if parsed:
            return parsed.isoformat()
        try:
            return dt.date.fromisoformat(text).isoformat()
        except ValueError:
            return f"?{text}"  # unparseable: never equal to a real date
    return text


def parse_date(text: str, fmt: str) -> dt.date | None:
    """Parse a printed date using the order of %Y/%m/%d in ``fmt``; digits may be Arabic-Indic."""
    nums = _NUM.findall(arabic.normalize_digits(text))
    order = [c for c in re.findall(r"%([Ymd])", fmt)]
    if len(nums) != 3 or len(order) != 3:
        return None
    parts = dict(zip(order, (int(n) for n in nums), strict=True))
    try:
        return dt.date(parts["Y"], parts["m"], parts["d"])
    except ValueError:
        return None


def score_field(expected: object, predicted: object, spec: FieldSpec) -> FieldScore:
    exp, pred = canonical(expected, spec), canonical(predicted, spec)
    if exp is None:
        return FieldScore(None, pred, pred is None, None, pred is not None, False)
    if pred is None:
        return FieldScore(exp, None, False, 1.0, False, True)
    cer = Levenshtein.distance(exp, pred) / max(len(exp), 1)
    return FieldScore(exp, pred, exp == pred, cer, False, False)
