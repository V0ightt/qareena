"""Deterministic single-field checks. Each returns a reason code, or None when the value passes.

These never accept anything on their own; they can only demote a field (to null / check).
"""

from __future__ import annotations

import datetime as dt
import re

from rapidfuzz.distance import Levenshtein

from qareena.config import FieldKind, FieldSpec
from qareena.text import arabic

# Document boilerplate that must never be returned as a field value.
BOILERPLATE = (
    "SPECIMEN",
    "NOT A VALID DOCUMENT",
    "نموذج تجريبي",
    "غير صالح",
)
_SHORT_LABEL = 5  # labels shorter than this must match exactly, so real names are never flagged
_FUZZY = 0.8
_NUM = re.compile(r"\d+")
_ARTICLE = "ال"


def label_echo(value: str, labels: frozenset[str] | set[str]) -> str | None:
    """Return the label the value echoes, if any.

    Compared at the *sound* layer because models often re-spell a Kurdish label with Arabic
    letters (ە -> ه, ڵ -> ل). A value echoes a label if it equals it, starts with it, or (for
    labels of 5+ letters) is at least 80% similar to it.
    """
    v = _letters(arabic.sound(value))
    if not v:
        return None
    for label in (*labels, *BOILERPLATE):
        lab = _letters(arabic.sound(label))
        if not lab:
            continue
        if v == lab:
            return label
        if len(lab) >= _SHORT_LABEL and (
            v.startswith(lab) or Levenshtein.normalized_similarity(v, lab) >= _FUZZY
        ):
            return label
    return None


def _letters(text: str) -> str:
    """Letters and digits only: dots, spaces and punctuation are ignored when comparing."""
    return "".join(ch for ch in text if ch.isalnum())


_SEP = "[." + chr(0x0640) + " -]"  # dot, tatweel, space or hyphen between letters
_GARBLED = re.compile(r"^(?:\w" + _SEP + r"){2,}\w?" + _SEP + "?$")


def garbled(value: str) -> bool:
    """Single letters separated by dots or spaces ("د.ه.س.ل.ت.ن"): a model spelling out glyphs it
    could not read. Treated as unreadable, never as a value."""
    text = arabic.strict(value)
    return bool(_GARBLED.match(text)) and sum(ch.isalpha() for ch in text) >= 3


def parse_date(text: str, fmt: str) -> dt.date | None:
    """Parse a printed date using the order of %Y/%m/%d in ``fmt``; digits may be Arabic-Indic."""
    nums = _NUM.findall(arabic.normalize_digits(text))
    order = re.findall(r"%([Ymd])", fmt)
    if len(nums) != 3 or len(order) != 3:
        return None
    parts = dict(zip(order, (int(n) for n in nums), strict=True))
    try:
        return dt.date(parts["Y"], parts["m"], parts["d"])
    except ValueError:
        return None


def _date_shape(fmt: str) -> str:
    """'%d/%m/%Y' -> a regex for the printed shape, separators included."""
    return re.escape(fmt).replace("%Y", r"\d{4}").replace("%m", r"\d{1,2}").replace("%d", r"\d{1,2}")


def format_error(value: str, spec: FieldSpec) -> str | None:
    """Kind- and pattern-level validation on the canonical (strict, ASCII-digit) value."""
    text = arabic.strict(value)
    if spec.kind == FieldKind.DATE:
        fmt, compact = spec.date_format or "%Y/%m/%d", text.replace(" ", "")
        if parse_date(compact, fmt) is None:
            return "DATE_UNPARSEABLE"  # a partial date such as «?1995»
        if not re.fullmatch(_date_shape(fmt), compact):
            return "DATE_FORMAT_DIFFERS"  # not copied as printed, e.g. ISO taken from the MRZ
    if spec.kind == FieldKind.NAME_PART:
        core = arabic.spelling(text).replace(" ", "")
        if len(core.removeprefix(_ARTICLE)) < 2:
            return "NAME_TRUNCATED"  # «ال» alone: the rest of the name was not read
    if spec.kind in (FieldKind.DIGITS, FieldKind.CODE):
        text = text.replace(" ", "").upper()
    if spec.kind == FieldKind.DIGITS and not text.isdigit():
        return "FORMAT_NOT_DIGITS"
    if spec.pattern and not re.fullmatch(spec.pattern, text):
        return "FORMAT_INVALID"
    if spec.kind == FieldKind.ENUM and spec.values and text not in {arabic.strict(v) for v in spec.values}:
        return "VALUE_NOT_ALLOWED"
    if spec.kind in (FieldKind.NAME_PART, FieldKind.FULL_NAME) and not arabic.is_arabic_letters(text):
        return "NAME_NOT_ARABIC_LETTERS"
    return None
