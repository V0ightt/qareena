"""Arabic text normalisation for *comparison*, never for display.

Three layers, applied cumulatively and only as far as a comparison needs:

1. ``strict``   — invisible differences only: Unicode NFKC, diacritics, tatweel,
                  bidi/zero-width controls, Arabic-Indic and Persian digits, whitespace.
                  Two strings equal after this are the *same* text.
2. ``spelling`` — the same letter written by a different convention: alef forms,
                  ta marbuta / ha, alef maqsura / ya, Persian/Kurdish code points for
                  kaf and ya. Equal after this means *spelling variant*.
3. ``sound``    — different letters that Iraqis use for the same sound (گ/ك/ق, چ/ج,
                  پ/ب, ڤ/ف) and Sorani letters folded to their Arabic neighbours.
                  Equal only after this means *sound-alike*; never treated as a match.

Code points are written as escapes on purpose: Arabic literals in source are easy to
corrupt or misread in editors that apply the bidi algorithm.
"""

from __future__ import annotations

import re
import unicodedata

# --- layer 1: strict ---------------------------------------------------------------

_DIGITS = str.maketrans(
    {
        **{0x0660 + i: str(i) for i in range(10)},  # Arabic-Indic ٠-٩
        **{0x06F0 + i: str(i) for i in range(10)},
    }  # Extended Arabic-Indic (Persian/Urdu) ۰-۹
)

_TATWEEL = "\u0640"
# Harakat, tanwin, shadda, sukun, superscript alef and Quranic annotation marks.
_MARKS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e8\u06ea-\u06ed]")
# Bidi and zero-width controls: ZWSP..RLM, ALM, LRE..RLO, LRI..PDI, BOM/ZWNBSP.
_CONTROLS = re.compile("[\u200b-\u200f\u061c\u202a-\u202e\u2066-\u2069\ufeff]")
_SPACE = re.compile(r"\s+")


def normalize_digits(text: str) -> str:
    """Map Arabic-Indic and Persian digits to ASCII 0-9. Other characters are untouched."""
    return text.translate(_DIGITS)


def strict(text: str) -> str:
    """Remove differences a reader cannot see. Safe for any comparison, including IDs."""
    text = unicodedata.normalize("NFKC", text)  # also folds Arabic presentation forms
    text = _CONTROLS.sub("", text)
    text = _MARKS.sub("", text).replace(_TATWEEL, "")
    text = normalize_digits(text)
    return _SPACE.sub(" ", text).strip()


# --- layer 2: spelling --------------------------------------------------------------

_SPELLING = str.maketrans(
    {
        "\u0623": "\u0627",  # أ -> ا
        "\u0625": "\u0627",  # إ -> ا
        "\u0622": "\u0627",  # آ -> ا
        "\u0671": "\u0627",  # ٱ -> ا
        "\u0629": "\u0647",  # ة -> ه
        "\u0649": "\u064a",  # ى -> ي
        "\u06a9": "\u0643",  # ک (Persian/Kurdish kaf) -> ك
        "\u06cc": "\u064a",  # ی (Persian/Kurdish ya) -> ي
    }
)


def spelling(text: str) -> str:
    """Strict normalisation plus same-letter spelling conventions."""
    return strict(text).translate(_SPELLING)


# --- layer 3: sound -----------------------------------------------------------------

_SOUND = str.maketrans(
    {
        "\u06af": "\u0643",  # گ -> ك
        "\u0642": "\u0643",  # ق -> ك   (گاطع / كاطع / قاطع)
        "\u0686": "\u062c",  # چ -> ج
        "\u067e": "\u0628",  # پ -> ب
        "\u06a4": "\u0641",  # ڤ -> ف
        "\u0695": "\u0631",  # ڕ (Sorani) -> ر
        "\u06b5": "\u0644",  # ڵ (Sorani) -> ل
        "\u06c6": "\u0648",  # ۆ (Sorani) -> و
        "\u06ce": "\u064a",  # ێ (Sorani) -> ي
        "\u06d5": "\u0647",  # ە (Sorani) -> ه
    }
)


def sound(text: str) -> str:
    """Spelling normalisation plus Iraqi sound-alike letters. For flagging, never matching."""
    return spelling(text).translate(_SOUND)


SORANI_LETTERS = frozenset("\u0695\u06b5\u06c6\u06ce\u06d5\u06a4")


def has_sorani(text: str) -> bool:
    """True if the text contains letters used in Sorani Kurdish but not in Arabic."""
    return any(ch in SORANI_LETTERS for ch in text)


def is_arabic_letters(text: str, allow: str = " ") -> bool:
    """True if every character is an Arabic-script letter (after strict) or in ``allow``."""
    for ch in strict(text):
        if ch in allow:
            continue
        if not ("\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f") or not ch.isalpha():
            return False
    return bool(strict(text))
