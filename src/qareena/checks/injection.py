"""Flag instruction-like text printed on a document (prompt injection via the image).

The design does not *depend* on this detector: the VLM has no tools, its output is schema-
constrained and code makes every decision, so printed instructions have nothing to hijack.
The detector exists so a reviewer sees the attempt (SUSPICIOUS_TEXT) instead of it passing
silently. It reads the independent OCR text, not the VLM output, because a successful
injection could also suppress the VLM's mention of it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from qareena.text import arabic

# Matched on spelling-folded Arabic (alef forms unified) and lower-cased Latin text.
_PATTERNS = [
    r"\bignore\b.{0,40}\b(instruction|previous|above|rules)",
    r"\b(system|assistant)\s*:",
    r"\b(approve|approved|mark as verified|verified)\b",
    r"\bprompt\b",
    r"تجاهل.{0,30}(التعليمات|الاوامر|القواعد|ما سبق)",
    r"(ملاحظه|تعليمات|رساله)\s+(للنظام|للمساعد|للذكاء)",
    r"اعتبر.{0,40}(مطابق|موافق|مقبول|صحيح)",
    r"(وافق|اقبل)\s+(على\s+)?(هذا\s+)?(الملف|الطلب)",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


def suspicious_lines(lines: Iterable[str]) -> list[str]:
    """OCR lines that look like instructions to an AI system."""
    hits = []
    for line in lines:
        text = arabic.spelling(line).lower()
        if any(p.search(text) for p in _COMPILED):
            hits.append(line)
    return hits
