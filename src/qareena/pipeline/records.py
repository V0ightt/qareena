"""Records shared by the pipeline, the reviewer workbench and the evaluation harness.

Everything here serialises to JSON, because every decision is stored as a trace that a
reviewer or judge can inspect: what was read, where on the photo, which independent sources
agreed, which rules fired, and why the packet went where it went.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Band(StrEnum):
    ACCEPT = "accept"  # مطابق
    CHECK = "check"  # تأكد
    REVIEW = "review"  # راجع


class Outcome(StrEnum):
    PRE_CHECKED = "PRE_CHECKED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    AWAITING_RETAKE = "AWAITING_RETAKE"


class Source(StrEnum):
    """Independent evidence that agrees with the VLM's reading of a field."""

    OCR = "ocr"  # the OCR engine read the same value, unambiguously
    MRZ = "mrz"  # the machine-readable zone (check digits valid) carries the same value
    CROSS_DOC = "cross_doc"  # another document carries the same value
    REREAD = "reread"  # a zoomed re-read of the field's crop returned the same value


class Evidence(BaseModel):
    ocr_matches: int = 0
    ocr_similarity: float = 0.0
    ocr_score: float | None = None
    box: tuple[float, float, float, float] | None = None  # pixels on the read image
    lp_min: float | None = None
    lp_mean: float | None = None
    lp_tokens: int = 0


class FieldRecord(BaseModel):
    doc_type: str
    field: str
    raw: str | None  # exactly what the VLM returned
    value: str | None  # canonical value used by rules (None = treated as unreadable)
    critical: bool = False
    required: bool = False
    evidence: Evidence = Field(default_factory=Evidence)
    sources: list[Source] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)  # stable reason codes
    blocking: bool = False  # a reason that forces review regardless of confidence
    p_correct: float | None = None
    band: Band | None = None

    def add_reason(self, code: str, *, blocking: bool = False) -> None:
        if code not in self.reasons:
            self.reasons.append(code)
        self.blocking = self.blocking or blocking

    def add_source(self, source: Source) -> None:
        if source not in self.sources:
            self.sources.append(source)


class DocumentRead(BaseModel):
    doc_type: str
    image_id: str
    quality: dict[str, Any] = Field(default_factory=dict)
    fields: dict[str, FieldRecord] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    mrz: dict[str, Any] | None = None  # parsed MRZ when the document has one
    ocr_lines: list[dict[str, Any]] = Field(default_factory=list)  # text, score, box (read image px)
    timings: dict[str, float] = Field(default_factory=dict)


class CrossCheck(BaseModel):
    rule: str
    members: list[tuple[str, str]]  # (doc_type, field)
    result: str  # SAME | VARIANT | SOUND_ALIKE | PARTIAL | CONFLICT | MISSING
    reasons: list[str] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)


class PacketResult(BaseModel):
    packet_id: str
    outcome: Outcome
    documents: dict[str, DocumentRead]
    cross_checks: list[CrossCheck] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    retake: dict[str, str] = Field(default_factory=dict)  # doc_type -> reason code
    fingerprint: dict[str, Any] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
