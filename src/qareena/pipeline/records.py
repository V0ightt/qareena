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
    """Evidence that agrees with a field's value, apart from the reading the value came from."""

    OCR = "ocr"  # the OCR engine read the same value at this field's printed label
    OCR_UNANCHORED = "ocr_unanchored"  # ... somewhere unclaimed, but the label was not found: a signal only
    MRZ = "mrz"  # the machine-readable zone (check digits valid) carries the same value
    CROSS_DOC = "cross_doc"  # another document carries the same value
    OCR_CONSENSUS = "ocr_consensus"  # the OCR read this exact name at its label on both business cards
    REREAD = "reread"  # a zoomed re-read of the field's crop returned the same value


class Evidence(BaseModel):
    ocr_matches: int = 0
    ocr_similarity: float = 0.0
    ocr_score: float | None = None
    ocr_anchored: bool = False  # the field's printed label was found and the match sits at it
    ocr_foreign: bool = False  # the value was found only at another field's label (a field shift)
    box: tuple[float, float, float, float] | None = None  # pixels on the read image
    label_box: tuple[float, float, float, float] | None = None  # the field's printed label, if found
    ocr_text: str | None = None  # what the OCR read at this field's label (shown to the reviewer)
    ocr_line_ids: tuple[int, ...] = ()  # ... as indices into the document's ocr_lines
    first_raw: str | None = None  # the VLM's full-page answer, when a crop re-read or the OCR replaced it
    vlm_raw: str | None = None  # the VLM's answer, when the OCR consensus replaced it (ADR-018)
    lp_min: float | None = None
    lp_mean: float | None = None
    lp_tokens: int = 0


class FieldRecord(BaseModel):
    doc_type: str
    field: str
    raw: str | None  # exactly the text read: the VLM's, or the OCR's where its consensus was adopted
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
    actions: list[dict[str, Any]] = Field(default_factory=list)  # evidence-loop log
    fingerprint: dict[str, Any] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
