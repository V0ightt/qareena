"""Packet routing: a fixed, readable policy. The cheapest fix comes first.

1. A document missing, unreadable for a photo reason, or of the wrong type, with retakes
   left  -> AWAITING_RETAKE (one Iraqi-Arabic instruction per document).
2. Anything open (a field in check/review, a cross-document problem, an uncertified policy,
   a degraded system or the kill switch) -> NEEDS_REVIEW.
3. Otherwise -> PRE_CHECKED: a reviewer still approves with one click.
"""

from __future__ import annotations

from collections.abc import Mapping

from qareena.pipeline.records import Band, CrossCheck, DocumentRead, Outcome

RETAKE_REASONS = frozenset(
    {
        "BLUR",
        "GLARE",
        "FINGER",
        "CUT_CORNER",
        "TOO_FAR",
        "TOO_DARK",
        "TOO_BRIGHT",
        "TILTED",
        "TOO_SMALL",
        "WRONG_DOCUMENT",
        "DOC_MISSING",
        "NOT_A_DOCUMENT",
    }
)
REVIEW_REASONS = frozenset({"SUSPICIOUS_TEXT", "UNSUPPORTED_DOCUMENT", "READ_FAILED"})


def route(
    required: list[str] | tuple[str, ...],
    reads: Mapping[str, DocumentRead],
    checks: list[CrossCheck],
    *,
    certified: bool,
    retakes_used: Mapping[str, int],
    retake_limit: int,
    degraded: bool = False,
    kill_switch: bool = False,
) -> tuple[Outcome, list[str], dict[str, str]]:
    reasons: list[str] = []
    retake: dict[str, str] = {}
    for doc_type in required:
        doc = reads.get(doc_type)
        # "QUALITY_<X>" marks a photo read best-effort after retakes ran out: still a quality failure
        reason = (
            "DOC_MISSING"
            if doc is None
            else next(
                (
                    r.removeprefix("QUALITY_")
                    for r in doc.reasons
                    if r.split(":", 1)[0] in RETAKE_REASONS or r.startswith("QUALITY_")
                ),
                None,
            )
        )
        if reason is not None:
            if retakes_used.get(doc_type, 0) < retake_limit:
                retake[doc_type] = reason
            else:
                reasons.append(f"RETAKES_EXHAUSTED:{doc_type}:{reason}")

    for doc in reads.values():
        for r in doc.reasons:
            if r in REVIEW_REASONS and r not in reasons:
                reasons.append(r)
    if kill_switch:
        reasons.append("KILL_SWITCH")
    if degraded:
        reasons.append("SYSTEM_DEGRADED")
    if retake and not kill_switch:
        return Outcome.AWAITING_RETAKE, reasons, retake

    open_fields = [
        f"{d}.{f}"
        for d, doc in reads.items()
        for f, r in doc.fields.items()
        if r.band in (Band.CHECK, Band.REVIEW)
    ]
    if open_fields:
        reasons.append(f"OPEN_FIELDS:{len(open_fields)}")
    if any(c.result in ("CONFLICT", "MISSING", "SOUND_ALIKE") for c in checks):
        reasons.append("CROSS_CHECK_OPEN")
    if not certified:
        reasons.append("POLICY_NOT_CERTIFIED")
    if reasons:
        return Outcome.NEEDS_REVIEW, reasons, retake
    return Outcome.PRE_CHECKED, reasons, retake
