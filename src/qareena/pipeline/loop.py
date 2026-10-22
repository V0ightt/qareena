"""The evidence loop: the one agentic step (ADR-003).

For each uncertain field, a written policy picks the next evidence action, within a budget.
Both actions work on a crop around the field's printed label (found by the OCR reader,
because labels are printed reliably):

1. ``reocr``   — the OCR engine reads the crop, upscaled; a verbatim match adds an OCR source.
                 Independent evidence, and cheap (CPU), so it goes first.
2. ``reread``  — the VLM reads the crop for that value only. The *same* model agreeing with
                 itself is not independent evidence (errors repeat), so it only adds a REREAD
                 signal for the scorer; disagreement flags the field for review.

It never changes a value on its own, never accepts anything, and stops when the field has an
independent source, when its actions are spent, or when the packet's time budget runs out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from qareena.config import Catalog, FieldKind, FieldSpec
from qareena.eval.compare import canonical
from qareena.pipeline.grounding import ground
from qareena.pipeline.records import DocumentRead, FieldRecord, Source
from qareena.pipeline.scoring import INDEPENDENT
from qareena.readers.ocr import OcrLine
from qareena.text import arabic

UPSCALE = 2.0


@dataclass
class Budget:
    per_field: int
    deadline: float  # time.perf_counter() value after which no new action starts
    used: dict[str, int] = field(default_factory=dict)

    def allow(self, key: str) -> bool:
        return self.used.get(key, 0) < self.per_field and time.perf_counter() < self.deadline

    def spend(self, key: str) -> None:
        self.used[key] = self.used.get(key, 0) + 1


def independent_sources(rec: FieldRecord) -> list[Source]:
    return [s for s in rec.sources if s in INDEPENDENT]


def needs_evidence(rec: FieldRecord) -> bool:
    """Uncertain = a critical field with a value, not blocked, and no independent source yet."""
    return rec.critical and rec.value is not None and not rec.blocking and not independent_sources(rec)


def label_box(spec: FieldSpec, lines: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    """Box of the OCR line carrying the field's printed label (Arabic label or an alias)."""
    targets = [arabic.spelling(lab).replace(" ", "") for lab in (spec.label_ar, *spec.aliases)]
    best, best_len = None, 0
    for ln in lines:
        text = arabic.spelling(ln["text"]).replace(" ", "")
        for t in targets:
            if t and t in text and len(t) > best_len:
                best, best_len = tuple(ln["box"]), len(t)
    return best


def crop_near_label(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Region around a label that contains its value, whether printed below it or beside it (RTL)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    lh, lw = max(y1 - y0, 8.0), max(x1 - x0, 8.0)
    cx0 = max(0, int(x0 - 5 * lw))  # values sit below or to the left of an RTL label
    cx1 = min(w, int(x1 + 0.3 * lw))
    cy0 = max(0, int(y0 - 0.6 * lh))
    cy1 = min(h, int(y1 + 2.8 * lh))
    return img[cy0:cy1, cx0:cx1].copy()


def run_loop(
    doc: DocumentRead, img: np.ndarray, catalog: Catalog, *, reread, reocr, budget: Budget
) -> list[dict]:
    """Apply the policy to every uncertain field of one document. Returns the action log."""
    actions: list[dict] = []
    spec_doc = catalog.documents[doc.doc_type]
    for name, rec in doc.fields.items():
        spec = spec_doc.fields[name]
        key = f"{doc.doc_type}.{name}"
        if not needs_evidence(rec):
            continue
        box = label_box(spec, doc.ocr_lines)
        if box is None:
            rec.add_reason("LABEL_NOT_FOUND")
            continue
        crop = crop_near_label(img, box)
        if crop.size == 0:
            continue
        if budget.allow(key):  # 1) OCR on the upscaled crop: independent and cheap
            budget.spend(key)
            t0 = time.perf_counter()
            big = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
            lines: list[OcrLine] = list(reocr(big))
            g = ground(rec.raw, spec.kind if spec.kind != FieldKind.FULL_NAME else FieldKind.TEXT, lines)
            if g.exact:
                rec.add_source(Source.OCR)
                if "NOT_GROUNDED" in rec.reasons:
                    rec.reasons.remove("NOT_GROUNDED")
            actions.append(
                {
                    "field": key,
                    "action": "reocr",
                    "matched": g.exact,
                    "seconds": round(time.perf_counter() - t0, 3),
                }
            )
        if not independent_sources(rec) and budget.allow(key):  # 2) zoomed VLM re-read
            budget.spend(key)
            t0 = time.perf_counter()
            second = reread(doc.doc_type, name, crop)
            value2 = canonical(second, spec) if second is not None else None
            agree = value2 is not None and value2 == rec.value
            if agree:
                rec.add_source(Source.REREAD)
            elif second is not None:
                rec.add_reason("REREAD_DISAGREES")
            actions.append(
                {
                    "field": key,
                    "action": "reread",
                    "result": second,
                    "agree": agree,
                    "seconds": round(time.perf_counter() - t0, 3),
                }
            )
    return actions
