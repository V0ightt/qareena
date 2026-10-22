"""Run one packet end to end: quality -> read twice -> ground -> rules -> score -> route.

Readers are injected behind small protocols so the pipeline can be tested without a GPU.
The OCR reader runs on a CPU thread while the VLM runs on the GPU. Any reader failure
degrades the packet to NEEDS_REVIEW; nothing here can crash a packet into PRE_CHECKED.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from qareena.checks.injection import suspicious_lines
from qareena.config import Runtime
from qareena.pipeline import router, rules
from qareena.pipeline.grounding import ground
from qareena.pipeline.logprobs import field_logprob
from qareena.pipeline.quality import assess
from qareena.pipeline.records import DocumentRead, FieldRecord, PacketResult
from qareena.pipeline.scoring import PROVISIONAL, LogisticScorer, assign_band, load_certificate
from qareena.readers.ocr import OcrResult
from qareena.readers.ollama import OllamaClient, OllamaError, TokenLogprob
from qareena.readers.prompts import SYSTEM, field_schema, user_prompt

log = logging.getLogger(__name__)
WRONG_DOC_NULL_SHARE = 0.7  # this share of required fields empty on a good photo -> wrong document


@dataclass(frozen=True)
class VlmRead:
    data: dict[str, Any] | None
    content: str
    tokens: tuple[TokenLogprob, ...]
    metrics: dict[str, float]
    error: str | None = None


class VlmReader(Protocol):
    def read(self, doc_type: str, png: bytes) -> VlmRead: ...


class OcrReader(Protocol):
    def read(self, image: np.ndarray) -> OcrResult: ...


class OllamaVlmReader:
    def __init__(self, runtime: Runtime, client: OllamaClient) -> None:
        self._rt, self._client = runtime, client

    def read(self, doc_type: str, png: bytes) -> VlmRead:
        doc = self._rt.catalog.documents[doc_type]
        try:
            r = self._client.chat_json(
                model=self._rt.settings.vlm_model,
                system=SYSTEM,
                user=user_prompt(doc),
                schema=field_schema(doc),
                images=[png],
            )
        except OllamaError as exc:
            return VlmRead(None, "", (), {}, f"READER_FAILED: {exc}")
        return VlmRead(
            r.data, r.content, r.logprobs, r.metrics, None if r.ok else f"INVALID_JSON: {r.parse_error}"
        )


def encode_png(img: np.ndarray, long_side: int = 1600) -> bytes:
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("could not encode image")
    return buf.tobytes()


class Pipeline:
    def __init__(
        self,
        runtime: Runtime,
        vlm: VlmReader,
        ocr: OcrReader,
        *,
        today: dt.date | None = None,
        scorer: LogisticScorer | None = None,
    ) -> None:
        self.rt, self.vlm, self.ocr = runtime, vlm, ocr
        self.today = today or dt.date.today()
        self.cert = load_certificate(runtime)
        self.scorer = scorer or (
            LogisticScorer(self.cert.weights, certified=True) if self.cert else PROVISIONAL
        )
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    # --- one document -------------------------------------------------------------------

    def read_document(
        self, doc_type: str, img: np.ndarray, image_id: str, *, force: bool = False
    ) -> DocumentRead:
        t0 = time.perf_counter()
        q = assess(img)
        doc = DocumentRead(
            doc_type=doc_type,
            image_id=image_id,
            quality={"ok": q.ok, "reason": q.reason, "metrics": q.metrics, "notes": list(q.notes)},
        )
        doc.timings["quality_s"] = round(time.perf_counter() - t0, 3)
        if not q.ok and q.reason in router.RETAKE_REASONS and not force:
            doc.reasons.append(q.reason)  # cheapest fix first: no model time spent
            return doc
        if not q.ok:
            doc.reasons.append(f"QUALITY_{q.reason}")  # retakes exhausted: read best-effort, flag it
        region = q.rectified if q.rectified is not None else img
        doc.quality["read_size"] = [int(region.shape[1]), int(region.shape[0])]
        fut = self._pool.submit(self.ocr.read, region)
        vlm = self.vlm.read(doc_type, encode_png(region))
        ocr = fut.result()
        doc.timings.update(
            {
                "vlm_s": round(vlm.metrics.get("wall_s", 0.0), 3),
                "ocr_s": round(ocr.seconds, 3),
                "total_s": round(time.perf_counter() - t0, 3),
            }
        )
        doc.ocr_lines = [
            {"text": ln.text, "score": round(ln.score, 3), "box": [round(v, 1) for v in ln.bbox]}
            for ln in ocr.lines
        ]
        if suspicious_lines([ln.text for ln in ocr.lines]):
            doc.reasons.append("SUSPICIOUS_TEXT")
        spec_doc = self.rt.catalog.documents[doc_type]
        if vlm.error:
            doc.reasons.append(vlm.error.split(":", 1)[0])
        data = vlm.data or {}
        for name, spec in spec_doc.fields.items():
            raw = data.get(name)
            raw = None if raw is None else str(raw)
            rec = rules.make_record(doc_type, name, raw, self.rt.catalog)
            if raw:
                g = ground(raw, spec.kind, ocr.lines)
                rec.evidence.ocr_matches, rec.evidence.ocr_similarity = g.matches, round(g.similarity, 4)
                rec.evidence.ocr_score, rec.evidence.box = g.ocr_score, g.box
                lp = field_logprob(vlm.content, vlm.tokens, name, raw)
                if lp is not None:
                    rec.evidence.lp_min, rec.evidence.lp_mean, rec.evidence.lp_tokens = (
                        lp.min_lp,
                        lp.mean_lp,
                        lp.n_tokens,
                    )
            doc.fields[name] = rec
        required = [n for n, s in spec_doc.fields.items() if s.required]
        empty = sum(doc.fields[n].raw is None for n in required)
        if vlm.error is None and required and empty / len(required) >= WRONG_DOC_NULL_SHARE and q.ok:
            doc.reasons.append("WRONG_DOCUMENT")
        return doc

    # --- the packet ---------------------------------------------------------------------

    def run(
        self,
        packet_id: str,
        images: Mapping[str, np.ndarray],
        *,
        retakes_used: Mapping[str, int] | None = None,
        kill_switch: bool = False,
    ) -> PacketResult:
        t0 = time.perf_counter()
        retakes_used = retakes_used or {}
        limit = self.rt.policy.packet.retakes_per_document
        reads: dict[str, DocumentRead] = {}
        for doc_type, img in images.items():
            if doc_type not in self.rt.catalog.documents:
                continue
            force = retakes_used.get(doc_type, 0) >= limit
            reads[doc_type] = self.read_document(doc_type, img, f"{packet_id}/{doc_type}", force=force)
        for doc in reads.values():
            if doc.fields:
                rules.apply_field_rules(doc, self.rt.catalog)
        rules.national_card_rules(reads.get(rules.FRONT), reads.get(rules.BACK), self.today)
        for k in (rules.CHAMBER, rules.TAX):
            rules.validity_rules(reads.get(k), self.today)
        checks = rules.cross_document({k: v for k, v in reads.items() if v.fields}, self.rt.policy)
        for doc in reads.values():
            spec_doc = self.rt.catalog.documents[doc.doc_type]
            for name, rec in doc.fields.items():
                assign_band(rec, spec_doc.fields[name].kind, self.scorer, self.cert)
        degraded = any(
            r.startswith(("READER_FAILED", "INVALID_JSON")) for d in reads.values() for r in d.reasons
        )
        outcome, reasons, retake = router.route(
            self.rt.policy.packet.required_documents,
            reads,
            checks,
            certified=self.cert is not None,
            retakes_used=retakes_used,
            retake_limit=limit,
            degraded=degraded,
            kill_switch=kill_switch,
        )
        return PacketResult(
            packet_id=packet_id,
            outcome=outcome,
            documents=reads,
            cross_checks=checks,
            reasons=reasons,
            retake=retake,
            fingerprint=self.rt.fingerprint,
            timings={"total_s": round(time.perf_counter() - t0, 3)},
        )


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"not an image: {path.name}")
    return img


def open_fields(result: PacketResult) -> list[FieldRecord]:
    return [
        r
        for d in result.documents.values()
        for r in d.fields.values()
        if r.band is not None and r.band.value != "accept"
    ]
