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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from qareena.checks.fields import garbled, label_echo
from qareena.checks.injection import suspicious_lines
from qareena.config import Runtime
from qareena.pipeline import router, rules
from qareena.pipeline.grounding import LabelMap, ground_field, map_labels, reads_reversed
from qareena.pipeline.logprobs import field_logprob
from qareena.pipeline.loop import Budget, crop_near_label, run_loop
from qareena.pipeline.quality import assess
from qareena.pipeline.records import DocumentRead, FieldRecord, PacketResult
from qareena.pipeline.scoring import PROVISIONAL, LogisticScorer, assign_band, load_certificate
from qareena.readers.ocr import OcrLine, OcrResult
from qareena.readers.ollama import OllamaClient, OllamaError, TokenLogprob
from qareena.readers.prompts import SYSTEM, VALUE_SCHEMA, crop_prompt, field_schema, json_keys, user_prompt

log = logging.getLogger(__name__)
WRONG_DOC_NULL_SHARE = 0.7  # this share of required fields empty on a good photo -> wrong document
RETRY_ON = ("READER_LOOPED", "INVALID_JSON")  # model failures on this image: one retry at another size
PROBE_SIDE = 960  # orientation probe: long side of the image the OCR reads
PROBE_MIN_SCORE = 0.75  # only confident text counts towards an orientation
PROBE_MARGIN = 1.5  # an upside-down reading must be clearly better before we turn the image
PROBE_UPRIGHT = 150.0  # this much confident text the right way up: upright, skip the other probe
# (dev renders: upright min 73 / median 225; upside down max 75 / median 20)
LOOKS_LIKE_MIN = 3  # distinctive labels of another document type needed to call it the wrong one
LOOKS_LIKE_MARGIN = 3  # ...and this many more than the expected type's labels
RECOVER_MAX = 4  # crop re-reads per document for required fields answered with a label or glyph soup


@dataclass(frozen=True)
class VlmRead:
    data: dict[str, Any] | None
    content: str
    tokens: tuple[TokenLogprob, ...]
    metrics: dict[str, float]
    error: str | None = None
    keys: dict[str, str] = field(default_factory=dict)  # field -> JSON key in ``content`` (for logprobs)


class VlmReader(Protocol):
    def read(self, doc_type: str, png: bytes) -> VlmRead: ...

    def read_field(self, doc_type: str, field: str, png: bytes) -> str | None: ...


class OcrReader(Protocol):
    def read(self, image: np.ndarray) -> OcrResult: ...


class OllamaVlmReader:
    def __init__(self, runtime: Runtime, client: OllamaClient) -> None:
        self._rt, self._client = runtime, client

    def read(self, doc_type: str, png: bytes) -> VlmRead:
        doc = self._rt.catalog.documents[doc_type]
        keys = json_keys(doc)  # the printed Arabic labels are the JSON keys
        try:
            r = self._client.chat_json(
                model=self._rt.settings.vlm_model,
                system=SYSTEM,
                user=user_prompt(doc),
                schema=field_schema(doc),
                images=[png],
            )
        except OllamaError as exc:
            # A generation stopped for repeating itself is the model failing on this image,
            # not the server failing: it is retried smaller, and never marks the system degraded.
            code = "READER_LOOPED" if "repeat limit" in str(exc) else "READER_FAILED"
            return VlmRead(None, "", (), {}, f"{code}: {exc}")
        data = None if r.data is None else {name: r.data.get(key) for key, name in keys.items()}
        return VlmRead(
            data,
            r.content,
            r.logprobs,
            r.metrics,
            None if r.ok else f"INVALID_JSON: {r.parse_error}",
            {name: key for key, name in keys.items()},
        )

    def read_field(self, doc_type: str, field: str, png: bytes) -> str | None:
        """Zoomed re-read of one field from a crop; None if unreadable or on any failure."""
        doc = self._rt.catalog.documents[doc_type]
        try:
            r = self._client.chat_json(
                model=self._rt.settings.vlm_model,
                system=SYSTEM,
                user=crop_prompt(doc, field),
                schema=VALUE_SCHEMA,
                images=[png],
                logprobs=False,
                num_predict=128,
            )
        except OllamaError:
            return None
        value = (r.data or {}).get("value")
        return None if value is None else str(value)


def _rotate(img: np.ndarray, degrees: int) -> np.ndarray:
    """Clockwise rotation by a multiple of 90 degrees."""
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(img, codes[degrees]) if degrees in codes else img


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
        self._regions: dict[str, np.ndarray] = {}

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def read_images(self) -> dict[str, np.ndarray]:
        """What the readers saw in the last ``run`` (flattened documents). Evidence boxes are in
        these coordinates, so the reviewer is shown these images."""
        return dict(self._regions)

    # --- one document -------------------------------------------------------------------

    def read_document(
        self, doc_type: str, img: np.ndarray, image_id: str, *, force: bool = False
    ) -> DocumentRead:
        t0 = time.perf_counter()
        q = assess(img)
        doc = DocumentRead(
            doc_type=doc_type,
            image_id=image_id,
            quality={
                "ok": q.ok,
                "reason": q.reason,
                "metrics": q.metrics,
                "notes": list(q.notes),
                # the document outline in the photo; evidence boxes are on the rectified read image
                "quad": None if q.quad is None else [[round(float(v), 1) for v in p] for p in q.quad],
            },
        )
        doc.timings["quality_s"] = round(time.perf_counter() - t0, 3)
        if not q.ok and q.reason in router.RETAKE_REASONS and not force:
            doc.reasons.append(q.reason)  # cheapest fix first: no model time spent
            return doc
        if not q.ok:
            doc.reasons.append(f"QUALITY_{q.reason}")  # retakes exhausted: read best-effort, flag it
        region = q.rectified if q.rectified is not None else img
        region, turned = self._upright(region)
        if turned:
            doc.quality["rotated"] = turned
        self._regions[doc_type] = region
        doc.quality["read_size"] = [int(region.shape[1]), int(region.shape[0])]
        fut = self._pool.submit(self.ocr.read, region)
        reader = self.rt.policy.reader
        vlm = self.vlm.read(doc_type, encode_png(region, reader.long_side_for(doc_type)))
        if vlm.error and vlm.error.startswith(RETRY_ON):
            doc.reasons.append(f"NOTE_RETRIED_RESIZED:{vlm.error.split(':', 1)[0]}")
            retry = self.vlm.read(doc_type, encode_png(region, reader.retry_for(doc_type)))
            if retry.error is None or retry.error.startswith("READER_FAILED"):
                vlm = retry  # read, or the server itself failed (that one does degrade the system)
            else:
                vlm = VlmRead(None, "", (), retry.metrics, f"READ_FAILED: {retry.error}")
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
        data = {k: None if v is None else str(v) for k, v in (vlm.data or {}).items()}
        labels = map_labels(spec_doc, ocr.lines)
        recovered, backwards = self._recover(doc, data, region, ocr.lines, labels)
        for name, spec in spec_doc.fields.items():
            raw = data.get(name)
            rec = rules.make_record(doc_type, name, raw, self.rt.catalog)
            ev = rec.evidence
            ev.label_box = labels.label_box.get(name)
            claimed = labels.claims.get(name, ())
            ev.ocr_text = " ".join(ocr.lines[j].text for j in claimed) if claimed else None
            if name in recovered:
                ev.first_raw = recovered[name]
                rec.add_reason("NOTE_RECOVERED_FROM_CROP")
            if name in backwards:
                rec.add_reason("READ_REVERSED", blocking=True)  # Arabic read left to right
            if raw:
                g = ground_field(name, raw, spec.kind, ocr.lines, labels)
                ev.ocr_matches, ev.ocr_similarity = g.matches, round(g.similarity, 4)
                ev.ocr_score, ev.box = g.ocr_score, g.box
                ev.ocr_anchored, ev.ocr_foreign = g.anchored, g.foreign
                lp = (
                    None
                    if name in recovered
                    else field_logprob(vlm.content, vlm.tokens, vlm.keys.get(name, name), raw)
                )
                if lp is not None:
                    ev.lp_min, ev.lp_mean, ev.lp_tokens = lp.min_lp, lp.mean_lp, lp.n_tokens
            doc.fields[name] = rec
        required = [n for n, s in spec_doc.fields.items() if s.required]
        empty = sum(doc.fields[n].raw is None for n in required)
        looks_like = self._looks_like(doc_type, ocr.lines, labels)
        if looks_like is not None:
            doc.reasons.append(f"WRONG_DOCUMENT:{looks_like}")  # printed labels of another document
        elif vlm.error is None and required and empty / len(required) >= WRONG_DOC_NULL_SHARE and q.ok:
            doc.reasons.append("WRONG_DOCUMENT")
        return doc

    def _looks_like(self, doc_type: str, lines: tuple[OcrLine, ...], labels: LabelMap) -> str | None:
        """Another document type, if clearly more of *its* printed labels are on this image.

        Only labels distinctive of each type count (shared ones such as «تاريخ الإصدار» say
        nothing). An unfamiliar layout of the right document finds few labels of any type, so
        it is never sent back by this rule.
        """
        catalog = self.rt.catalog
        mine = {spec.label_ar for spec in catalog.documents[doc_type].fields.values()}
        found_mine = len(labels.label_box)
        best, best_n = None, 0
        for other, spec_doc in catalog.documents.items():
            if other == doc_type:
                continue
            found = map_labels(spec_doc, lines).label_box
            n = sum(1 for f in found if spec_doc.fields[f].label_ar not in mine)
            if n > best_n:
                best, best_n = other, n
        if best is not None and best_n >= LOOKS_LIKE_MIN and best_n >= found_mine + LOOKS_LIKE_MARGIN:
            return best
        return None

    def _upright(self, region: np.ndarray) -> tuple[np.ndarray, int]:
        """Turn a sideways or upside-down document upright before anyone reads it.

        A small OCR probe reads the image in the two orientations that its shape allows
        (0/180 for a landscape image, 90/270 for a portrait one; our documents are all
        landscape). Text read upside down or sideways comes out short and unsure, so the
        orientation with clearly more confident text wins; a tie keeps the image as it is.
        Returns (image, degrees turned clockwise).
        """
        h, w = region.shape[:2]
        angles = (90, 270) if h > 1.15 * w else (0, 180)
        scale = min(1.0, PROBE_SIDE / max(h, w))
        small = (
            cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1
            else region
        )
        scores = {}
        for angle in angles:
            lines = self.ocr.read(_rotate(small, angle)).lines
            scores[angle] = sum(len(ln.text) * ln.score for ln in lines if ln.score >= PROBE_MIN_SCORE)
            if angle == 0 and scores[0] >= PROBE_UPRIGHT:
                return region, 0  # clearly the right way up: most documents stop here
        first, second = angles
        if angles == (0, 180):
            best = 180 if scores[180] > PROBE_MARGIN * scores[0] + 1 else 0
        else:
            best = first if scores[first] >= scores[second] else second
        return (_rotate(region, best), best) if best else (region, 0)

    def _recover(
        self,
        doc: DocumentRead,
        data: dict[str, str | None],
        region: np.ndarray,
        lines: tuple[OcrLine, ...],
        labels: LabelMap,
    ) -> tuple[dict[str, str], set[str]]:
        """Re-read, from a crop at its label, a field whose first answer is unusable.

        Unusable means a printed label (some layouts get their Kurdish label returned as the
        value), spelled-out glyphs, or Arabic read backwards (the OCR at the label matches the
        answer reversed). That answer is never kept: the field is read once more from a close-up
        crop around its printed label, and the new value goes through every rule like any other
        reading (it has no logprobs, so it scores lower). At most ``RECOVER_MAX`` per document.

        Returns (field -> the rejected first answer, fields still read backwards).
        """
        t0 = time.perf_counter()
        printed = self.rt.catalog.printed_labels()
        spec_doc = self.rt.catalog.documents[doc.doc_type]
        out: dict[str, str] = {}
        backwards: set[str] = set()

        def unusable(name: str, value: str) -> str | None:
            if label_echo(value, printed) is not None or garbled(value):
                return "label"
            if reads_reversed(name, value, spec_doc.fields[name].kind, lines, labels):
                return "reversed"
            return None

        for name, spec in spec_doc.fields.items():
            raw = data.get(name)
            if raw is None or (problem := unusable(name, raw)) is None:
                continue
            if problem == "reversed":
                backwards.add(name)
            elif not (spec.required or spec.critical):
                continue  # an optional field may simply stay unread
            if len(out) >= RECOVER_MAX or name not in labels.label_box:
                continue
            crop = crop_near_label(region, labels.label_box[name])
            if crop.size == 0:
                continue
            second = self.vlm.read_field(doc.doc_type, name, encode_png(crop, 800))
            if second is None or unusable(name, second) is not None:
                continue
            out[name], data[name] = raw, second
            backwards.discard(name)
        if out:
            doc.timings["recover_s"] = round(time.perf_counter() - t0, 3)
        return out, backwards

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
        self._regions = {}  # never reuse a previous packet's images
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
        actions: list[dict] = []
        if self.rt.settings.features.evidence_loop:
            loop_cfg = self.rt.policy.evidence_loop
            budget = Budget(loop_cfg.max_actions_per_field, t0 + loop_cfg.packet_budget_seconds)
            for doc_type, doc in reads.items():
                region = self._regions.get(doc_type)
                if region is None or not doc.fields:
                    continue
                actions += run_loop(
                    doc,
                    region,
                    self.rt.catalog,
                    budget=budget,
                    reread=lambda d, f, crop: self.vlm.read_field(d, f, encode_png(crop, 800)),
                    reocr=lambda big: self.ocr.read(big).lines,
                )
        for doc in reads.values():
            spec_doc = self.rt.catalog.documents[doc.doc_type]
            for name, rec in doc.fields.items():
                assign_band(rec, spec_doc.fields[name].kind, self.scorer, self.cert)
        degraded = any(r.startswith("READER_FAILED") for d in reads.values() for r in d.reasons)
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
            actions=actions,
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
