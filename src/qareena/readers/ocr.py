"""Arabic line OCR on CPU via RapidOCR (PaddleOCR PP-OCRv5 models on ONNX Runtime).

The OCR reader is the second, independent reader: it supplies text lines with polygons and
per-line confidence, which the grounding step uses as evidence boxes. It runs on the CPU so
the GPU stays dedicated to the VLM; both can work in parallel.

RapidOCR >= 3.9 defaults to PP-OCRv6, which supports no Arabic (neither recogniser nor detector
routing), so PP-OCRv5 detection and the Arabic PP-OCRv5 recogniser are selected explicitly.
Models are downloaded on first use; pre-fetch them before an offline demo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OcrLine:
    text: str
    score: float
    polygon: tuple[tuple[float, float], ...]  # pixel coordinates, 4 points

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs, ys = [p[0] for p in self.polygon], [p[1] for p in self.polygon]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass(frozen=True)
class OcrResult:
    lines: tuple[OcrLine, ...]
    seconds: float
    width: int
    height: int


class ArabicOcr:
    def __init__(self, *, detector: str = "mobile", use_cls: bool = False) -> None:
        """``detector``: PP-OCRv5 "mobile" (fast) or "server" (more accurate). PP-OCRv6 cannot be
        used: RapidOCR refuses to route it for Arabic."""
        from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

        if detector not in ("mobile", "server"):
            raise ValueError("detector must be 'mobile' or 'server'")
        self._engine = RapidOCR(
            params={
                "Det.lang_type": LangDet.CH,  # the PP-OCRv5 detector is multilingual despite its name
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Det.model_type": ModelType.MOBILE if detector == "mobile" else ModelType.SERVER,
                "Rec.lang_type": LangRec.ARABIC,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_type": ModelType.MOBILE,
                "Global.use_cls": use_cls,
                "Global.log_level": "warning",
                "EngineConfig.onnxruntime.use_cuda": False,
            }
        )

    def read(self, image: np.ndarray) -> OcrResult:
        h, w = image.shape[:2]
        t0 = time.perf_counter()
        out = self._engine(image)
        secs = time.perf_counter() - t0
        boxes = getattr(out, "boxes", None)
        txts = getattr(out, "txts", None) or ()
        scores = getattr(out, "scores", None) or ()
        lines = []
        if boxes is not None:
            for box, txt, sc in zip(boxes, txts, scores, strict=False):
                poly = tuple((float(x), float(y)) for x, y in np.asarray(box).reshape(-1, 2))
                lines.append(OcrLine(str(txt), float(sc), poly))
        return OcrResult(tuple(lines), secs, w, h)
