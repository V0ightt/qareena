"""Field confidence (P(correct)) and bands, behind the certification gate.

A ``Scorer`` maps a field's evidence to P(correct). The calibrated scorer is a logistic
regression fitted on the *development* split; its thresholds are certified on the
*calibration* split (ADR-005) and stored in a certificate bound to the policy hash and model.
Until such a certificate exists, a provisional scorer is used for display only: nothing can
be accepted, so nothing can be pre-checked (the certification gate).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from qareena.config import FieldKind, Runtime
from qareena.pipeline.records import Band, FieldRecord, Source

# Evidence that does not come from the VLM itself. A same-model re-read is a signal, not a source.
INDEPENDENT = frozenset({Source.OCR, Source.MRZ, Source.CROSS_DOC})

FEATURES = (
    "bias",
    "lp_min",
    "lp_mean",
    "src_ocr",  # anchored at the field's own label
    "ocr_unanchored",
    "ocr_ambiguous",
    "ocr_similarity",
    "ocr_score",
    "src_mrz",
    "src_cross_doc",
    "src_reread",
    "n_caution",
    "recovered",
    "kind_name",
    "kind_number",
    "kind_date",
    "kind_text",
)
NAME_KINDS = {FieldKind.NAME_PART, FieldKind.FULL_NAME}
NUMBER_KINDS = {FieldKind.DIGITS, FieldKind.CODE, FieldKind.MRZ_LINE}


def features(rec: FieldRecord, kind: FieldKind) -> list[float]:
    ev = rec.evidence
    caution = [r for r in rec.reasons if not r.startswith("NOTE_")]
    return [
        1.0,
        max(ev.lp_min, -20.0) if ev.lp_min is not None else -5.0,
        max(ev.lp_mean, -20.0) if ev.lp_mean is not None else -5.0,
        float(Source.OCR in rec.sources),
        float(Source.OCR_UNANCHORED in rec.sources),
        float(ev.ocr_matches > 1),
        ev.ocr_similarity,
        ev.ocr_score or 0.0,
        float(Source.MRZ in rec.sources),
        float(Source.CROSS_DOC in rec.sources),
        float(Source.REREAD in rec.sources),
        float(len(caution)),
        float("NOTE_RECOVERED_FROM_CROP" in rec.reasons),
        float(kind in NAME_KINDS),
        float(kind in NUMBER_KINDS),
        float(kind == FieldKind.DATE),
        float(kind in (FieldKind.TEXT, FieldKind.ENUM)),
    ]


class Scorer(Protocol):
    certified: bool

    def p_correct(self, x: list[float]) -> float: ...


@dataclass(frozen=True)
class LogisticScorer:
    weights: tuple[float, ...]
    certified: bool = False

    def p_correct(self, x: list[float]) -> float:
        z = sum(w * v for w, v in zip(self.weights, x, strict=True))
        return 1.0 / (1.0 + math.exp(-max(min(z, 40.0), -40.0)))


# Hand-set, uncertified weights: plausible ordering for display while no certificate exists.
PROVISIONAL = LogisticScorer(
    weights=(-1.0, 0.6, 0.4, 2.0, 0.8, -0.8, 1.5, 1.0, 2.0, 1.5, 1.2, -1.2, -0.5, -0.3, -0.2, 0.0, 0.0),
    certified=False,
)


@dataclass(frozen=True)
class Certificate:
    """Thresholds certified on held-out calibration data for one policy + model combination."""

    policy_sha256: str
    vlm_model: str
    weights: tuple[float, ...]
    tau_critical: float | None
    tau_other: float | None
    report: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Certificate:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            d["policy_sha256"],
            d["vlm_model"],
            tuple(d["weights"]),
            d.get("tau_critical"),
            d.get("tau_other"),
            d.get("report", {}),
        )

    def matches(self, runtime: Runtime) -> bool:
        return self.policy_sha256 == runtime.policy_sha256 and self.vlm_model == runtime.settings.vlm_model


def certificate_path(runtime: Runtime) -> Path:
    return runtime.settings.data_dir / "certification" / f"{runtime.policy_sha256[:16]}.json"


def load_certificate(runtime: Runtime) -> Certificate | None:
    path = certificate_path(runtime)
    if not path.exists():
        return None
    cert = Certificate.load(path)
    return cert if cert.matches(runtime) else None


def assign_band(rec: FieldRecord, kind: FieldKind, scorer: Scorer, cert: Certificate | None) -> None:
    """Set ``p_correct`` and ``band``. Empty optional fields get no band (shown as blank)."""
    if rec.value is None:
        rec.band = Band.REVIEW if rec.required or rec.blocking else None
        return
    rec.p_correct = round(scorer.p_correct(features(rec, kind)), 4)
    if rec.blocking:
        rec.band = Band.REVIEW
        return
    caution = [r for r in rec.reasons if not r.startswith("NOTE_") and r not in ("OCR_AMBIGUOUS_BOX",)]
    tau = None if cert is None else (cert.tau_critical if rec.critical else cert.tau_other)
    accept = (
        scorer.certified
        and tau is not None
        and rec.p_correct >= tau
        and not caution
        and (not rec.critical or any(src in INDEPENDENT for src in rec.sources))  # ADR-004
    )
    rec.band = Band.ACCEPT if accept else Band.CHECK
