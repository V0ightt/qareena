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
from qareena.text import arabic

# Evidence that does not come from the VLM itself. A same-model re-read is a signal, not a source.
INDEPENDENT = frozenset({Source.OCR, Source.MRZ, Source.CROSS_DOC})

KINDS = (
    FieldKind.NAME_PART,
    FieldKind.FULL_NAME,
    FieldKind.DIGITS,
    FieldKind.CODE,
    FieldKind.MRZ_LINE,
    FieldKind.DATE,
    FieldKind.TEXT,
    FieldKind.ENUM,
)
# A verbatim OCR match at the field's own label means different things per kind: near proof
# for a 10-digit number, weaker for a name (the readers can share a misreading of similar glyphs).
EXACT_BY_KIND = (
    FieldKind.NAME_PART,
    FieldKind.FULL_NAME,
    FieldKind.DIGITS,
    FieldKind.CODE,
    FieldKind.DATE,
    FieldKind.TEXT,
    FieldKind.ENUM,
)
FEATURES = (
    "bias",
    "lp_min",
    "lp_mean",
    "lp_missing",
    "src_ocr",  # anchored at the field's own label
    "ocr_unanchored",
    "ocr_ambiguous",
    "ocr_exact",  # anchored, and the value occurs in the OCR line at the spelling layer
    "ocr_verbatim",  # anchored, and the OCR text at the label IS the value, letter for letter
    "ocr_score",
    "src_mrz",
    "src_cross_doc",
    "src_reread",
    "n_independent",
    "n_caution",
    "recovered",
    "value_len",
    *(f"kind_{k.value}" for k in KINDS),
    *(f"ocr_verbatim_x_{k.value}" for k in EXACT_BY_KIND),
    "mrz_x_digits",
    "cross_doc_x_name_part",
)
NAME_KINDS = {FieldKind.NAME_PART, FieldKind.FULL_NAME}
NUMBER_KINDS = {FieldKind.DIGITS, FieldKind.CODE, FieldKind.MRZ_LINE}


def features(rec: FieldRecord, kind: FieldKind) -> list[float]:
    ev = rec.evidence
    caution = [r for r in rec.reasons if not r.startswith("NOTE_")]
    anchored = Source.OCR in rec.sources
    exact = float(anchored and ev.ocr_similarity >= 0.999)
    verbatim = float(anchored and _verbatim(rec.value, ev.ocr_text, kind))
    mrz = float(Source.MRZ in rec.sources)
    cross = float(Source.CROSS_DOC in rec.sources)
    return [
        1.0,
        max(ev.lp_min, -20.0) if ev.lp_min is not None else 0.0,
        max(ev.lp_mean, -20.0) if ev.lp_mean is not None else 0.0,
        float(ev.lp_min is None),
        float(anchored),
        float(Source.OCR_UNANCHORED in rec.sources),
        float(ev.ocr_matches > 1),
        exact,
        verbatim,
        ev.ocr_score or 0.0,
        mrz,
        cross,
        float(Source.REREAD in rec.sources),
        float(len({s for s in rec.sources if s in INDEPENDENT})),
        float(len(caution)),
        float("NOTE_RECOVERED_FROM_CROP" in rec.reasons),
        min(len(rec.value or ""), 40) / 20.0,
        *(float(kind == k) for k in KINDS),
        *(verbatim * float(kind == k) for k in EXACT_BY_KIND),
        mrz * float(kind == FieldKind.DIGITS),
        cross * float(kind == FieldKind.NAME_PART),
    ]


def _verbatim(value: str | None, ocr_text: str | None, kind: FieldKind) -> bool:
    """The OCR text at the label equals the whole value: same letters (ى is not ي, ة is not ه),
    same digits, nothing missing. Agreement only at the spelling layer, or a value that is a
    mere substring of the OCR line («8699» in «88699»), is weaker evidence."""
    if not value or not ocr_text:
        return False
    if kind in (FieldKind.DIGITS, FieldKind.DATE):
        digits = lambda t: "".join(ch for ch in arabic.normalize_digits(t) if ch.isdigit())  # noqa: E731
        return digits(value) == digits(ocr_text)
    return arabic.strict(value).replace(" ", "").upper() == arabic.strict(ocr_text).replace(" ", "").upper()


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
_HAND_SET = {  # plausible ordering for display while no certificate exists; never accepts
    "bias": -1.0,
    "lp_min": 0.3,
    "src_ocr": 1.5,
    "ocr_exact": 1.0,
    "src_mrz": 1.5,
    "src_cross_doc": 1.0,
    "n_caution": -1.5,
    "recovered": -0.5,
}
PROVISIONAL = LogisticScorer(weights=tuple(_HAND_SET.get(f, 0.0) for f in FEATURES), certified=False)


@dataclass(frozen=True)
class Certificate:
    """Thresholds certified on held-out calibration data for one exact configuration: policy,
    model, reading code, this scorer and the feature flags. Change any of them and it no
    longer applies."""

    policy_sha256: str
    vlm_model: str
    weights: tuple[float, ...]
    tau_critical: float | None
    tau_other: float | None
    report: dict = field(default_factory=dict)
    reading_code_sha256: str = ""
    scoring_sha256: str = ""
    feature_flags: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Certificate:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            d["policy_sha256"],
            d["vlm_model"],
            tuple(d["weights"]),
            d.get("tau_critical"),
            d.get("tau_other"),
            d.get("report", {}) | {"features": d.get("features", list(FEATURES))},
            d.get("reading_code_sha256", ""),
            d.get("scoring_sha256", ""),
            d.get("feature_flags", {}),
        )

    def matches(self, runtime: Runtime) -> bool:
        return (
            self.policy_sha256 == runtime.policy_sha256
            and self.vlm_model == runtime.settings.vlm_model
            and len(self.weights) == len(FEATURES)
            and self.report.get("features", list(FEATURES)) == list(FEATURES)
            and self.reading_code_sha256 == runtime.reading_code_sha256
            and self.scoring_sha256 == runtime.scoring_sha256
            and self.feature_flags == runtime.settings.features.model_dump()
        )


def certificate_path(runtime: Runtime) -> Path:
    return runtime.settings.data_dir / "certification" / f"{runtime.policy_sha256[:16]}.json"


def load_certificate(runtime: Runtime) -> Certificate | None:
    path = certificate_path(runtime)
    if not path.exists():
        return None
    cert = Certificate.load(path)
    return cert if cert.matches(runtime) else None


def eligible(rec: FieldRecord) -> bool:
    """May this field be accepted at all, whatever its score? The one rule shared by the band
    assignment and threshold certification, so what is certified is exactly what is accepted."""
    if rec.value is None or rec.blocking:
        return False
    caution = [r for r in rec.reasons if not r.startswith("NOTE_") and r not in ("OCR_AMBIGUOUS_BOX",)]
    return not caution and (not rec.critical or any(src in INDEPENDENT for src in rec.sources))  # ADR-004


def assign_band(rec: FieldRecord, kind: FieldKind, scorer: Scorer, cert: Certificate | None) -> None:
    """Set ``p_correct`` and ``band``. Empty optional fields get no band (shown as blank)."""
    if rec.value is None:
        rec.band = Band.REVIEW if rec.required or rec.blocking else None
        return
    rec.p_correct = round(scorer.p_correct(features(rec, kind)), 4)
    if rec.blocking:
        rec.band = Band.REVIEW
        return
    tau = None if cert is None else (cert.tau_critical if rec.critical else cert.tau_other)
    accept = scorer.certified and tau is not None and rec.p_correct >= tau and eligible(rec)
    rec.band = Band.ACCEPT if accept else Band.CHECK
