"""Metrics from a run's traces against the split's labels-by-construction (DESIGN §9.2).

Every field becomes one ``FieldRow``. The metrics are small aggregations over the rows, with
n and an interval on every rate, so the same run also serves the ablations and the fit:

- reading quality on legible fields, false fills on unreadable ones, false nulls;
- error among **accepted** fields (critical / other) with the one-sided Clopper–Pearson bound,
  and coverage (accepted ÷ legible), under a given scorer and thresholds;
- packet safety: unsafe pre-checks and planted-defect escapes;
- the quality gate: retake recall and false retakes;
- evidence-box IoU, calibration (adaptive ECE, Brier), latency.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qareena.config import Catalog, FieldKind
from qareena.eval.compare import score_field
from qareena.pipeline.records import FieldRecord
from qareena.pipeline.scoring import NAME_KINDS, NUMBER_KINDS, eligible, features
from qareena.stats import clopper_pearson_upper, wilson_interval

SCORED_STATES = ("legible", "unreadable", "absent")  # ambiguous fields are reported, never scored


@dataclass
class FieldRow:
    packet: str
    doc: str
    field: str
    kind: FieldKind
    critical: bool
    required: bool
    state: str  # legible | unreadable | ambiguous | absent | not_read (the document was not read)
    condition: str
    layout: str
    held_out: bool
    expected: str | None
    predicted: str | None
    correct: bool
    first_correct: bool  # the VLM's first full-page answer, before any rule (ablation A)
    eligible: bool
    x: list[float] = field(repr=False)
    lp_min: float | None = None
    iou: float | None = None
    sources: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass
class PacketRow:
    packet: str
    scenario: str
    defects: tuple[str, ...]
    expected: str
    outcome: str
    reasons: tuple[str, ...]
    wall_s: float
    expected_retakes: dict[str, str]
    retakes: dict[str, str]
    doc_conditions: dict[str, str]


def kind_group(kind: FieldKind) -> str:
    if kind in NAME_KINDS:
        return "name"
    if kind in NUMBER_KINDS:
        return "number"
    return "date" if kind == FieldKind.DATE else "text"


def load(split_dir: Path, run_dir: Path, catalog: Catalog) -> tuple[list[FieldRow], list[PacketRow]]:
    rows: list[FieldRow] = []
    packets: list[PacketRow] = []
    for trace_path in sorted(run_dir.glob("*.json")):
        if trace_path.name == "run.json":
            continue
        result = json.loads(trace_path.read_text(encoding="utf-8"))
        truth = json.loads((split_dir / result["packet_id"] / "truth.json").read_text(encoding="utf-8"))
        packets.append(
            PacketRow(
                packet=result["packet_id"],
                scenario=truth["scenario"],
                defects=tuple(truth["defects"]),
                expected=truth["expected_outcome"],
                outcome=result["outcome"],
                reasons=tuple(result["reasons"]),
                wall_s=float(result.get("wall_s", 0.0)),
                expected_retakes=truth["expected_retakes"],
                retakes=result.get("retake", {}),
                doc_conditions={d: v["condition"] for d, v in truth["documents"].items()},
            )
        )
        rows += _field_rows(truth, result, catalog)
    return rows, packets


def _field_rows(truth: dict, result: dict, catalog: Catalog) -> list[FieldRow]:
    out = []
    for doc_type, tdoc in truth["documents"].items():
        rdoc = result["documents"].get(doc_type)
        spec_doc = catalog.documents[doc_type]
        values = truth["values"][doc_type]
        for name, spec in spec_doc.fields.items():
            label = tdoc["fields"].get(name, {"state": "absent", "box": None})
            rec_json = (rdoc or {}).get("fields", {}).get(name)
            state = label["state"] if rec_json is not None else "not_read"
            expected_raw = values.get(name)
            exp = score_field(expected_raw if state != "unreadable" else None, None, spec).expected
            if rec_json is None:
                rec = FieldRecord(doc_type=doc_type, field=name, raw=None, value=None)
            else:
                rec = FieldRecord.model_validate(rec_json)
            truth_value = None if state in ("unreadable", "absent") else expected_raw
            s = score_field(truth_value, rec.value, spec)
            first = rec.evidence.first_raw if rec.evidence.first_raw is not None else rec.raw
            out.append(
                FieldRow(
                    packet=truth["packet_id"],
                    doc=doc_type,
                    field=name,
                    kind=spec.kind,
                    critical=spec.critical,
                    required=spec.required,
                    state=state,
                    condition=tdoc["condition"],
                    layout=tdoc["layout"],
                    held_out=bool(tdoc.get("held_out_layout")),
                    expected=exp,
                    predicted=s.predicted,
                    correct=s.correct,
                    first_correct=score_field(truth_value, first, spec).correct,
                    eligible=eligible(rec),
                    x=features(rec, spec.kind),
                    lp_min=rec.evidence.lp_min,
                    iou=_iou(rec.evidence.box, label.get("box"), rdoc) if rec.value is not None else None,
                    sources=tuple(str(src) for src in rec.sources),
                    reasons=tuple(rec.reasons),
                )
            )
    return out


def _iou(box: Sequence[float] | None, truth: Sequence[float] | None, rdoc: dict | None) -> float | None:
    """IoU in photo pixels; evidence boxes live on the rectified read image."""
    if box is None or truth is None or rdoc is None:
        return None
    q = rdoc.get("quality", {})
    quad, size = q.get("quad"), q.get("read_size")
    x0, y0, x1, y1 = box
    if quad and size:
        w, h = size
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        m = cv2.getPerspectiveTransform(src, np.float32(quad))
        pts = cv2.perspectiveTransform(np.float32([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]]), m)[:, 0]
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
    tx0, ty0, tx1, ty1 = truth
    iw, ih = max(0.0, min(x1, tx1) - max(x0, tx0)), max(0.0, min(y1, ty1) - max(y0, ty0))
    inter = iw * ih
    union = (x1 - x0) * (y1 - y0) + (tx1 - tx0) * (ty1 - ty0) - inter
    return float(inter / union) if union > 0 else 0.0


# --- aggregation -------------------------------------------------------------------------------


def rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}


def bound(errors: int, n: int) -> dict[str, Any]:
    return {
        "errors": errors,
        "n": n,
        "rate": round(errors / n, 4) if n else None,
        "upper95": round(clopper_pearson_upper(errors, n), 4),
    }


def p_of(x: Sequence[float], weights: Sequence[float]) -> float:
    z = float(np.dot(weights, x))
    return 1.0 / (1.0 + float(np.exp(-max(min(z, 40.0), -40.0))))


def accepted(r: FieldRow, weights: Sequence[float], taus: tuple[float | None, float | None]) -> bool:
    tau = taus[0] if r.critical else taus[1]
    return tau is not None and r.eligible and p_of(r.x, weights) >= tau


def summarize(
    rows: list[FieldRow],
    packets: list[PacketRow],
    *,
    weights: Sequence[float],
    taus: tuple[float | None, float | None],
) -> dict[str, Any]:
    read = [r for r in rows if r.state != "not_read"]
    legible = [r for r in read if r.state == "legible"]
    unreadable = [r for r in read if r.state == "unreadable"]
    valued = [r for r in read if r.predicted is not None and r.state in SCORED_STATES]
    acc = [r for r in valued if accepted(r, weights, taus)]
    acc_crit = [r for r in acc if r.critical]
    acc_other = [r for r in acc if not r.critical]
    out: dict[str, Any] = {
        "packets": len(packets),
        "fields_read": len(read),
        "states": dict(Counter(r.state for r in rows)),
        "reading": {
            "exact_legible": rate(sum(r.correct for r in legible), len(legible)),
            "first_pass_exact_legible": rate(sum(r.first_correct for r in legible), len(legible)),
            "false_null": rate(sum(r.predicted is None for r in legible), len(legible)),
            "false_fill": rate(sum(r.predicted is not None for r in unreadable), len(unreadable)),
            "by_kind": _by(legible, lambda r: kind_group(r.kind)),
            "by_condition": _by(legible, lambda r: r.condition),
            "by_layout": _by(legible, lambda r: "held-out C" if r.held_out else r.layout.upper()),
        },
        "accepted": {
            "critical": bound(sum(not r.correct for r in acc_crit), len(acc_crit)),
            "other": bound(sum(not r.correct for r in acc_other), len(acc_other)),
            "coverage": rate(sum(1 for r in acc if r.state == "legible"), len(legible)),
        },
        "evidence_box_iou50": rate(
            sum(r.iou is not None and r.iou >= 0.5 for r in acc), sum(r.iou is not None for r in acc)
        ),
        "calibration": _calibration([r for r in valued if r.eligible], weights),
        "packets_outcome": _outcomes(packets),
        "quality_gate": _gate(packets),
        "latency_s": _latency(packets),
    }
    return out


def _by(rows: Iterable[FieldRow], key) -> dict[str, Any]:
    groups: dict[str, list[FieldRow]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    return {k: rate(sum(r.correct for r in v), len(v)) for k, v in sorted(groups.items())}


def _calibration(rows: list[FieldRow], weights: Sequence[float], bins: int = 10) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    p = np.array([p_of(r.x, weights) for r in rows])
    y = np.array([float(r.correct) for r in rows])
    order = np.argsort(p)
    ece = 0.0
    for chunk in np.array_split(order, min(bins, len(rows))):  # equal-mass bins
        if len(chunk):
            ece += len(chunk) / len(rows) * abs(float(p[chunk].mean()) - float(y[chunk].mean()))
    return {"n": len(rows), "ece_adaptive": round(ece, 4), "brier": round(float(np.mean((p - y) ** 2)), 4)}


def _outcomes(packets: list[PacketRow]) -> dict[str, Any]:
    confusion = Counter(f"{p.expected} -> {p.outcome}" for p in packets)
    pre = [p for p in packets if p.outcome == "PRE_CHECKED"]
    planted = [p for p in packets if p.defects or p.expected != "PRE_CHECKED"]
    return {
        "confusion": dict(sorted(confusion.items())),
        "pre_checked": len(pre),
        "planted_defect_escapes": bound(sum(p.outcome == "PRE_CHECKED" for p in planted), len(planted)),
    }


def _gate(packets: list[PacketRow]) -> dict[str, Any]:
    should = caught = normal = false_retake = 0
    for p in packets:
        for doc in p.doc_conditions:
            exp, got = p.expected_retakes.get(doc), p.retakes.get(doc)
            if exp:
                should += 1
                caught += got == exp
            else:
                normal += 1
                false_retake += got is not None and got not in ("DOC_MISSING", "WRONG_DOCUMENT")
    return {"retake_recall": rate(caught, should), "false_retake": rate(false_retake, normal)}


def _latency(packets: list[PacketRow]) -> dict[str, Any]:
    walls = sorted(p.wall_s for p in packets if p.outcome != "AWAITING_RETAKE")
    if not walls:
        return {"n": 0}
    return {
        "n": len(walls),
        "p50": round(statistics.median(walls), 1),
        "p95": round(walls[min(len(walls) - 1, round(0.95 * (len(walls) - 1)))], 1),
    }


def unsafe_prechecks(rows: list[FieldRow], packets: list[PacketRow]) -> list[str]:
    """Pre-checked packets with a wrong critical field or a planted defect (target: none)."""
    wrong = {r.packet for r in rows if r.critical and r.state in SCORED_STATES and not r.correct}
    return [
        p.packet
        for p in packets
        if p.outcome == "PRE_CHECKED" and (p.packet in wrong or p.defects or p.expected != "PRE_CHECKED")
    ]
