"""Fit the confidence scorer on dev, certify thresholds on calibration (ADR-005).

The scorer is an L2-regularised logistic regression over ``scoring.FEATURES``: small, monotone
enough to explain, and nothing to tune but one penalty. It is fitted **only on the dev split**.
Thresholds are then certified **only on the calibration split**, read once per frozen scorer:
for critical and for other fields separately, the fixed grid is scanned from strict to loose
and the loosest threshold whose one-sided Clopper–Pearson bound on the error among accepted
fields stays under alpha is kept. Only fields the production rule could accept count
(``scoring.eligible``), so what is certified is exactly what is accepted.

The certificate is bound to the policy hash and the VLM model: change either and nothing can
be pre-checked until it is certified again.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from qareena.config import Runtime
from qareena.eval.metrics import SCORED_STATES, FieldRow, p_of
from qareena.pipeline.scoring import FEATURES, certificate_path
from qareena.stats import certify_threshold


def training_rows(rows: Sequence[FieldRow]) -> list[FieldRow]:
    """Fields the system gave a value for, with a scorable truth (ambiguous ones excluded)."""
    return [r for r in rows if r.predicted is not None and r.state in SCORED_STATES]


def fit_logistic(rows: Sequence[FieldRow], l2: float = 1.0) -> tuple[float, ...]:
    """Maximum-likelihood weights with an L2 penalty (the bias is not penalised)."""
    x = np.array([r.x for r in rows], dtype=float)
    y = np.array([float(r.correct) for r in rows])
    if len(rows) == 0 or y.min() == y.max():
        raise ValueError("need both correct and wrong fields to fit the scorer")
    penalty = np.full(x.shape[1], l2)
    penalty[0] = 0.0

    def loss(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = np.clip(x @ w, -40, 40)
        p = 1 / (1 + np.exp(-z))
        eps = 1e-12
        nll = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        grad = x.T @ (p - y) + penalty * w
        return float(nll + 0.5 * np.sum(penalty * w * w)), grad

    res = minimize(loss, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"scorer fit did not converge: {res.message}")
    return tuple(round(float(v), 6) for v in res.x)


def check_same_configuration(run_fingerprint: dict[str, Any], runtime: Runtime) -> None:
    """Refuse to certify traces made by another policy, model, decision code or flag set."""
    current = runtime.fingerprint
    keys = ("policy_sha256", "catalog_sha256", "code_sha256", "vlm_model", "features")
    differ = [k for k in keys if run_fingerprint.get(k) != current.get(k)]
    if differ:
        raise ValueError(
            f"calibration traces come from another configuration ({', '.join(differ)}); run again"
        )


def certify(rows: Sequence[FieldRow], weights: Sequence[float], runtime: Runtime) -> dict[str, Any]:
    """Certify tau for critical and other fields on calibration rows; write the certificate."""
    pol = runtime.policy.thresholds
    report: dict[str, Any] = {}
    taus: dict[str, float | None] = {}
    for name, crit, cls in (("critical", True, pol.critical), ("other", False, pol.other)):
        cand = [r for r in training_rows(rows) if r.critical == crit and r.eligible]
        cert = certify_threshold(
            [p_of(r.x, weights) for r in cand],
            [r.correct for r in cand],
            grid=pol.grid,
            alpha=cls.alpha,
            confidence=cls.confidence,
            min_accepted=cls.min_accepted,
        )
        taus[name] = cert.threshold
        report[name] = {
            "eligible": len(cand),
            "threshold": cert.threshold,
            "alpha": cls.alpha,
            "steps": [s.__dict__ for s in cert.steps],
        }
    certificate = {
        "policy_sha256": runtime.policy_sha256,
        "vlm_model": runtime.settings.vlm_model,
        "code_sha256": runtime.code_sha256,
        "feature_flags": runtime.settings.features.model_dump(),
        "features": list(FEATURES),
        "weights": list(weights),
        "tau_critical": taus["critical"],
        "tau_other": taus["other"],
        "report": report,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    path = certificate_path(runtime)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(certificate, indent=1), encoding="utf-8", newline="\n")
    return certificate


def save_weights(weights: Sequence[float], path: Path, *, fitted_on: str, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"features": list(FEATURES), "weights": list(weights), "fitted_on": fitted_on, "n": n}, indent=1
        ),
        encoding="utf-8",
        newline="\n",
    )
