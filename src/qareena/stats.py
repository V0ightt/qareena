"""Small, exact statistics for honest reporting and threshold certification.

* ``clopper_pearson_upper`` — one-sided exact upper bound on an error rate. Used for every
  safety claim ("accepted critical fields are wrong <= 1% at 95% confidence").
* ``wilson_interval`` — two-sided interval for descriptive rates (coverage, recall).
* ``certify_threshold`` — fixed-sequence testing (Learn-Then-Test style): scan thresholds
  from strict to loose, stop at the first whose bound exceeds alpha, keep the one before.
  Testing in a fixed order controls the family-wise error without a Bonferroni correction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, norm


def clopper_pearson_upper(errors: int, n: int, confidence: float = 0.95) -> float:
    """Exact one-sided upper confidence bound for a binomial proportion."""
    _check_counts(errors, n)
    if n == 0 or errors >= n:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, n - errors))


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided Wilson score interval. Returns (0, 1) when n == 0."""
    _check_counts(successes, n)
    if n == 0:
        return 0.0, 1.0
    z = float(norm.ppf(0.5 + confidence / 2))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class ThresholdStep:
    threshold: float
    accepted: int
    errors: int
    upper_bound: float
    passed: bool


@dataclass(frozen=True)
class Certification:
    """Outcome of certifying a confidence threshold on held-out calibration data."""

    threshold: float | None  # None: nothing can be auto-accepted; everything goes to a human
    alpha: float
    confidence: float
    steps: tuple[ThresholdStep, ...]

    @property
    def certified(self) -> bool:
        return self.threshold is not None

    @property
    def chosen(self) -> ThresholdStep | None:
        return next((s for s in self.steps if s.threshold == self.threshold), None)


def certify_threshold(
    scores: Sequence[float] | np.ndarray,
    correct: Sequence[bool] | np.ndarray,
    *,
    grid: Sequence[float],
    alpha: float,
    confidence: float = 0.95,
    min_accepted: int = 1,
) -> Certification:
    """Pick the loosest threshold whose error among accepted items is certified <= ``alpha``.

    ``grid`` must be fixed before looking at the calibration labels. It is scanned from the
    strictest (highest) threshold down. A threshold is skipped, not tested, when it accepts
    fewer items than ``min_accepted`` or fewer than could *ever* certify ``alpha`` even with
    zero errors (``min_n_for_zero_errors``). Both conditions depend only on the scores, not
    the labels, so skipping spends none of the error budget. Without this rule a strict
    threshold with a handful of items would fail and end the scan before any useful one is
    tested. The first tested threshold that fails ends the scan.
    """
    s = np.asarray(scores, dtype=float)
    c = np.asarray(correct, dtype=bool)
    if s.shape != c.shape or s.ndim != 1:
        raise ValueError("scores and correct must be 1-D and the same length")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    floor = max(min_accepted, min_n_for_zero_errors(alpha, confidence))

    steps: list[ThresholdStep] = []
    chosen: float | None = None
    for t in sorted(set(grid), reverse=True):
        mask = s >= t
        n = int(mask.sum())
        if n < floor:
            continue
        k = int((~c[mask]).sum())
        ub = clopper_pearson_upper(k, n, confidence)
        ok = ub <= alpha
        steps.append(ThresholdStep(float(t), n, k, ub, ok))
        if not ok:
            break
        chosen = float(t)
    return Certification(chosen, alpha, confidence, tuple(steps))


def min_n_for_zero_errors(alpha: float, confidence: float = 0.95) -> int:
    """Smallest n for which 0 errors in n gives a one-sided upper bound <= alpha.

    With 0 errors the Clopper-Pearson bound is 1 - (1 - confidence)^(1/n), so
    n >= ln(1 - confidence) / ln(1 - alpha). For alpha = 1% at 95% this is 299.
    """
    if not 0 < alpha < 1 or not 0 < confidence < 1:
        raise ValueError("alpha and confidence must be in (0, 1)")
    return math.ceil(math.log(1 - confidence) / math.log(1 - alpha) - 1e-12)


def _check_counts(k: int, n: int) -> None:
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"invalid counts k={k}, n={n}")
