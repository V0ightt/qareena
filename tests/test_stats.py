import numpy as np
import pytest

from qareena.stats import certify_threshold, clopper_pearson_upper, wilson_interval


@pytest.mark.parametrize(
    ("k", "n", "expected"),
    [(0, 80, 0.0368), (0, 30, 0.0950), (0, 600, 0.00498), (2, 600, 0.01048), (5, 1150, 0.00909)],
)
def test_clopper_pearson_known_values(k, n, expected):
    assert clopper_pearson_upper(k, n) == pytest.approx(expected, abs=5e-5)


def test_clopper_pearson_edges():
    assert clopper_pearson_upper(0, 0) == 1.0
    assert clopper_pearson_upper(5, 5) == 1.0
    with pytest.raises(ValueError, match="invalid counts"):
        clopper_pearson_upper(3, 2)


def test_wilson_known_value_and_edges():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)


GRID = [0.99, 0.95, 0.90, 0.80, 0.50]


def test_all_correct_certifies_the_loosest_threshold():
    scores = np.linspace(0.5, 1.0, 1000)
    cert = certify_threshold(scores, np.ones(1000, bool), grid=GRID, alpha=0.01)
    assert cert.threshold == 0.50
    assert cert.certified
    assert cert.chosen.errors == 0


def test_stops_at_first_failure_and_keeps_the_previous_threshold():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.5, 1.0, 5000)
    correct = np.ones(5000, bool)
    correct[(scores < 0.95) & (rng.uniform(size=5000) < 0.2)] = False  # errors below 0.95 only
    cert = certify_threshold(scores, correct, grid=GRID, alpha=0.02)
    assert cert.threshold == 0.95
    assert not cert.steps[-1].passed
    assert cert.steps[-1].threshold == 0.90


def test_failing_first_threshold_certifies_nothing():
    scores = np.full(500, 0.999)
    correct = np.zeros(500, bool)
    cert = certify_threshold(scores, correct, grid=GRID, alpha=0.01)
    assert cert.threshold is None
    assert not cert.certified


def test_thresholds_with_too_few_accepted_items_are_skipped_not_failed():
    scores = np.concatenate([np.full(10, 0.995), np.full(990, 0.92)])
    cert = certify_threshold(scores, np.ones(1000, bool), grid=GRID, alpha=0.01, min_accepted=300)
    assert next(s.threshold for s in cert.steps) == 0.90  # 0.99 and 0.95 accept only 10 -> skipped
    assert cert.threshold == 0.50


def test_min_n_for_zero_errors_matches_the_bound():
    from qareena.stats import min_n_for_zero_errors

    n = min_n_for_zero_errors(0.01, 0.95)
    assert n == 299
    assert clopper_pearson_upper(0, n) <= 0.01 < clopper_pearson_upper(0, n - 1)
    assert min_n_for_zero_errors(0.02, 0.95) == 149


def test_tiny_strict_buckets_cannot_end_the_scan():
    # 20 items at 0.999 can never certify 1%; they must be skipped, not fail the scan
    scores = np.concatenate([np.full(20, 0.999), np.full(980, 0.85)])
    cert = certify_threshold(scores, np.ones(1000, bool), grid=GRID, alpha=0.01)
    assert cert.threshold == 0.50
    assert cert.steps[0].accepted >= 299


def test_input_validation():
    with pytest.raises(ValueError, match="same length"):
        certify_threshold([0.9], [True, False], grid=GRID, alpha=0.01)
    with pytest.raises(ValueError, match="alpha"):
        certify_threshold([0.9], [True], grid=GRID, alpha=0.0)
    with pytest.raises(ValueError, match="NaN"):
        certify_threshold([float("nan")], [True], grid=GRID, alpha=0.01)
