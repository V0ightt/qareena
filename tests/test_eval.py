"""Scorer fit, certification and metric arithmetic on synthetic field rows (no models)."""

import json

import numpy as np
import pytest

from qareena.config import FieldKind, Settings, load_runtime
from qareena.eval import fit, metrics
from qareena.eval.metrics import FieldRow, PacketRow
from qareena.pipeline.scoring import FEATURES, load_certificate


def row(correct, ocr, critical=True, state="legible", eligible=True, packet="P1"):
    x = [0.0] * len(FEATURES)
    x[0] = 1.0
    x[FEATURES.index("src_ocr")] = float(ocr)
    return FieldRow(
        packet=packet,
        doc="tax_card",
        field="tax_id",
        kind=FieldKind.DIGITS,
        critical=critical,
        required=True,
        state=state,
        condition="scan",
        layout="a",
        held_out=False,
        expected="1",
        predicted="1" if correct else "2",
        correct=correct,
        first_correct=correct,
        eligible=eligible,
        x=x,
    )


def rows_with(n_good_ocr, n_bad_ocr, n_good_plain, n_bad_plain, **kw):
    return (
        [row(True, 1, **kw) for _ in range(n_good_ocr)]
        + [row(False, 1, **kw) for _ in range(n_bad_ocr)]
        + [row(True, 0, **kw) for _ in range(n_good_plain)]
        + [row(False, 0, **kw) for _ in range(n_bad_plain)]
    )


def test_fit_learns_that_ocr_agreement_means_correct():
    w = fit.fit_logistic(rows_with(300, 2, 50, 50))
    with_ocr, without = row(True, 1).x, row(True, 0).x
    assert metrics.p_of(with_ocr, w) > 0.95
    assert metrics.p_of(without, w) < 0.6


def test_fit_refuses_labels_of_one_kind_only():
    with pytest.raises(ValueError, match="both correct and wrong"):
        fit.fit_logistic(rows_with(10, 0, 10, 0))


def test_certification_writes_a_certificate_the_pipeline_accepts(tmp_path):
    runtime = load_runtime(Settings(_env_file=None, data_dir=tmp_path))
    calib = rows_with(700, 1, 100, 100) + rows_with(400, 1, 50, 50, critical=False)
    w = fit.fit_logistic(calib)
    cert = fit.certify(calib, w, runtime)
    assert cert["tau_critical"] is not None
    assert cert["features"] == list(FEATURES)
    loaded = load_certificate(runtime)
    assert loaded is not None
    assert loaded.tau_critical == cert["tau_critical"]


def test_too_few_eligible_fields_certify_nothing(tmp_path):
    runtime = load_runtime(Settings(_env_file=None, data_dir=tmp_path))
    calib = rows_with(50, 1, 10, 10)  # far below min_accepted for critical fields
    cert = fit.certify(calib, fit.fit_logistic(calib), runtime)
    assert cert["tau_critical"] is None


def test_certificate_for_other_features_is_ignored(tmp_path):
    runtime = load_runtime(Settings(_env_file=None, data_dir=tmp_path))
    calib = rows_with(700, 1, 100, 100)
    fit.certify(calib, fit.fit_logistic(calib), runtime)
    path = next((tmp_path / "certification").glob("*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["features"] = data["features"][:-1]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_certificate(runtime) is None


def test_accepted_error_bound_and_coverage():
    rows = rows_with(90, 0, 10, 0)
    w = fit.fit_logistic(rows_with(300, 2, 50, 50))
    s = metrics.summarize(rows, [], weights=w, taus=(0.9, 0.9))
    assert s["accepted"]["critical"]["n"] == 90  # only the OCR-backed ones clear 0.9
    assert s["accepted"]["critical"]["errors"] == 0
    assert s["accepted"]["critical"]["upper95"] == pytest.approx(1 - 0.05 ** (1 / 90), abs=1e-3)
    assert s["accepted"]["coverage"]["rate"] == 0.9


def test_unsafe_precheck_counts_wrong_critical_fields_and_planted_defects():
    rows = [row(True, 1, packet="ok"), row(False, 1, packet="bad")]
    packets = [
        PacketRow("ok", "clean", (), "PRE_CHECKED", "PRE_CHECKED", (), 1.0, {}, {}, {}),
        PacketRow("bad", "clean", (), "PRE_CHECKED", "PRE_CHECKED", (), 1.0, {}, {}, {}),
        PacketRow("planted", "expired_id", ("EXPIRED",), "NEEDS_REVIEW", "PRE_CHECKED", (), 1.0, {}, {}, {}),
    ]
    assert metrics.unsafe_prechecks(rows, packets) == ["bad", "planted"]


def test_iou_maps_the_read_image_back_to_the_photo():
    rdoc = {"quality": {"quad": [[100, 100], [1100, 100], [1100, 700], [100, 700]], "read_size": [1001, 601]}}
    # a box on the rectified image is shifted by the quad's offset in the photo
    assert metrics._iou((0, 0, 100, 50), (100, 100, 200, 150), rdoc) == pytest.approx(1.0, abs=0.01)
    assert metrics._iou((0, 0, 100, 50), (100, 100, 200, 150), {"quality": {}}) == 0.0


def test_calibration_of_a_perfect_scorer_is_near_zero():
    rows = rows_with(500, 0, 0, 500)
    w = fit.fit_logistic(rows)
    cal = metrics._calibration(rows, w)
    assert cal["ece_adaptive"] < 0.02
    assert np.isfinite(cal["brier"])


def test_risk_coverage_ranks_errors_last_for_a_good_score():
    good = rows_with(90, 0, 0, 10)  # the 10 wrong ones have no OCR support
    w = fit.fit_logistic(rows_with(300, 2, 50, 50))
    rc = metrics.risk_coverage(good, lambda r: metrics.p_of(r.x, w))
    assert rc["coverage_at_risk_0.01"] == pytest.approx(0.9)
    worst = metrics.risk_coverage(good, lambda r: -metrics.p_of(r.x, w))  # errors ranked first
    assert rc["aurc"] < worst["aurc"]


def test_report_renders_every_section():
    rows = rows_with(90, 1, 10, 5)
    w = fit.fit_logistic(rows)
    packets = [
        PacketRow("P1", "clean", (), "PRE_CHECKED", "NEEDS_REVIEW", (), 20.0, {}, {}, {"tax_card": "scan"})
    ]
    s = metrics.summarize(rows, packets, weights=w, taus=(None, None))
    s["ablations"] = metrics.ablations(rows, w)
    text = metrics.markdown(s, "test")
    for heading in ("## Reading", "## Accepted automatically", "## Packets", "## What each part buys"):
        assert heading in text


@pytest.mark.parametrize(("code", "degrees"), [(0, 90), (1, 180), (2, 270)])
def test_unrotate_inverts_the_turn_the_pipeline_made(code, degrees):
    import cv2

    img = np.zeros((60, 100), np.uint8)
    img[10, 70] = 255  # one marked pixel at (x=70, y=10)
    turned = cv2.rotate(img, code)  # 0: clockwise, 1: 180, 2: counter-clockwise
    ty, tx = (int(v[0]) for v in np.nonzero(turned))
    pts, size = metrics.unrotate(np.float32([[tx, ty]]), (turned.shape[1], turned.shape[0]), degrees)
    assert size == (100, 60)
    assert tuple(pts[0]) == (70, 10)


def test_certification_refuses_traces_from_other_code(tmp_path):
    runtime = load_runtime(Settings(_env_file=None, data_dir=tmp_path))
    fit.check_same_configuration(runtime.fingerprint, runtime)  # same configuration: fine
    other = runtime.fingerprint | {"reading_code_sha256": "0" * 64}
    with pytest.raises(ValueError, match="reading_code_sha256"):
        fit.check_same_configuration(other, runtime)
    # a different scorer is fine: the traces are scored afresh
    fit.check_same_configuration(runtime.fingerprint | {"scoring_sha256": "0" * 64}, runtime)
