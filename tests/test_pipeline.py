"""End-to-end pipeline tests with fake readers: no GPU, no OCR models, deterministic."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from qareena.config import Settings, load_runtime
from qareena.pipeline.records import Band, Outcome
from qareena.pipeline.run import Pipeline, VlmRead
from qareena.pipeline.scoring import FEATURES, certificate_path
from qareena.readers.ocr import OcrLine, OcrResult
from qareena.synth import degrade

sys.path.insert(0, str(Path(__file__).parent))
from test_quality import card, on_table
from test_rules import TODAY, base_values


class FakeVlm:
    def __init__(self, values, fail=()):
        self.values, self.fail, self.calls = values, set(fail), []

    def read(self, doc_type, png):
        self.calls.append(doc_type)
        if doc_type in self.fail:
            return VlmRead(None, "", (), {}, "READER_FAILED: connection refused")
        data = self.values[doc_type]
        return VlmRead(data, json.dumps(data, ensure_ascii=False), (), {"wall_s": 0.01})

    def read_field(self, doc_type, field, png):
        self.calls.append(f"{doc_type}.{field}")
        return self.values[doc_type].get(field)


class FakeOcr:
    """Pretends the OCR engine read every value exactly (grounded)."""

    def __init__(self, values):
        self.lines = [
            OcrLine(str(v), 0.97, ((0, i * 30), (300, i * 30), (300, i * 30 + 25), (0, i * 30 + 25)))
            for i, v in enumerate(x for d in values.values() for x in d.values() if x)
        ]

    def read(self, image):
        return OcrResult(tuple(self.lines), 0.01, image.shape[1], image.shape[0])


def images(blur=()):
    good = on_table(card())
    return {d: (degrade.gaussian_blur(good, 6.0) if d in blur else good) for d in base_values()}


@pytest.fixture
def runtime(tmp_path):
    return load_runtime(Settings(_env_file=None, data_dir=tmp_path))


def certify(runtime, tau=0.5):
    path = certificate_path(runtime)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = [3.0] + [0.0] * (len(FEATURES) - 1)  # permissive: p ~ 0.95 for every field
    path.write_text(
        json.dumps(
            {
                "policy_sha256": runtime.policy_sha256,
                "vlm_model": runtime.settings.vlm_model,
                "weights": weights,
                "tau_critical": tau,
                "tau_other": tau,
            }
        ),
        encoding="utf-8",
    )


def run(runtime, values=None, **kw):
    values = values or base_values()
    vlm = FakeVlm(values, kw.pop("fail", ()))
    p = Pipeline(runtime, vlm, FakeOcr(values), today=TODAY)
    try:
        return p.run("PKT-1", kw.pop("imgs", images()), **kw), vlm
    finally:
        p.close()


def test_without_a_certificate_nothing_can_be_prechecked(runtime):
    result, _ = run(runtime)
    assert result.outcome == Outcome.NEEDS_REVIEW
    assert "POLICY_NOT_CERTIFIED" in result.reasons
    assert all(r.band != Band.ACCEPT for d in result.documents.values() for r in d.fields.values())


def test_consistent_packet_is_prechecked_once_certified(runtime):
    certify(runtime)
    result, _ = run(runtime)
    assert result.outcome == Outcome.PRE_CHECKED, result.reasons
    assert result.fingerprint["policy_sha256"] == runtime.policy_sha256


def test_certificate_for_another_policy_is_ignored(runtime):
    certify(runtime)
    path = certificate_path(runtime)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["policy_sha256"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    result, _ = run(runtime)
    assert "POLICY_NOT_CERTIFIED" in result.reasons


def test_blurred_photo_asks_for_a_retake_without_spending_model_time(runtime):
    result, vlm = run(runtime, imgs=images(blur=("tax_card",)))
    assert result.outcome == Outcome.AWAITING_RETAKE
    assert result.retake == {"tax_card": "BLUR"}
    assert "tax_card" not in vlm.calls


def test_exhausted_retakes_read_best_effort_and_go_to_review(runtime):
    certify(runtime)
    result, vlm = run(runtime, imgs=images(blur=("tax_card",)), retakes_used={"tax_card": 3})
    assert result.outcome == Outcome.NEEDS_REVIEW
    assert "tax_card" in vlm.calls
    assert any(r.startswith("RETAKES_EXHAUSTED:tax_card") for r in result.reasons)


def test_reader_failure_degrades_to_review(runtime):
    certify(runtime)
    result, _ = run(runtime, fail=("chamber_of_commerce_id",))
    assert result.outcome == Outcome.NEEDS_REVIEW
    assert "SYSTEM_DEGRADED" in result.reasons


def test_kill_switch_sends_everything_to_review(runtime):
    certify(runtime)
    result, _ = run(runtime, kill_switch=True)
    assert result.outcome == Outcome.NEEDS_REVIEW
    assert "KILL_SWITCH" in result.reasons


def test_name_conflict_is_reviewed_with_the_field_in_red(runtime):
    certify(runtime)
    v = base_values()
    v["chamber_of_commerce_id"]["member_name"] = "علي عبدالله حسن الجبوري"
    result, _ = run(runtime, values=v)
    assert result.outcome == Outcome.NEEDS_REVIEW
    gf = result.documents["iq_national_card_front"].fields["grandfather_name"]
    assert gf.band == Band.REVIEW
    assert "NAME_CONFLICT" in gf.reasons


def test_missing_document_asks_for_it(runtime):
    imgs = images()
    imgs.pop("tax_card")
    result, _ = run(runtime, imgs=imgs)
    assert result.outcome == Outcome.AWAITING_RETAKE
    assert result.retake == {"tax_card": "DOC_MISSING"}


def test_result_is_json_serialisable(runtime):
    result, _ = run(runtime)
    assert json.loads(result.model_dump_json())["packet_id"] == "PKT-1"


def test_wrong_document_is_detected(runtime):
    v = base_values()
    v["tax_card"] = dict.fromkeys(v["tax_card"])  # the model found none of the tax-card fields
    result, _ = run(runtime, values=v)
    assert result.retake.get("tax_card") == "WRONG_DOCUMENT"


def test_unused_numpy_import_guard():
    assert np is not None
