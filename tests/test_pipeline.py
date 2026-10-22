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
    def __init__(self, values, fail=(), crop=None):
        self.values, self.fail, self.calls = values, set(fail), []
        self.crop = crop or {}  # (doc_type, field) -> answer to a crop re-read

    def read(self, doc_type, png):
        self.calls.append(doc_type)
        if doc_type in self.fail:
            return VlmRead(None, "", (), {}, "READER_FAILED: connection refused")
        data = self.values[doc_type]
        return VlmRead(data, json.dumps(data, ensure_ascii=False), (), {"wall_s": 0.01})

    def read_field(self, doc_type, field, png):
        self.calls.append(f"{doc_type}.{field}")
        return self.crop.get((doc_type, field), self.values[doc_type].get(field))


# Each test document gets its own card tint, so the fake OCR can tell which one it is reading
# (rectification resizes every card to the same size).
TINTS = {
    "iq_national_card_front": (236, 240, 244),
    "iq_national_card_back": (222, 240, 244),
    "chamber_of_commerce_id": (236, 224, 244),
    "tax_card": (236, 240, 226),
}


def _row(text, x0, x1, y, h, score=0.97):
    return OcrLine(str(text), score, ((x0, y), (x1, y), (x1, y + h), (x0, y + h)))


class FakeOcr:
    """Reads each document like a real card: every printed label with its value to its left
    (Arabic reading order), MRZ lines unlabelled at the bottom. Every value is read exactly."""

    def __init__(self, values, catalog, labels=True):
        self.pages = {}
        for doc_type, vals in values.items():
            lines = []
            for i, (name, spec) in enumerate(catalog.documents[doc_type].fields.items()):
                y = 40 + 60 * i
                if labels and spec.kind.value != "mrz_line":
                    lines.append(_row(spec.label_ar, 1350, 1600, y + 5, 30))
                if vals.get(name):
                    lines.append(_row(vals[name], 700, 1300, y, 40))
            self.pages[doc_type] = tuple(lines)

    def read(self, image):
        h, w = image.shape[:2]
        tint = np.median(image.reshape(-1, 3), axis=0)
        doc = min(TINTS, key=lambda d: float(np.abs(np.array(TINTS[d]) - tint).sum()))
        return OcrResult(self.pages.get(doc, ()), 0.01, w, h)


def tinted(doc_type, img):
    out = img.copy()
    out[(img == (236, 240, 244)).all(axis=2)] = TINTS[doc_type]
    return out


def images(blur=()):
    imgs = {}
    for d in base_values():
        good = on_table(tinted(d, card()))
        imgs[d] = degrade.gaussian_blur(good, 6.0) if d in blur else good
    return imgs


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


def run(runtime, values=None, *, printed=None, crop=None, labels=True, **kw):
    """``values``: what the VLM reads; ``printed``: what is on the card (the OCR reads it exactly)."""
    values = values or base_values()
    vlm = FakeVlm(values, kw.pop("fail", ()), crop)
    ocr = FakeOcr(printed or values, runtime.catalog, labels=labels)
    p = Pipeline(runtime, vlm, ocr, today=TODAY)
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


def test_field_shift_is_blocked_even_though_the_value_is_on_the_card(runtime):
    """The VLM returned the grandfather's name as the father's: it is printed, but elsewhere."""
    certify(runtime)
    read = base_values()
    read["iq_national_card_front"]["father_name"] = "حسين"
    result, _ = run(runtime, values=read, printed=base_values())
    father = result.documents["iq_national_card_front"].fields["father_name"]
    assert "VALUE_AT_OTHER_LABEL" in father.reasons
    assert father.band == Band.REVIEW
    assert result.outcome == Outcome.NEEDS_REVIEW


def test_label_echo_is_recovered_by_a_crop_reread_and_then_checked_normally(runtime):
    certify(runtime)
    read = base_values()
    read["iq_national_card_back"]["issuing_authority"] = "دەسەڵاتی دەرکردن"  # its Kurdish label
    right = base_values()["iq_national_card_back"]["issuing_authority"]
    result, vlm = run(
        runtime,
        values=read,
        printed=base_values(),
        crop={("iq_national_card_back", "issuing_authority"): right},
    )
    rec = result.documents["iq_national_card_back"].fields["issuing_authority"]
    assert "iq_national_card_back.issuing_authority" in vlm.calls
    assert rec.value == right
    assert rec.evidence.first_raw == "دەسەڵاتی دەرکردن"
    assert "NOTE_RECOVERED_FROM_CROP" in rec.reasons
    assert "LABEL_ECHO" not in rec.reasons
    assert result.outcome == Outcome.PRE_CHECKED, result.reasons


def test_unrecovered_label_echo_stays_unread(runtime):
    certify(runtime)
    read = base_values()
    read["iq_national_card_back"]["issuing_authority"] = "دەسەڵاتی دەرکردن"
    result, _ = run(
        runtime,
        values=read,
        printed=base_values(),
        crop={("iq_national_card_back", "issuing_authority"): None},
    )
    rec = result.documents["iq_national_card_back"].fields["issuing_authority"]
    assert rec.value is None
    assert "LABEL_ECHO" in rec.reasons
    assert result.outcome == Outcome.NEEDS_REVIEW


def test_mrz_misread_fillers_and_o_zero_are_repaired_only_with_valid_check_digits(runtime):
    certify(runtime)
    read = base_values()
    good = read["iq_national_card_back"]["mrz_line_2"]
    read["iq_national_card_back"]["mrz_line_2"] = good.replace("UTO", "UT0").replace("<<<", "<<<<<<", 1)
    result, _ = run(runtime, values=read, printed=base_values())
    rec = result.documents["iq_national_card_back"].fields["mrz_line_2"]
    assert rec.value == good
    assert "NOTE_MRZ_REPAIRED" in rec.reasons
    assert result.outcome == Outcome.PRE_CHECKED, result.reasons


def test_on_a_layout_without_known_labels_ocr_alone_cannot_accept_a_critical_field(runtime):
    certify(runtime)
    result, _ = run(runtime, labels=False)
    tax_id = result.documents["tax_card"].fields["tax_id"]  # critical, and no MRZ or other document has it
    assert tax_id.band != Band.ACCEPT
    assert result.outcome == Outcome.NEEDS_REVIEW
