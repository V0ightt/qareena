"""Web app: security headers, auth, CSRF guard, upload -> read -> review -> approve, audit, retention."""

import base64
import sys
import time
from pathlib import Path

import cv2
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from qareena.config import Settings, load_runtime
from qareena.pipeline.run import Pipeline
from qareena.web.app import create_app
from qareena.web.audit import verify as verify_audit

sys.path.insert(0, str(Path(__file__).parent))
from test_pipeline import FakeOcr, FakeVlm, certify, images
from test_rules import TODAY, base_values

PASSWORD = "s3cret-for-tests"  # noqa: S105 - test fixture
ORIGIN = {"Origin": "http://testserver"}
AUTH = {"Authorization": "Basic " + base64.b64encode(f"reviewer:{PASSWORD}".encode()).decode()}


@pytest.fixture
def setup(tmp_path):
    runtime = load_runtime(Settings(_env_file=None, data_dir=tmp_path))
    certify(runtime)
    values = base_values()

    def make_pipeline():
        return Pipeline(runtime, FakeVlm(values), FakeOcr(values, runtime.catalog), today=TODAY)

    app = create_app(runtime, make_pipeline, reviewer_password=PASSWORD)
    with TestClient(app) as client:
        yield client, runtime, tmp_path


def png(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def upload(client, docs=None):
    imgs = images()
    files = {d: (f"{d}.png", png(img), "image/png") for d, img in imgs.items() if docs is None or d in docs}
    return client.post("/cases", files=files, headers=ORIGIN, follow_redirects=False)


def wait_done(client, case_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get(f"/cases/{case_id}/status").json()
        if s["status"] in ("DONE", "FAILED"):
            return s
        time.sleep(0.2)
    raise AssertionError("case was not processed in time")


def test_pages_send_security_headers(setup):
    client, _, _ = setup
    r = client.get("/")
    assert r.status_code == 200
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert 'dir="rtl"' in r.text


def test_reviewer_pages_need_the_password(setup):
    client, _, _ = setup
    assert client.get("/review").status_code == 401
    bad = {"Authorization": "Basic " + base64.b64encode(b"reviewer:wrong").decode()}
    assert client.get("/review", headers=bad).status_code == 401
    assert client.get("/review", headers=AUTH).status_code == 200
    assert client.get("/system", headers=AUTH).status_code == 200


def test_cross_site_posts_are_refused(setup):
    client, _, _ = setup
    r = client.post("/cases", files={}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    assert client.post("/cases", files={}).status_code == 403  # no Origin or Referer at all
    assert client.post("/cases", files={}, headers={"Origin": "null"}).status_code == 403


def test_referrer_policy_keeps_the_origin_header_for_our_own_forms(setup):
    client, _, _ = setup
    # "no-referrer" would make browsers send Origin: null on form posts, refused by the guard
    assert client.get("/").headers["referrer-policy"] == "same-origin"


def test_a_non_image_is_rejected_with_an_iraqi_arabic_message(setup):
    client, _, _ = setup
    r = client.post(
        "/cases", files={"tax_card": ("x.png", b"<script>alert(1)</script>", "image/png")}, headers=ORIGIN
    )
    assert r.status_code == 400
    assert "نوع الملف مو مدعوم" in r.text  # Iraqi Arabic, names the formats


def test_full_flow_upload_read_review_approve(setup):
    client, _, tmp = setup
    r = upload(client)
    assert r.status_code == 303
    case_id = r.headers["location"].rsplit("/", 1)[1]
    assert wait_done(client, case_id)["outcome"] == "PRE_CHECKED"
    assert "وصلتنا كل الوثائق" in client.get(f"/cases/{case_id}").text

    page = client.get(f"/review/{case_id}", headers=AUTH)
    assert page.status_code == 200
    assert "ما تحققنا من أصالة الوثائق" in page.text
    img = client.get(f"/review/{case_id}/image/tax_card", headers=AUTH)
    assert img.headers["content-type"] == "image/jpeg"

    r = client.post(
        f"/review/{case_id}/decision",
        data={"verdict": "approve"},
        headers=AUTH | ORIGIN,
        follow_redirects=False,
    )
    assert r.status_code == 303
    log = (tmp / "audit" / "audit.jsonl").read_text(encoding="utf-8")
    assert "case_approve" in log
    assert verify_audit(tmp / "audit" / "audit.jsonl") == []
    assert "عبدالله" not in log  # no field values in the audit log


def test_stored_images_are_encrypted(setup):
    client, _, tmp = setup
    case_id = upload(client).headers["location"].rsplit("/", 1)[1]
    wait_done(client, case_id)
    raw = (tmp / "cases.sqlite").read_bytes()
    assert b"\xff\xd8\xff" not in raw  # no JPEG in the clear
    assert b"\x89PNG" not in raw


def test_flagged_fields_must_be_resolved_before_approval(setup):
    client, _, _ = setup
    case_id = (
        upload(client, docs={"tax_card"}).headers["location"].rsplit("/", 1)[1]
    )  # three documents missing
    status = wait_done(client, case_id)
    assert status["outcome"] == "AWAITING_RETAKE"
    # a case with open flags: write one directly through the reviewer API to check gating
    ctx_page = client.get(f"/review/{case_id}", headers=AUTH)
    assert ctx_page.status_code == 200
    r = client.post(f"/review/{case_id}/decision", data={"verdict": "reject"}, headers=AUTH | ORIGIN)
    assert r.status_code == 400  # a note is required for anything but approval
    r = client.post(
        f"/review/{case_id}/field",
        data={"doc_type": "tax_card", "field_name": "tax_id", "action": "edit", "value": "1"},
        headers=AUTH | ORIGIN,
    )
    assert r.status_code == 400  # a correction needs a note


def test_approval_is_blocked_while_a_flag_is_open(setup, monkeypatch):
    client, _, _ = setup
    case_id = upload(client).headers["location"].rsplit("/", 1)[1]
    wait_done(client, case_id)
    import qareena.web.app as web

    original = web._workbench_context

    def with_open_flag(case):
        ctx = original(case)
        ctx["unresolved"] = 1
        return ctx

    monkeypatch.setattr(web, "_workbench_context", with_open_flag)
    r = client.post(f"/review/{case_id}/decision", data={"verdict": "approve"}, headers=AUTH | ORIGIN)
    assert r.status_code == 409


def test_kill_switch_sends_new_cases_to_review(setup):
    client, _, _ = setup
    r = client.post(
        "/system/kill-switch",
        data={"on": "1", "note": "drill"},
        headers=AUTH | ORIGIN,
        follow_redirects=False,
    )
    assert r.status_code == 303
    case_id = upload(client).headers["location"].rsplit("/", 1)[1]
    assert wait_done(client, case_id)["outcome"] == "NEEDS_REVIEW"
    assert "مفعّل" in client.get("/system", headers=AUTH).text


def test_images_are_deleted_after_the_retention_window(setup):
    client, runtime, _ = setup
    case_id = upload(client).headers["location"].rsplit("/", 1)[1]
    wait_done(client, case_id)
    store = client.app.state.q.store
    hours = runtime.policy.retention.image_hours
    assert store.purge(time.time() + (hours - 1) * 3600) == 0
    assert store.purge(time.time() + (hours + 1) * 3600) > 0
    assert client.get(f"/review/{case_id}/image/tax_card", headers=AUTH).status_code == 404
    assert store.get(case_id).outcome is not None  # the decision record stays


def test_workbench_offers_the_ocr_reading_when_the_readers_disagree(tmp_path):
    runtime = load_runtime(Settings(_env_file=None, data_dir=tmp_path))
    certify(runtime)
    read, printed = base_values(), base_values()
    read["iq_national_card_front"]["laqab"] = "الجبوي"  # the VLM dropped a letter

    def make_pipeline():
        return Pipeline(runtime, FakeVlm(read), FakeOcr(printed, runtime.catalog), today=TODAY)

    with TestClient(create_app(runtime, make_pipeline, reviewer_password=PASSWORD)) as client:
        case_id = upload(client).headers["location"].rsplit("/", 1)[1]
        wait_done(client, case_id)
        page = client.get(f"/review/{case_id}", headers=AUTH).text
        assert "القارئ الثاني قرأ" in page
        assert "الجبوري" in page  # the OCR's reading, offered as one click


def test_submissions_are_rate_limited_per_address(setup, monkeypatch):
    client, _, _ = setup
    import qareena.web.app as web

    monkeypatch.setattr(web, "MAX_CASES_PER_HOUR", 2)
    codes = [
        client.post("/cases", files={"tax_card": ("x.png", b"x", "image/png")}, headers=ORIGIN).status_code
        for _ in range(3)
    ]
    assert codes[-1] == 429


def test_wrong_document_message_names_what_we_saw():
    from qareena.web import text

    msg = text.retake("WRONG_DOCUMENT:iq_national_card_front")
    assert "البطاقة الوطنية (الوجه)" in msg
    assert text.retake("WRONG_DOCUMENT") == text.RETAKE["WRONG_DOCUMENT"]
