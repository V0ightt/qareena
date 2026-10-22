"""The web app: applicant capture page, reviewer workbench, and the «النظام» system page.

- One process, one GPU: cases are processed one at a time by a single worker thread; pages
  poll a small status endpoint while a case is being read.
- Applicant pages are open; reviewer and system pages need the reviewer password (HTTP Basic,
  ``QAREENA_REVIEWER_PASSWORD``; a random one is printed at start-up when none is set).
- Every state-changing request must come from this site (Origin/Referer check against CSRF).
- Uploads pass the intake guard and are re-encoded before storage (no metadata, no polyglots),
  stored encrypted, and deleted after the retention window.
- CSP ``default-src 'self'``: no CDN, no inline script; Jinja autoescapes every value.
- A reviewer must resolve every flagged field (confirm, or correct with a note) before the file
  can be approved. The decision, never a field value, goes to the hash-chained audit log.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import UploadFile as FormFile  # what request.form() returns

from qareena.config import Runtime
from qareena.pipeline.intake import IntakeError, load_image_bytes
from qareena.pipeline.quality import assess
from qareena.pipeline.run import Pipeline
from qareena.web import text
from qareena.web.audit import AuditLog
from qareena.web.store import CaseStore, load_or_create_key

log = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
READ_SUFFIX = "#read"  # stored next to each upload: the flattened image the readers saw
MAX_CASES_PER_HOUR = 30  # per client address, for the open applicant page
MAX_CHECKS_PER_HOUR = 300  # instant quality checks (about 0.1 s of CPU each)
CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
    "frame-ancestors 'none'; form-action 'self'; base-uri 'none'"
)


@dataclass
class AppState:
    runtime: Runtime
    store: CaseStore
    audit: AuditLog
    make_pipeline: Callable[[], Pipeline]
    reviewer_password: str
    kill_switch_path: Path
    worker: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(1, thread_name_prefix="case")
    )
    _pipeline: Pipeline | None = None
    _recent: dict[str, list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def kill_switch(self) -> bool:
        return self.kill_switch_path.exists()

    def pipeline(self) -> Pipeline:
        if self._pipeline is None:  # models load lazily, on the first case
            self._pipeline = self.make_pipeline()
        return self._pipeline


def create_app(
    runtime: Runtime,
    make_pipeline: Callable[[], Pipeline],
    *,
    reviewer_password: str | None = None,
) -> FastAPI:
    data = runtime.settings.data_dir
    key = load_or_create_key(
        runtime.settings.data_key.get_secret_value() if runtime.settings.data_key else None,
        data / "keys" / "store.key",
    )
    password = reviewer_password or secrets.token_urlsafe(12)
    if reviewer_password is None:
        log.warning("no QAREENA_REVIEWER_PASSWORD set; reviewer password for this run: %s", password)
    state = AppState(
        runtime=runtime,
        store=CaseStore(data / "cases.sqlite", key, retention_hours=runtime.policy.retention.image_hours),
        audit=AuditLog(data / "audit" / "audit.jsonl"),
        make_pipeline=make_pipeline,
        reviewer_password=password,
        kill_switch_path=data / "KILL_SWITCH",
    )
    state.store.purge()

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        state.worker.shutdown(wait=False)
        if state._pipeline is not None:
            state._pipeline.close()
        state.store.close()

    app = FastAPI(title="Qareena", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.q = state
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.globals.update(text=text, catalog=runtime.catalog)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.method == "POST" and not _same_origin(request):
            return Response("cross-site request refused", status_code=403)
        path = request.url.path
        if path.startswith(("/review", "/system")) and not _authorised(request, state.reviewer_password):
            return Response(
                "reviewer password required",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="qareena", charset="UTF-8"'},
            )
        response = await call_next(request)
        response.headers.update(
            {
                "Content-Security-Policy": CSP,
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "same-origin",  # not no-referrer: it makes browsers send Origin: null
                "Cache-Control": "no-store",
                "Permissions-Policy": "camera=(self), microphone=(), geolocation=()",
            }
        )
        return response

    # --- applicant -----------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "capture.html", {"errors": {}, "case": None})

    @app.post("/cases")
    async def new_case(request: Request) -> Response:
        client = request.client.host if request.client else "?"
        if not _allow(state, client):
            raise HTTPException(429, "too many submissions; try again later")
        form = await request.form()
        uploads: dict[str, bytes] = {}
        errors: dict[str, str] = {}
        for doc_type in runtime.catalog.documents:
            item = form.get(doc_type)
            if isinstance(item, FormFile) and item.filename:
                data_bytes = await item.read()
                try:
                    uploads[doc_type] = _reencode(data_bytes)
                except IntakeError as exc:
                    errors[doc_type] = text.RETAKE.get(exc.code, exc.code)
        if errors or not uploads:
            if not uploads and not errors:
                errors["_"] = "اختار صورة وحدة على الأقل."
            return templates.TemplateResponse(
                request, "capture.html", {"errors": errors, "case": None}, status_code=400
            )
        case_id = state.store.create()
        for doc_type, blob in uploads.items():
            state.store.put_image(case_id, doc_type, blob)
        state.audit.append("case_created", case=case_id, documents=sorted(uploads))
        _enqueue(state, case_id)
        return RedirectResponse(f"/cases/{case_id}", status_code=303)

    @app.post("/check/{doc_type}")
    async def quick_check(request: Request, doc_type: str, file: UploadFile) -> JSONResponse:
        """The quality gate alone (about a second), so the applicant can retake at once."""
        if doc_type not in runtime.catalog.documents:
            raise HTTPException(404)
        if not _allow(
            state, "check:" + (request.client.host if request.client else "?"), MAX_CHECKS_PER_HOUR
        ):
            raise HTTPException(429, "too many checks; try again later")
        try:
            img = load_image_bytes(await file.read()).image
        except IntakeError as exc:
            return JSONResponse({"ok": False, "message": text.RETAKE.get(exc.code, exc.code)})
        q = assess(img)
        return JSONResponse(
            {
                "ok": q.ok,
                "message": "عاشت إيدك! الصورة واضحة." if q.ok else text.RETAKE.get(q.reason or "", q.reason),
            }
        )

    @app.get("/cases/{case_id}", response_class=HTMLResponse)
    def case_page(request: Request, case_id: str) -> HTMLResponse:
        case = _case_or_404(state, case_id)
        return templates.TemplateResponse(request, "status.html", {"case": case, "errors": {}})

    @app.get("/cases/{case_id}/status")
    def case_status(case_id: str) -> JSONResponse:
        case = _case_or_404(state, case_id)
        return JSONResponse({"status": case.status, "outcome": case.outcome})

    @app.post("/cases/{case_id}/retake")
    async def retake(request: Request, case_id: str) -> Response:
        case = _case_or_404(state, case_id)
        form = await request.form()
        doc_type = str(form.get("doc_type", ""))
        item = form.get("file")
        if doc_type not in runtime.catalog.documents or not isinstance(item, FormFile):
            raise HTTPException(400, "document and photo required")
        try:
            blob = _reencode(await item.read())
        except IntakeError as exc:
            return templates.TemplateResponse(
                request,
                "status.html",
                {"case": case, "errors": {doc_type: text.RETAKE.get(exc.code, exc.code)}},
                status_code=400,
            )
        state.store.put_image(case_id, doc_type, blob)
        used = state.store.add_retake(case_id, doc_type)
        state.audit.append("retake_uploaded", case=case_id, document=doc_type, retakes_used=used)
        _enqueue(state, case_id)
        return RedirectResponse(f"/cases/{case_id}", status_code=303)

    # --- reviewer ------------------------------------------------------------------------

    @app.get("/review", response_class=HTMLResponse)
    def queue(request: Request) -> HTMLResponse:
        cases = state.store.recent(200)
        order = {"NEEDS_REVIEW": 0, "PRE_CHECKED": 1, "AWAITING_RETAKE": 2, None: 3}
        cases.sort(key=lambda c: (c.decision is not None, order.get(c.outcome, 3), -c.created))
        return templates.TemplateResponse(request, "queue.html", {"cases": cases})

    @app.get("/review/{case_id}", response_class=HTMLResponse)
    def workbench(request: Request, case_id: str) -> HTMLResponse:
        case = _case_or_404(state, case_id)
        return templates.TemplateResponse(request, "workbench.html", _workbench_context(case))

    @app.get("/review/{case_id}/image/{doc_type}")
    def image(case_id: str, doc_type: str) -> Response:
        blob = state.store.image(case_id, doc_type + READ_SUFFIX) or state.store.image(case_id, doc_type)
        if blob is None:
            raise HTTPException(404, "image deleted after the retention window, or never uploaded")
        return Response(blob, media_type="image/jpeg")

    @app.post("/review/{case_id}/field")
    def decide_field(
        case_id: str,
        doc_type: str = Form(...),
        field_name: str = Form(...),
        action: str = Form(...),
        value: str = Form(""),
        note: str = Form(""),
    ) -> Response:
        case = _case_or_404(state, case_id)
        if action not in ("confirm", "edit"):
            raise HTTPException(400)
        if action == "edit" and not note.strip():
            raise HTTPException(400, "a note is required for every correction")
        decision = case.decision or {"fields": {}}
        if decision.get("final"):
            raise HTTPException(409, "this file is already decided")
        decision.setdefault("fields", {})[f"{doc_type}.{field_name}"] = {
            "action": action,
            "value": value if action == "edit" else None,
            "note": note.strip()[:500],
            "at": time.time(),
        }
        state.store.decide(case_id, decision)
        state.audit.append(
            "field_" + action,
            case=case_id,
            field=f"{doc_type}.{field_name}",
            value_sha256=hashlib.sha256(value.encode("utf-8")).hexdigest() if action == "edit" else None,
        )
        return RedirectResponse(f"/review/{case_id}#f-{doc_type}-{field_name}", status_code=303)

    @app.post("/review/{case_id}/decision")
    def decide_case(case_id: str, verdict: str = Form(...), note: str = Form("")) -> Response:
        case = _case_or_404(state, case_id)
        ctx = _workbench_context(case)
        if verdict not in ("approve", "reject", "escalate"):
            raise HTTPException(400)
        if verdict == "approve" and ctx["unresolved"]:
            raise HTTPException(409, "resolve every flagged field before approving")
        if verdict != "approve" and not note.strip():
            raise HTTPException(400, "a note is required")
        decision = case.decision or {"fields": {}}
        decision["final"] = {"verdict": verdict, "note": note.strip()[:500], "at": time.time()}
        state.store.decide(case_id, decision)
        state.audit.append(
            "case_" + verdict, case=case_id, outcome=case.outcome, fingerprint=runtime.fingerprint
        )
        return RedirectResponse("/review", status_code=303)

    # --- system --------------------------------------------------------------------------

    @app.get("/system", response_class=HTMLResponse)
    def system(request: Request) -> HTMLResponse:
        from qareena.pipeline.scoring import certificate_path, load_certificate

        cert = load_certificate(runtime)
        return templates.TemplateResponse(
            request,
            "system.html",
            {
                "fingerprint": runtime.fingerprint,
                "policy": runtime.policy,
                "features": runtime.settings.features,
                "cert": cert,
                "cert_path": certificate_path(runtime),
                "kill_switch": state.kill_switch,
                "key_configured": runtime.settings.data_key is not None,
            },
        )

    @app.post("/system/kill-switch")
    def kill_switch(on: str = Form(...), note: str = Form("")) -> Response:
        if not note.strip():
            raise HTTPException(400, "a note is required")
        if on == "1":
            state.kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
            state.kill_switch_path.write_text(note.strip()[:500], encoding="utf-8")
        else:
            state.kill_switch_path.unlink(missing_ok=True)
        state.audit.append("kill_switch", on=on == "1", note=note.strip()[:500])
        return RedirectResponse("/system", status_code=303)

    return app


# --- helpers -----------------------------------------------------------------------------


def _reencode(data: bytes) -> bytes:
    """Intake checks, then a fresh JPEG: no metadata, no trailing payloads, upright."""
    img = load_image_bytes(data).image
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise IntakeError("CORRUPT_IMAGE", "re-encoding failed")
    return buf.tobytes()


def _process(state: AppState, case_id: str) -> None:
    case = state.store.get(case_id)
    if case is None:
        return
    state.store.set_status(case_id, "PROCESSING")
    try:
        images = {}
        for doc_type, blob in state.store.images(case_id).items():
            if not doc_type.endswith(READ_SUFFIX):
                images[doc_type] = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        pipe = state.pipeline()
        result = pipe.run(case_id, images, retakes_used=case.retakes, kill_switch=state.kill_switch)
        for doc_type, region in pipe.read_images().items():
            ok, buf = cv2.imencode(".jpg", region, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                state.store.put_image(case_id, doc_type + READ_SUFFIX, buf.tobytes())
        state.store.save_result(case_id, json.loads(result.model_dump_json()))
        state.audit.append(
            "case_read",
            case=case_id,
            outcome=result.outcome.value,
            reasons=result.reasons,
            fingerprint=result.fingerprint,
        )
    except Exception:  # a crash must never leave a case looking pre-checked
        log.exception("case %s failed", case_id)
        state.store.set_status(case_id, "FAILED")
        state.audit.append("case_failed", case=case_id)


def _enqueue(state: AppState, case_id: str) -> None:
    state.store.set_status(case_id, "QUEUED")
    state.worker.submit(_process, state, case_id)


def _allow(state: AppState, client: str, limit: int | None = None) -> bool:
    limit = MAX_CASES_PER_HOUR if limit is None else limit  # read at call time, not at definition
    now = time.time()
    with state._lock:
        recent = [t for t in state._recent.get(client, []) if now - t < 3600]
        if len(recent) >= limit:
            return False
        state._recent[client] = [*recent, now]
        return True


def _same_origin(request: Request) -> bool:
    source = request.headers.get("origin") or request.headers.get("referer")
    if not source:
        return False
    host = request.headers.get("host", "")
    return source.split("://", 1)[-1].split("/", 1)[0] == host


def _authorised(request: Request, password: str) -> bool:
    header = request.headers.get("authorization", "")
    if not header.startswith("Basic "):
        return False
    try:
        _, _, given = base64.b64decode(header[6:]).decode("utf-8").partition(":")
    except ValueError:
        return False
    return hmac.compare_digest(given.encode("utf-8"), password.encode("utf-8"))


def _case_or_404(state: AppState, case_id: str):
    if not case_id.isalnum() or len(case_id) > 32:
        raise HTTPException(404)
    case = state.store.get(case_id)
    if case is None:
        raise HTTPException(404)
    return case


def _workbench_context(case) -> dict[str, Any]:
    """Documents, fields and the flags a reviewer must resolve before approving."""
    result = case.result or {}
    decided = (case.decision or {}).get("fields", {})
    documents = []
    unresolved = 0
    for doc_type, doc in result.get("documents", {}).items():
        fields = []
        for name, rec in doc.get("fields", {}).items():
            key = f"{doc_type}.{name}"
            band = rec.get("band")
            flagged = band in ("check", "review")
            resolved = key in decided
            unresolved += flagged and not resolved
            fields.append({"name": name, "rec": rec, "flagged": flagged, "decision": decided.get(key)})
        size = doc.get("quality", {}).get("read_size") or [1, 1]
        documents.append({"type": doc_type, "doc": doc, "fields": fields, "size": size})
    return {
        "case": case,
        "result": result,
        "documents": documents,
        "unresolved": unresolved,
        "cross_checks": result.get("cross_checks", []),
    }
