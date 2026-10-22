"""Case storage: encrypted at rest, images deleted after the retention window.

One SQLite file holds cases and their document images. Images and pipeline results contain
personal data, so both are encrypted with Fernet (AES-128-CBC + HMAC-SHA256) under the key in
``QAREENA_DATA_KEY``. Without a configured key a random one is created once in the data
directory, which is fine for a demo laptop and is stated on the «النظام» page.

Images are deleted ``policy.retention.image_hours`` after upload (``purge``). The decision
record (outcome, reviewer decision) is kept: in production its retention follows ZainCash /
CBI record rules, not this demo setting.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
  id TEXT PRIMARY KEY,
  created REAL NOT NULL,
  updated REAL NOT NULL,
  status TEXT NOT NULL,          -- QUEUED | PROCESSING | DONE | FAILED
  outcome TEXT,                  -- PRE_CHECKED | NEEDS_REVIEW | AWAITING_RETAKE
  retakes TEXT NOT NULL,         -- JSON: doc_type -> retakes used
  decision TEXT,                 -- JSON: the reviewer's decision, once made
  result BLOB                    -- encrypted JSON PacketResult
);
CREATE TABLE IF NOT EXISTS images (
  case_id TEXT NOT NULL REFERENCES cases(id),
  doc_type TEXT NOT NULL,
  created REAL NOT NULL,
  data BLOB NOT NULL,            -- encrypted image bytes as uploaded (after intake checks)
  PRIMARY KEY (case_id, doc_type)
);
"""


@dataclass(frozen=True)
class CaseRecord:
    id: str
    created: float
    updated: float
    status: str
    outcome: str | None
    retakes: dict[str, int]
    decision: dict[str, Any] | None
    result: dict[str, Any] | None
    documents: tuple[str, ...]  # doc types with an image still stored


def load_or_create_key(configured: str | None, path: Path) -> bytes:
    if configured:
        key = configured.encode("ascii")
        Fernet(key)  # raises on a malformed key: fail at start-up, not on first upload
        return key
    if path.exists():
        return path.read_bytes().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    path.write_bytes(key)
    return key


class CaseStore:
    def __init__(self, path: Path, key: bytes, *, retention_hours: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(key)
        self._retention_s = retention_hours * 3600
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    # --- writes ---------------------------------------------------------------------------

    def create(self) -> str:
        case_id = uuid.uuid4().hex[:16]
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO cases (id, created, updated, status, retakes) VALUES (?, ?, ?, 'QUEUED', '{}')",
                (case_id, now, now),
            )
        return case_id

    def put_image(self, case_id: str, doc_type: str, data: bytes) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO images (case_id, doc_type, created, data) VALUES (?, ?, ?, ?)",
                (case_id, doc_type, time.time(), self._fernet.encrypt(data)),
            )

    def set_status(self, case_id: str, status: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE cases SET status = ?, updated = ? WHERE id = ?", (status, time.time(), case_id)
            )

    def save_result(self, case_id: str, result: dict[str, Any]) -> None:
        blob = self._fernet.encrypt(json.dumps(result, ensure_ascii=False).encode("utf-8"))
        with self._lock:
            self._db.execute(
                "UPDATE cases SET status = 'DONE', outcome = ?, result = ?, updated = ? WHERE id = ?",
                (result.get("outcome"), blob, time.time(), case_id),
            )

    def add_retake(self, case_id: str, doc_type: str) -> int:
        with self._lock:
            row = self._db.execute("SELECT retakes FROM cases WHERE id = ?", (case_id,)).fetchone()
            retakes = json.loads(row[0]) if row else {}
            retakes[doc_type] = retakes.get(doc_type, 0) + 1
            self._db.execute(
                "UPDATE cases SET retakes = ?, status = 'QUEUED', updated = ? WHERE id = ?",
                (json.dumps(retakes), time.time(), case_id),
            )
            return retakes[doc_type]

    def decide(self, case_id: str, decision: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE cases SET decision = ?, updated = ? WHERE id = ?",
                (json.dumps(decision, ensure_ascii=False), time.time(), case_id),
            )

    def purge(self, now: float | None = None) -> int:
        """Delete images past the retention window. Returns how many were deleted."""
        cutoff = (now or time.time()) - self._retention_s
        with self._lock:
            return self._db.execute("DELETE FROM images WHERE created < ?", (cutoff,)).rowcount

    # --- reads ----------------------------------------------------------------------------

    def get(self, case_id: str) -> CaseRecord | None:
        row = self._db.execute(
            "SELECT id, created, updated, status, outcome, retakes, decision, result FROM cases WHERE id = ?",
            (case_id,),
        ).fetchone()
        return None if row is None else self._record(row)

    def recent(self, limit: int = 100) -> list[CaseRecord]:
        rows = self._db.execute(
            "SELECT id, created, updated, status, outcome, retakes, decision, result FROM cases "
            "ORDER BY created DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._record(r) for r in rows]

    def image(self, case_id: str, doc_type: str) -> bytes | None:
        row = self._db.execute(
            "SELECT data FROM images WHERE case_id = ? AND doc_type = ?", (case_id, doc_type)
        ).fetchone()
        return None if row is None else self._decrypt(row[0])

    def images(self, case_id: str) -> dict[str, bytes]:
        rows = self._db.execute("SELECT doc_type, data FROM images WHERE case_id = ?", (case_id,)).fetchall()
        return {d: self._decrypt(b) for d, b in rows}

    def _record(self, row: tuple) -> CaseRecord:
        docs = tuple(
            r[0]
            for r in self._db.execute("SELECT doc_type FROM images WHERE case_id = ?", (row[0],)).fetchall()
        )
        return CaseRecord(
            id=row[0],
            created=row[1],
            updated=row[2],
            status=row[3],
            outcome=row[4],
            retakes=json.loads(row[5]),
            decision=json.loads(row[6]) if row[6] else None,
            result=json.loads(self._decrypt(row[7])) if row[7] else None,
            documents=docs,
        )

    def _decrypt(self, blob: bytes) -> bytes:
        try:
            return self._fernet.decrypt(blob)
        except InvalidToken as exc:  # wrong key or tampered row: never return garbage
            raise RuntimeError("stored data cannot be decrypted with the configured key") from exc
