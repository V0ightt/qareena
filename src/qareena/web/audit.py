"""Append-only, hash-chained audit log, separate from the case store.

Every entry holds the hash of the entry before it, so a deleted or edited line breaks the chain
(``verify``). Entries carry event names, case ids, reason codes and the configuration
fingerprint, never field values or images: the log can be kept and shared without personal data.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


class AuditLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._last = self._tail_hash()

    def append(self, event: str, **data: Any) -> str:
        with self._lock:
            entry = {"ts": round(time.time(), 3), "event": event, "data": data, "prev": self._last}
            body = json.dumps(entry, sort_keys=True, ensure_ascii=False)
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            with self.path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps({**entry, "hash": digest}, sort_keys=True, ensure_ascii=False) + "\n")
            self._last = digest
            return digest

    def _tail_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = GENESIS
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = json.loads(line)["hash"]
        return last


def verify(path: Path) -> list[str]:
    """Problems in the chain (empty when intact)."""
    problems, prev = [], GENESIS
    if not path.exists():
        return problems
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            claimed = entry.pop("hash")
            if entry.get("prev") != prev:
                problems.append(f"line {n}: chain broken (an entry before it was removed or reordered)")
            body = json.dumps(entry, sort_keys=True, ensure_ascii=False)
            if hashlib.sha256(body.encode("utf-8")).hexdigest() != claimed:
                problems.append(f"line {n}: entry was edited")
            prev = claimed
    return problems
