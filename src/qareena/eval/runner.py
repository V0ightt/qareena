"""Run the real pipeline over every packet of a built split and keep every trace.

One JSON trace per packet (the full ``PacketResult``) plus ``run.json`` with the configuration
fingerprint, the split's manifest hash and the timing. Metrics are computed afterwards from
these files (``eval.metrics``), so one run serves every metric and every ablation. A run can
be resumed: packets that already have a trace are skipped.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qareena.config import Runtime
from qareena.pipeline.intake import IntakeError, load_image_bytes
from qareena.pipeline.run import OcrReader, Pipeline, VlmReader


def run_split(
    split_dir: Path,
    out_dir: Path,
    *,
    runtime: Runtime,
    vlm: VlmReader,
    ocr: OcrReader,
    limit: int | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    record = json.loads((split_dir / "split.json").read_text(encoding="utf-8"))
    today = dt.date.fromisoformat(record["today"])
    out_dir.mkdir(parents=True, exist_ok=True)
    packets = sorted(p for p in split_dir.iterdir() if p.is_dir())[:limit]
    started = time.perf_counter()
    pipe = Pipeline(runtime, vlm, ocr, today=today)
    done = 0
    try:
        for pdir in packets:
            trace = out_dir / f"{pdir.name}.json"
            if trace.exists():
                continue
            images, rejected = {}, {}
            for doc_type in runtime.catalog.documents:
                path = pdir / f"{doc_type}.png"
                if not path.exists():
                    continue
                try:
                    images[doc_type] = load_image_bytes(path.read_bytes()).image
                except IntakeError as exc:
                    rejected[doc_type] = exc.code
            t0 = time.perf_counter()
            result = pipe.run(pdir.name, images)
            data = json.loads(result.model_dump_json())
            data["intake_rejected"] = rejected
            data["wall_s"] = round(time.perf_counter() - t0, 3)
            trace.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8", newline="\n")
            done += 1
            progress(f"{pdir.name}: {result.outcome.value} in {data['wall_s']:.1f}s")
    finally:
        pipe.close()
    run = {
        "split": record["split"],
        "split_manifest_sha256": record["manifest_sha256"],
        "packets": len(packets),
        "new_traces": done,
        "fingerprint": runtime.fingerprint,
        "finished": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "wall_s": round(time.perf_counter() - started, 1),
    }
    (out_dir / "run.json").write_text(
        json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    return run
