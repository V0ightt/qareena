"""The odd-input suite (DESIGN §9.1, §14): strange inputs the system must survive.

Each case is a fictional packet whose four slots hold ordinary clean documents, except one
slot that holds something odd. The pass criterion is the same for every case: **no crash,
never PRE_CHECKED, and a reason the applicant or reviewer can act on**.

Usage:
  uv run python tools/odd_inputs.py build            # writes var/odd/<case>/
  uv run python tools/odd_inputs.py run              # real readers; prints the table, writes report.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

from qareena.synth import degrade
from qareena.synth.packets import make_packet
from qareena.synth.render import Renderer

OUT = Path("var/odd")
TODAY = dt.date(2026, 10, 22)
SLOT = "tax_card"  # the slot that gets the odd input


def _photo(img: np.ndarray, rng: np.random.Generator, bg: np.ndarray | None = None) -> np.ndarray:
    h, w = img.shape[:2]
    alpha = np.full((h, w), 255, np.uint8)
    ink = np.zeros((h, w), np.uint8)
    bg = degrade.background(rng, (round(h * 1.45), round(w * 1.45))) if bg is None else bg
    photo, _, _ = degrade.place_on_background(img, alpha, ink, bg, rng, fill=0.7)
    return degrade.camera(photo, rng)


def build() -> None:
    rng = random.Random(4242)
    nrng = np.random.default_rng(4242)
    packet = make_packet("ODD", "clean", rng, today=TODAY)
    base = OUT / "_clean"
    base.mkdir(parents=True, exist_ok=True)
    with Renderer() as r:
        clean = {}
        for doc_type, values in packet.values.items():
            res = r.render(f"{doc_type}_a", values, base / f"{doc_type}.png")
            clean[doc_type] = cv2.imread(str(res.image))
        other = r.render(
            "other_document_a",
            {
                "name": packet.meta["owner"],
                "account": "4471209",
                "address": "بغداد، الكرادة",
                "amount": "87500",
            },
            base / "other.png",
        )
        other_img = cv2.imread(str(other.image))
    tax = clean[SLOT]
    front = clean["iq_national_card_front"]
    cases: dict[str, tuple[object, str]] = {}  # name -> (image or raw bytes, what should happen)
    cases["rotated_90"] = (cv2.rotate(_photo(tax, nrng), cv2.ROTATE_90_CLOCKWISE), "turned upright and read")
    cases["upside_down"] = (cv2.rotate(_photo(tax, nrng), cv2.ROTATE_180), "turned upright and read")
    two = degrade.background(nrng, (1600, 2600))
    two[150 : 150 + front.shape[0] // 2, 100 : 100 + front.shape[1] // 2] = cv2.resize(
        front, None, fx=0.5, fy=0.5
    )
    small_tax = cv2.resize(tax, None, fx=0.5, fy=0.5)
    two[700 : 700 + small_tax.shape[0], 1300 : 1300 + small_tax.shape[1]] = small_tax
    cases["two_documents"] = (degrade.camera(two, nrng), "review or retake; never pre-checked")
    white = np.full((round(tax.shape[0] * 1.45), round(tax.shape[1] * 1.45), 3), 245, np.uint8)
    cases["white_on_white"] = (_photo(tax, nrng, white), "read, or a retake")
    cases["unsupported_document"] = (_photo(other_img, nrng), "WRONG_DOCUMENT")
    cases["no_document"] = (degrade.background(nrng, (1400, 2000)), "WRONG_DOCUMENT or retake")
    face = degrade.background(nrng, (1400, 1100))
    cv2.ellipse(face, (550, 650), (260, 340), 0, 0, 360, (110, 140, 190), -1)
    cases["selfie"] = (face, "WRONG_DOCUMENT or retake")
    screen = _photo(tax, nrng)
    yy = np.arange(screen.shape[0])[:, None]
    screen = np.clip(screen.astype(np.float32) * (0.92 + 0.08 * np.sin(yy / 2.0))[..., None], 0, 255).astype(
        np.uint8
    )
    screen[:60] = 25  # a phone status bar
    cases["screenshot"] = (screen, "read like any photo (a soft flag is future work)")
    cases["whatsapp_crushed"] = (
        degrade.jpeg(degrade.downscale(_photo(tax, nrng), 0.35), 12),
        "retake or review",
    )
    cases["tiny"] = (cv2.resize(tax, (420, 265)), "TOO_SMALL")
    cases["blank_page"] = (np.full((1080, 1712, 3), 255, np.uint8), "WRONG_DOCUMENT or retake")
    cases["wrong_slot"] = (_photo(front, nrng), "WRONG_DOCUMENT (an ID card in the tax-card slot)")
    note = degrade.background(nrng, (1080, 1712))
    cv2.rectangle(note, (300, 200), (1400, 880), (240, 240, 240), -1)
    cases["instructions_only"] = (note, "WRONG_DOCUMENT; never acts on text")
    ok, png = cv2.imencode(".png", tax)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    cases["truncated_file"] = (png.tobytes()[: len(png) // 3], "CORRUPT_IMAGE at intake")
    cases["not_an_image"] = (b"%!PS-Adobe-3.0\n%% not an image\n", "UNSUPPORTED_FORMAT at intake")
    cases["pixel_bomb"] = (_png_bomb(), "TOO_MANY_PIXELS at intake")
    for name, (payload, expect) in cases.items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        for doc_type, img in clean.items():
            if doc_type != SLOT:
                cv2.imwrite(str(d / f"{doc_type}.png"), img)
        target = d / f"{SLOT}.png"
        if isinstance(payload, bytes):
            target.write_bytes(payload)
        else:
            cv2.imwrite(str(target), payload)
        (d / "case.json").write_text(
            json.dumps({"case": name, "slot": SLOT, "expect": expect}, ensure_ascii=False), encoding="utf-8"
        )
    print(f"built {len(cases)} cases under {OUT}")


def _png_bomb(side: int = 9000) -> bytes:
    """A tiny file that would decode to side x side pixels (81 MP, over the 40 MP cap)."""
    raw = zlib.compress(b"\x00" + b"\x00" * side, 9)  # one row; the decoder must refuse before inflating

    def chunk(t: bytes, data: bytes) -> bytes:
        return len(data).to_bytes(4, "big") + t + data + zlib.crc32(t + data).to_bytes(4, "big")

    ihdr = side.to_bytes(4, "big") * 2 + bytes([8, 0, 0, 0, 0])
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")


def run() -> int:
    from qareena.config import Settings, load_runtime
    from qareena.pipeline.intake import IntakeError, load_image_bytes
    from qareena.pipeline.run import OllamaVlmReader, Pipeline
    from qareena.readers.ocr import ArabicOcr
    from qareena.readers.ollama import OllamaClient

    rt = load_runtime(Settings())
    rows, failures = [], 0
    with OllamaClient(rt.settings.ollama_url, timeout_s=300) as client:
        pipe = Pipeline(rt, OllamaVlmReader(rt, client), ArabicOcr(), today=TODAY)
        try:
            for d in sorted(p for p in OUT.iterdir() if p.is_dir() and not p.name.startswith("_")):
                case = json.loads((d / "case.json").read_text(encoding="utf-8"))
                images, rejected = {}, {}
                for doc_type in rt.catalog.documents:
                    try:
                        images[doc_type] = load_image_bytes((d / f"{doc_type}.png").read_bytes()).image
                    except IntakeError as exc:
                        rejected[doc_type] = exc.code
                try:
                    result = pipe.run(d.name, images)
                    outcome = result.outcome.value
                    why = (
                        rejected.get(SLOT)
                        or result.retake.get(SLOT)
                        or ", ".join(result.documents.get(SLOT).reasons if result.documents.get(SLOT) else [])
                        or ", ".join(result.reasons[:3])
                    )
                    crashed = False
                except Exception as exc:  # the criterion is "no crash": record it, keep going
                    outcome, why, crashed = "CRASH", f"{type(exc).__name__}: {exc}", True
                passed = not crashed and outcome != "PRE_CHECKED" and bool(why)
                failures += not passed
                rows.append((d.name, case["expect"], outcome, why, "pass" if passed else "FAIL"))
                print(f"{'pass' if passed else 'FAIL'}  {d.name:22s} {outcome:16s} {why}", flush=True)
        finally:
            pipe.close()
    lines = [
        "# Odd-input suite",
        "",
        "| Case | Expected | Outcome | Reason | Result |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {c} | {e} | {f} |" for a, b, c, e, f in rows]
    lines.append(f"\n{len(rows) - failures}/{len(rows)} pass (no crash, never pre-checked, a reason given).")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("build", "run"))
    args = ap.parse_args()
    sys.exit(build() or 0 if args.action == "build" else run())
