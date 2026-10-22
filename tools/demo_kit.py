"""Build the files for the 3-minute demo (DESIGN §16). Fictional SPECIMEN documents only.

  1-clean/                      four photos of a consistent packet
  2-glare/tax_card_glare.png    a tax-card photo with glare (should get the Iraqi-Arabic retake message)
  2-glare/tax_card_retake.png   the retake of the same card
  3-conflict-injection/         the grandfather's name differs between documents, and the tax
                                card carries a printed instruction to approve the file

Usage:  uv run python tools/demo_kit.py [--out var/demo]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

import cv2
import numpy as np

from qareena.synth import degrade
from qareena.synth.dataset import capture
from qareena.synth.packets import INJECTION_TEXT, make_packet
from qareena.synth.render import Renderer

TODAY = dt.date(2026, 10, 22)


def photo(r: Renderer, doc_type: str, values: dict, path: Path, seed: int, condition: str = "photo") -> dict:
    rendered = r.render(f"{doc_type}_a", values, path.with_suffix(".render.png"), style="ink_blue")
    rng = np.random.default_rng(seed)
    cap = capture(rendered, condition, rng, np.random.default_rng(seed + 1))
    cv2.imwrite(str(path), cap.image)
    rendered.image.unlink()
    return {"condition": condition, "expected_quality": cap.expected_quality}


def glare_photo(r: Renderer, values: dict, path: Path, seed: int) -> None:
    """A big hotspot over the middle of the card: clearly a retake."""
    rendered = r.render("tax_card_a", values, path.with_suffix(".render.png"), style="ink_blue")
    rng = np.random.default_rng(seed)
    cap = capture(rendered, "photo", rng, np.random.default_rng(seed + 1))
    h, w = cap.image.shape[:2]
    img, _ = degrade.glare_over(cap.image, (w * 0.35, h * 0.3, w * 0.7, h * 0.55), strength=1.0, pad=0.6)
    cv2.imwrite(str(path), img)
    rendered.image.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("var/demo"))
    args = ap.parse_args()
    notes = {}
    with Renderer() as r:
        clean = make_packet("DEMO-1", "clean", random.Random(101), today=TODAY)
        d = args.out / "1-clean"
        d.mkdir(parents=True, exist_ok=True)
        for i, (doc_type, values) in enumerate(clean.values.items()):
            notes[f"1-clean/{doc_type}"] = photo(r, doc_type, values, d / f"{doc_type}.png", 10 + i)
        d = args.out / "2-glare"
        d.mkdir(parents=True, exist_ok=True)
        glare_photo(r, clean.values["tax_card"], d / "tax_card_glare.png", 30)
        notes["2-glare/tax_card_retake"] = photo(
            r, "tax_card", clean.values["tax_card"], d / "tax_card_retake.png", 31
        )
        conflict = make_packet("DEMO-3", "grandfather_conflict", random.Random(103), today=TODAY)
        conflict.values["tax_card"]["injection"] = INJECTION_TEXT
        d = args.out / "3-conflict-injection"
        d.mkdir(parents=True, exist_ok=True)
        for i, (doc_type, values) in enumerate(conflict.values.items()):
            notes[f"3-conflict-injection/{doc_type}"] = photo(
                r, doc_type, values, d / f"{doc_type}.png", 50 + i
            )
        notes["3-conflict-injection/what"] = {
            "front grandfather": conflict.values["iq_national_card_front"]["grandfather_name"],
            "chamber member name": conflict.values["chamber_of_commerce_id"]["member_name"],
            "printed instruction": INJECTION_TEXT,
        }
    (args.out / "README.json").write_text(json.dumps(notes, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"demo kit written to {args.out}")


if __name__ == "__main__":
    main()
