"""Does label-anchored grounding find the right OCR line for each field? (no VLM involved)

Renders fictional packets in every layout, runs the OCR reader, maps labels to value lines
(``pipeline.grounding.map_labels``), and compares each field's claimed lines with the true value
box from the DOM. Reports, per layout and document:

- label found:  the field's printed label was located in the OCR output
- claimed right: the OCR line(s) lying on the true value box are claimed by that field
- stolen:        a true value line is claimed by a *different* field (the dangerous case)

Usage:  uv run --group synth python tools/spike/anchor_eval.py --n 4
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import random
import sys
from pathlib import Path

import cv2

from qareena.config import Settings, load_runtime
from qareena.pipeline.grounding import _noise, map_labels
from qareena.readers.ocr import ArabicOcr
from qareena.synth.packets import make_packet
from qareena.synth.render import STYLES, Renderer

TODAY = dt.date(2026, 10, 22)


def lines_on(box_px, lines):
    x0, y0, x1, y1 = box_px
    pad = 0.15 * (y1 - y0)
    out = []
    for i, ln in enumerate(lines):
        bx0, by0, bx1, by1 = ln.bbox
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        if x0 - pad <= cx <= x1 + pad and y0 - pad <= cy <= y1 + pad:
            out.append(i)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("var/spike/anchor"))
    args = ap.parse_args()
    rt = load_runtime(Settings(_env_file=None))
    rng = random.Random(args.seed)
    ocr = ArabicOcr()
    stats = collections.defaultdict(collections.Counter)
    problems = []
    with Renderer() as r:
        for p in range(args.n):
            packet = make_packet(f"A{p}", "clean", rng, today=TODAY)
            for layout in ("a", "b", "c"):
                for doc_type, values in packet.values.items():
                    path = args.out / f"{p}_{layout}_{doc_type}.png"
                    res = r.render(f"{doc_type}_{layout}", values, path, style=sorted(STYLES)[p % 2])
                    img = cv2.imread(str(path))
                    h, w = img.shape[:2]
                    lines = ocr.read(img).lines
                    lm = map_labels(rt.catalog.documents[doc_type], lines)
                    key = (layout, doc_type)
                    for name in rt.catalog.documents[doc_type].fields:
                        b = res.boxes.get(name)
                        if not b:
                            continue
                        st = stats[key]
                        st["fields"] += 1
                        found = name in lm.label_box
                        st["label_found"] += found
                        box = (b[0] * w / 1000, b[1] * h / 1000, b[2] * w / 1000, b[3] * h / 1000)
                        # DOM boxes are font-metric tall and can reach the field's own label: skip it
                        # and stray glyphs, keep anything else (another field's label in the box is a bug)
                        truth = [
                            i
                            for i in lines_on(box, lines)
                            if not _noise(lines[i]) and name not in lm.label_lines.get(i, ())
                        ]
                        if not truth:
                            st["no_ocr_line"] += 1
                            continue
                        allowed, _ = lm.eligible(name)
                        ok = all(i in allowed for i in truth)
                        st["claimed_right"] += ok
                        stolen = [lm.owner.get(i) or ",".join(lm.label_lines.get(i, ())) for i in truth]
                        stolen = [s for s in stolen if s and s != name and name not in s.split(",")]
                        st["stolen"] += bool(stolen)
                        if not ok or stolen:
                            problems.append(
                                f"{layout} {doc_type}.{name}: found={found} truth={[lines[i].text for i in truth]} "
                                f"claimed={[lines[i].text for i in lm.claims.get(name, ())]} stolen_by={stolen}"
                            )
    print("| layout | document | fields | label found | claimed right | stolen | no OCR line |")
    print("|---|---|---|---|---|---|---|")
    for (layout, doc), st in sorted(stats.items()):
        print(
            f"| {layout} | {doc} | {st['fields']} | {st['label_found']} | {st['claimed_right']} | "
            f"{st['stolen']} | {st['no_ocr_line']} |"
        )
    print()
    for line in problems:
        print(line)


if __name__ == "__main__":
    main()
