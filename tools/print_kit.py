"""Print kit for the phone-photo tier (DESIGN §9.1, §11 step 6). Fictional SPECIMEN documents only.

Renders the photo packets of each split, lays every document out at its physical size on A4
pages with cut marks (ID-1 cards 85.6 x 54 mm; paper documents A5, 210 x 148 mm), and writes
a shot list: which phone, which condition, and for "finger" which field the finger goes over.

    split   packets  (DESIGN §9.1)
    dev        4
    calib      8
    test_a     4
    test_b    12   (one held-out layout per document type, like the digital Test-B)

People in these packets are new: none of them is in any digital split. Print batches are
per split, so a photo can never leak into another split.

Usage:  uv run python tools/print_kit.py [--out var/print]
Then:   print the PDFs at 100% scale ("actual size"), cut, laminate the ID cards,
        and shoot following shotlist.csv.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from qareena.synth.dataset import SPLITS, unique_packets
from qareena.synth.render import STYLES, Renderer

TODAY = dt.date(2026, 10, 22)
PHOTO_PACKETS = {"dev": 4, "calib": 8, "test_a": 4, "test_b": 12}
CONDITIONS = (
    "normal",
    "low light",
    "lamp glare",
    "angle 30-45 deg",
    "finger over a field",
    "cut corner",
    "cluttered table",
)
DPI = 300
MM = DPI / 25.4
A4 = (round(210 * MM), round(297 * MM))
CARD_MM = (85.6, 54.0)
PAPER_MM = (210.0, 148.0)
MARGIN = round(10 * MM)
GAP = round(8 * MM)
FONT = ImageFont.load_default(size=round(2.5 * MM))


def physical_size(img: Image.Image) -> tuple[float, float]:
    aspect = img.width / img.height
    return CARD_MM if abs(aspect - CARD_MM[0] / CARD_MM[1]) < 0.08 else PAPER_MM


def cut_marks(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> None:
    n = round(4 * MM)
    for x, y, dx, dy in ((x0, y0, -1, -1), (x1, y0, 1, -1), (x0, y1, -1, 1), (x1, y1, 1, 1)):
        draw.line([(x + dx * 2, y), (x + dx * n, y)], fill=0, width=2)
        draw.line([(x, y + dy * 2), (x, y + dy * n)], fill=0, width=2)


def layout(items: list[tuple[str, Path]], out: Path) -> None:
    """Shelf-pack the documents onto A4 pages at physical size; one multi-page PDF."""
    pages, page, draw = [], None, None
    x = y = shelf = 0

    def new_page() -> None:
        nonlocal page, draw, x, y, shelf
        page = Image.new("RGB", A4, "white")
        draw = ImageDraw.Draw(page)
        pages.append(page)
        x, y, shelf = MARGIN, MARGIN, 0

    new_page()
    # cards first (eight to a page), then paper documents (one to a page, sideways)
    items = sorted(items, key=lambda it: physical_size(Image.open(it[1])) != CARD_MM)
    for code, path in items:
        img = Image.open(path).convert("RGB")
        w_mm, h_mm = physical_size(img)
        w, h = round(w_mm * MM), round(h_mm * MM)
        if w > A4[0] - 2 * MARGIN:  # an A5 paper document goes on the sheet sideways
            img, (w, h) = img.rotate(90, expand=True), (h, w)
        if x + w > A4[0] - MARGIN:  # next shelf
            x, y, shelf = MARGIN, y + shelf + GAP, 0
        if y + h + round(5 * MM) > A4[1] - MARGIN:
            new_page()
        page.paste(img.resize((w, h), Image.LANCZOS), (x, y))
        cut_marks(draw, x, y, x + w, y + h)
        draw.text((x, y + h + round(1.5 * MM)), code, fill=(90, 90, 90), font=FONT)  # outside the cut line
        x += w + GAP
        shelf = max(shelf, h + round(5 * MM))
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=DPI)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("var/print"))
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()
    seen: set[str] = set()
    for spec in SPLITS.values():  # every digital person is taken already
        for _ in unique_packets(spec, args.seed, TODAY, seen):
            pass
    rows = []
    rng = random.Random(args.seed + 77)  # conditions, phones, styles
    with Renderer(scale=3.0) as r:
        for split, n in PHOTO_PACKETS.items():
            spec = dataclasses.replace(
                SPLITS[split], name=f"{split}-photo", packets=n, stream=SPLITS[split].stream + 10
            )
            d = args.out / split
            d.mkdir(parents=True, exist_ok=True)
            items, truth = [], {}
            for k, packet in enumerate(unique_packets(spec, args.seed, TODAY, seen)):
                for j, (doc_type, values) in enumerate(packet.values.items()):
                    layout_id = rng.choice(spec.layouts)
                    if split == "test_b" and j == k % 4:
                        layout_id = "c"  # each test_b packet has one document in the held-out layout
                    code = f"{packet.packet_id}/{doc_type}"
                    path = d / f"{packet.packet_id}__{doc_type}.png"
                    res = r.render(f"{doc_type}_{layout_id}", values, path, style=rng.choice(sorted(STYLES)))
                    items.append((code, path))
                    condition = CONDITIONS[(k * 4 + j) % len(CONDITIONS)]
                    finger = (
                        rng.choice(
                            [f for f, b in res.ink_boxes.items() if b and not f.startswith(("mrz", "_"))]
                        )
                        if "finger" in condition
                        else ""
                    )
                    rows.append(
                        {
                            "split": split,
                            "packet": packet.packet_id,
                            "document": doc_type,
                            "layout": layout_id,
                            "phone": "A" if (k + j) % 2 == 0 else "B",
                            "condition": condition,
                            "finger_over": finger,
                            "photo_file": f"{packet.packet_id}__{doc_type}.jpg",
                            "corners_clicked": "",
                        }
                    )
                truth[packet.packet_id] = {
                    "scenario": packet.scenario,
                    "defects": list(packet.defects),
                    "values": packet.values,
                }
            layout(items, d / f"print_{split}.pdf")
            (d / "truth.json").write_text(
                json.dumps(truth, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
            )
    with (args.out / "shotlist.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} documents; PDFs and shotlist.csv in {args.out}")


if __name__ == "__main__":
    main()
