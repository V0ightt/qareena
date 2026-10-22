"""Business documents (chamber ID, tax card): input size × prompt, and how often the VLM reads RTL backwards.

The first end-to-end packets showed a failure the ID-card spikes never did: on the wider
business documents the VLM sometimes reads the page left to right, so Arabic comes back
character-reversed («داجس» for «سجاد») and the left/right columns swap (issue date <->
valid-until). This measures it on clean renders of every layout:

- sizes: long side of the image sent to the model (1600 / 1280 / 1024 px)
- prompts: v3 (production) and v3 + one sentence on right-to-left reading order

Usage:  uv run --group synth python tools/spike/business_ab.py --n 6
Results: var/spike/business_ab/{results.jsonl, summary.md}
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import statistics
from pathlib import Path

import cv2
from rapidfuzz.distance import Levenshtein

from qareena.config import Settings, load_runtime
from qareena.eval.compare import score_field
from qareena.pipeline.run import encode_png
from qareena.readers.ollama import OllamaClient
from qareena.readers.prompts import SYSTEM, field_schema, json_keys, user_prompt
from qareena.synth.packets import make_packet
from qareena.synth.render import STYLES, Renderer
from qareena.text import arabic

TODAY = dt.date(2026, 10, 22)
DOCS = ("chamber_of_commerce_id", "tax_card")
LAYOUTS = ("a", "b", "c")
SIZES = (1600, 1280, 1024)
RTL = (
    "Arabic is written right to left: give every value in normal reading order, starting from its "
    "first (rightmost) letter, exactly as a person would type it."
)
VARIANTS = {"v3": "", "v3_rtl": RTL}


def reversed_read(expected: str | None, predicted: str | None) -> bool:
    """The answer matches the expected text much better when read backwards."""
    if not expected or not predicted:
        return False
    e = arabic.spelling(str(expected)).replace(" ", "")
    p = arabic.spelling(str(predicted)).replace(" ", "")
    if e == p or len(e) < 3 or not arabic.is_arabic_letters(e, allow=""):
        return False
    return Levenshtein.normalized_similarity(e, p[::-1]) > Levenshtein.normalized_similarity(e, p) + 0.2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--out", type=Path, default=Path("var/spike/business_ab"))
    args = ap.parse_args()
    rt = load_runtime(Settings(_env_file=None))
    rng = random.Random(args.seed)
    items = []
    with Renderer() as r:
        for i in range(args.n):
            packet = make_packet(f"B{i}", "clean", rng, today=TODAY)
            for layout in LAYOUTS:
                for doc_type in DOCS:
                    path = args.out / "images" / f"{i:02d}_{layout}_{doc_type}.png"
                    r.render(
                        f"{doc_type}_{layout}", packet.values[doc_type], path, style=sorted(STYLES)[i % 2]
                    )
                    items.append((i, layout, doc_type, path, packet.values[doc_type]))
    rows = []
    with (
        OllamaClient(timeout_s=300) as client,
        (args.out / "results.jsonl").open("w", encoding="utf-8") as fh,
    ):
        for i, layout, doc_type, path, truth in items:
            doc = rt.catalog.documents[doc_type]
            keys = json_keys(doc)
            img = cv2.imread(str(path))
            for size in SIZES:
                png = encode_png(img, size)
                for variant, extra in VARIANTS.items():
                    user = user_prompt(doc) + ("\n" + extra if extra else "")
                    res = client.chat_json(
                        model=rt.settings.vlm_model,
                        system=SYSTEM,
                        user=user,
                        schema=field_schema(doc),
                        images=[png],
                        logprobs=False,
                    )
                    data = {keys[k]: v for k, v in (res.data or {}).items() if k in keys}
                    scores = {
                        f: {
                            "correct": score_field(truth.get(f), data.get(f), spec).correct,
                            "reversed": reversed_read(truth.get(f), data.get(f)),
                            "pred": data.get(f),
                        }
                        for f, spec in doc.fields.items()
                        if truth.get(f) is not None
                    }
                    swapped = (
                        score_field(
                            truth.get("issue_date"), data.get("valid_until"), doc.fields["valid_until"]
                        ).correct
                        and score_field(
                            truth.get("valid_until"), data.get("issue_date"), doc.fields["issue_date"]
                        ).correct
                    )
                    row = {
                        "id": f"{i:02d}_{layout}_{doc_type}",
                        "layout": layout,
                        "doc_type": doc_type,
                        "size": size,
                        "variant": variant,
                        "scores": scores,
                        "swapped_dates": swapped,
                        "wall_s": res.metrics.get("wall_s"),
                    }
                    rows.append(row)
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
            print(f"{i:02d} {layout} {doc_type} done", flush=True)
    (args.out / "summary.md").write_text(summarize(rows), encoding="utf-8")
    print(summarize(rows))


def summarize(rows: list[dict]) -> str:
    out = [
        "# Business documents: size × prompt (clean renders)",
        "",
        "| variant | size | images | exact | docs with a reversed field | swapped dates | p50 s |",
        "|---|---|---|---|---|---|---|",
    ]
    for variant in VARIANTS:
        for size in SIZES:
            rs = [r for r in rows if r["variant"] == variant and r["size"] == size]
            if not rs:
                continue
            sc = [s for r in rs for s in r["scores"].values()]
            out.append(
                f"| {variant} | {size} | {len(rs)} | {sum(s['correct'] for s in sc)}/{len(sc)} | "
                f"{sum(any(s['reversed'] for s in r['scores'].values()) for r in rs)} | "
                f"{sum(r['swapped_dates'] for r in rs)} | {statistics.median(r['wall_s'] for r in rs):.1f} |"
            )
    out += [
        "",
        "## By layout (all sizes, all variants)",
        "",
        "| layout | doc | images | exact | reversed docs |",
        "|---|---|---|---|---|",
    ]
    for layout in LAYOUTS:
        for doc in DOCS:
            rs = [r for r in rows if r["layout"] == layout and r["doc_type"] == doc]
            sc = [s for r in rs for s in r["scores"].values()]
            out.append(
                f"| {layout} | {doc} | {len(rs)} | {sum(s['correct'] for s in sc)}/{len(sc)} | "
                f"{sum(any(s['reversed'] for s in r['scores'].values()) for r in rs)} |"
            )
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
