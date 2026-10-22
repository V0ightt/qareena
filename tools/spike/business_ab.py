"""Input size × prompt for every document type, and how often the VLM reads RTL backwards.

The first end-to-end packets showed a failure the ID-card spikes never did: on the wider
business documents the VLM sometimes reads the page left to right, so Arabic comes back
character-reversed («داجس» for «سجاد») and the left/right columns swap (issue date <->
valid-until). The day-1 spike also never varied the input size. This measures both on clean
renders of every layout:

- sizes: long side of the image sent to the model (1600 / 1280 / 1024 px)
- prompts: v3 (production) and v3 + one sentence on right-to-left reading order

Usage:  uv run --group synth python tools/spike/business_ab.py --n 6 [--docs cards] [--out ...]
Results: <out>/{results.jsonl, summary.md}; re-running resumes where it stopped.
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
from qareena.readers.ollama import OllamaClient, OllamaError
from qareena.readers.prompts import SYSTEM, field_schema, json_keys, user_prompt
from qareena.synth.packets import make_packet
from qareena.synth.render import STYLES, Renderer
from qareena.text import arabic

TODAY = dt.date(2026, 10, 22)
BUSINESS = ("chamber_of_commerce_id", "tax_card")
CARDS = ("iq_national_card_front", "iq_national_card_back")
LAYOUTS = ("a", "b", "c")
SIZES = (1600, 1280, 1024)
RTL = (
    "Arabic is written right to left: give every value in normal reading order, starting from its "
    "first (rightmost) letter, exactly as a person would type it."
)
VARIANTS = {"v3": "", "v3_rtl": RTL}


def reversed_read(expected: object, predicted: object) -> bool:
    """The answer matches the expected text much better when read backwards."""
    if not expected or not predicted:
        return False
    e = arabic.spelling(str(expected)).replace(" ", "")
    raw = arabic.spelling(str(predicted))
    p = raw.replace(" ", "")
    if e == p or len(e) < 3 or not arabic.is_arabic_letters(e, allow=""):
        return False
    words_back = "".join(w[::-1] for w in raw.split())  # the form seen in practice
    back = max(
        Levenshtein.normalized_similarity(e, p[::-1]), Levenshtein.normalized_similarity(e, words_back)
    )
    return back > Levenshtein.normalized_similarity(e, p) + 0.2


def truth_of(doc_type: str, values: dict) -> dict:
    if doc_type == "iq_national_card_back":
        return values | {f"mrz_line_{k + 1}": values["mrz"][k] for k in range(3)}
    return values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--out", type=Path, default=Path("var/spike/business_ab"))
    ap.add_argument("--docs", choices=("business", "cards"), default="business")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = ap.parse_args()
    docs = BUSINESS if args.docs == "business" else CARDS
    rt = load_runtime(Settings(_env_file=None))
    rng = random.Random(args.seed)
    results = args.out / "results.jsonl"
    done = set()
    if results.exists():  # resume: keep the calls already made
        for line in results.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["id"], r["size"], r["variant"]))
    items = []
    with Renderer() as r:
        for i in range(args.n):
            packet = make_packet(f"B{i}", "clean", rng, today=TODAY)
            for layout in LAYOUTS:
                for doc_type in docs:
                    path = args.out / "images" / f"{i:02d}_{layout}_{doc_type}.png"
                    r.render(
                        f"{doc_type}_{layout}", packet.values[doc_type], path, style=sorted(STYLES)[i % 2]
                    )
                    items.append((i, layout, doc_type, path, truth_of(doc_type, packet.values[doc_type])))
    with OllamaClient(timeout_s=300) as client, results.open("a", encoding="utf-8") as fh:
        for i, layout, doc_type, path, truth in items:
            item_id = f"{i:02d}_{layout}_{doc_type}"
            doc = rt.catalog.documents[doc_type]
            keys = json_keys(doc)
            img = cv2.imread(str(path))
            for size in SIZES:
                png = encode_png(img, size)
                for variant in args.variants:
                    if (item_id, size, variant) in done:
                        continue
                    user = user_prompt(doc) + ("\n" + VARIANTS[variant] if VARIANTS[variant] else "")
                    error, data, wall = None, {}, None
                    try:
                        res = client.chat_json(
                            model=rt.settings.vlm_model,
                            system=SYSTEM,
                            user=user,
                            schema=field_schema(doc),
                            images=[png],
                            logprobs=False,
                        )
                        data = {keys[k]: v for k, v in (res.data or {}).items() if k in keys}
                        wall = res.metrics.get("wall_s")
                    except OllamaError as exc:  # e.g. "token repeat limit reached": a looping answer
                        error = str(exc)[-120:]
                    scores = {
                        f: {
                            "correct": score_field(truth.get(f), data.get(f), spec).correct,
                            "reversed": reversed_read(truth.get(f), data.get(f)),
                            "pred": data.get(f),
                        }
                        for f, spec in doc.fields.items()
                        if truth.get(f) is not None
                    }
                    swapped = "valid_until" in doc.fields and (
                        score_field(
                            truth.get("issue_date"), data.get("valid_until"), doc.fields["valid_until"]
                        ).correct
                        and score_field(
                            truth.get("valid_until"), data.get("issue_date"), doc.fields["issue_date"]
                        ).correct
                    )
                    row = {
                        "id": item_id,
                        "layout": layout,
                        "doc_type": doc_type,
                        "size": size,
                        "variant": variant,
                        "scores": scores,
                        "swapped_dates": swapped,
                        "wall_s": wall,
                        "error": error,
                    }
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
            print(f"{item_id} done", flush=True)
    rows = [json.loads(x) for x in results.read_text(encoding="utf-8").splitlines()]
    (args.out / "summary.md").write_text(summarize(rows), encoding="utf-8")
    print(summarize(rows))


def summarize(rows: list[dict]) -> str:
    out = [
        "# Input size × prompt (clean renders)",
        "",
        "| variant | size | images | exact | docs with a reversed field | swapped dates | model errors | p50 s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for variant in VARIANTS:
        for size in SIZES:
            rs = [r for r in rows if r["variant"] == variant and r["size"] == size]
            if not rs:
                continue
            sc = [s for r in rs for s in r["scores"].values()]
            walls = [r["wall_s"] for r in rs if r.get("wall_s")]
            out.append(
                f"| {variant} | {size} | {len(rs)} | {sum(s['correct'] for s in sc)}/{len(sc)} | "
                f"{sum(any(s['reversed'] for s in r['scores'].values()) for r in rs)} | "
                f"{sum(r['swapped_dates'] for r in rs)} | {sum(bool(r.get('error')) for r in rs)} | "
                f"{statistics.median(walls) if walls else 0:.1f} |"
            )
    out += [
        "",
        "## By layout and size (variant v3)",
        "",
        "| layout | doc | " + " | ".join(str(s) for s in SIZES) + " |",
        "|---|---|" + "---|" * len(SIZES),
    ]
    for layout in LAYOUTS:
        for doc in sorted({r["doc_type"] for r in rows}):
            cells = []
            for size in SIZES:
                rs = [
                    r
                    for r in rows
                    if r["layout"] == layout
                    and r["doc_type"] == doc
                    and r["size"] == size
                    and r["variant"] == "v3"
                ]
                sc = [s for r in rs for s in r["scores"].values()]
                cells.append(f"{sum(s['correct'] for s in sc)}/{len(sc)}" if sc else "-")
            out.append(f"| {layout} | {doc} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
