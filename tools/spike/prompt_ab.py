"""Prompt A/B (follow-up to the day-1 spike): how should we ask for fields so the model stops
returning labels?

Run 2 showed ``issuing_authority`` and ``place_of_birth`` returned as their Kurdish label 20/20
times, even with every label listed as never-a-value. The model appears to match the English
JSON key to the Kurdish label by meaning («دەسەڵاتی دەرکردن» means "issuing authority").
This compares request designs on clean renders of every layout, including the held-out one:

- v0  current: English keys, label list
- v1  Arabic printed labels as the JSON keys
- v2  English keys, each answer a {label, value} pair (find the label first, then its value)
- v3  v1 plus an explicit description of where the value sits relative to the bilingual label

Usage:  uv run --group synth python tools/spike/prompt_ab.py --n 6
Results: var/spike/prompt_ab/{results.jsonl, summary.md}
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import statistics
from pathlib import Path
from typing import Any

import cv2

from qareena.checks.fields import garbled, label_echo
from qareena.config import DocumentSpec, FieldKind, Settings, load_runtime
from qareena.eval.compare import score_field
from qareena.readers.ollama import OllamaClient
from qareena.readers.prompts import SYSTEM
from qareena.synth.profiles import ISSUER, make_person
from qareena.synth.render import Renderer

TODAY = dt.date(2026, 10, 22)
DOCS = ("iq_national_card_front", "iq_national_card_back")
LAYOUTS = ("a", "b", "c")
VARIANTS = ("v0", "v1", "v2", "v3")
MODEL = "qwen3.5:4b"

LOCATE = (
    "Each field on this document has a small label: an Arabic label, often followed by a smaller "
    "Kurdish translation of the same label (for example «ناو» next to «الاسم», «دەسەڵاتی دەرکردن» "
    "next to «جهة الإصدار»). Both words are the LABEL. The VALUE is the larger, bolder text printed "
    "beside or below that label pair. Return the value only."
)


def v0_schema(doc: DocumentSpec) -> dict[str, Any]:
    """The pre-A/B request (frozen here so this experiment stays reproducible): English keys."""
    return {
        "type": "object",
        "properties": {name: {"type": ["string", "null"]} for name in doc.fields},
        "required": list(doc.fields),
        "additionalProperties": False,
    }


def v0_prompt(doc: DocumentSpec) -> str:
    lines = [f"Document: {doc.label_ar}", "Fields to transcribe (JSON key: printed label):"]
    for name, spec in doc.fields.items():
        hint = {
            FieldKind.DATE: " (a date; copy it as printed)",
            FieldKind.MRZ_LINE: " (one line of the machine-readable zone, 30 characters of A-Z, 0-9 and <)",
            FieldKind.DIGITS: " (digits only)",
        }.get(spec.kind, "")
        lines.append(f"- {name}: {spec.label_ar}{hint}")
    printed = [lab for spec in doc.fields.values() for lab in (spec.label_ar, *spec.aliases)]
    lines.append(
        "These words are printed LABELS on the document (Arabic, and Kurdish in Arabic script). "
        "They are never values; do not return any of them: " + " ، ".join(printed)
    )
    lines.append("Return the JSON object now.")
    return "\n".join(lines)


def png(path: Path, long_side: int = 1600) -> bytes:
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def request(variant: str, doc: DocumentSpec) -> tuple[str, str, dict[str, Any], dict[str, str]]:
    """(system, user, schema, key -> field name)."""
    if variant == "v0":
        return SYSTEM, v0_prompt(doc), v0_schema(doc), {f: f for f in doc.fields}
    if variant in ("v1", "v3"):
        keymap = {spec.label_ar: name for name, spec in doc.fields.items()}
        schema = {
            "type": "object",
            "properties": {k: {"type": ["string", "null"]} for k in keymap},
            "required": list(keymap),
            "additionalProperties": False,
        }
        lines = [
            f"Document: {doc.label_ar}",
            "The JSON keys are the Arabic labels printed on the document. For each key, find that label "
            "(or the label with the same meaning) on the document and return the value printed with it.",
        ]
        for name, spec in doc.fields.items():
            if spec.kind.value == "date":
                lines.append(f"- «{spec.label_ar}» is a date; copy it as printed.")
            elif spec.kind.value == "mrz_line":
                lines.append(
                    f"- «{spec.label_ar}» is line {name[-1]} of the machine-readable zone (30 characters)."
                )
        if variant == "v3":
            lines.append(LOCATE)
        lines.append("Return the JSON object now.")
        return SYSTEM, "\n".join(lines), schema, keymap
    # v2: label/value pairs
    pair = {
        "type": "object",
        "properties": {"label": {"type": "string"}, "value": {"type": ["string", "null"]}},
        "required": ["label", "value"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {f: pair for f in doc.fields},
        "required": list(doc.fields),
        "additionalProperties": False,
    }
    lines = [
        f"Document: {doc.label_ar}",
        "For every field give two things: `label`, the label text exactly as printed for that field "
        "(Arabic and any Kurdish translation), then `value`, the text printed with that label. "
        "The value is never the label or part of it.",
        "Fields (JSON key: Arabic label):",
    ]
    lines += [f"- {name}: {spec.label_ar}" for name, spec in doc.fields.items()]
    lines.append("Return the JSON object now.")
    return SYSTEM, "\n".join(lines), schema, {f: f for f in doc.fields}


def values_of(data: dict | None, variant: str, keymap: dict[str, str]) -> dict[str, Any]:
    if not data:
        return {}
    out = {}
    for key, field in keymap.items():
        v = data.get(key)
        if variant == "v2" and isinstance(v, dict):
            v = v.get("value")
        out[field] = v
    return out


def build(n: int, out: Path, seed: int) -> list[dict]:
    rng = random.Random(seed)
    items = []
    with Renderer() as r:
        for i in range(n):
            person = make_person(rng, today=TODAY)
            vals = person.as_dict() | {"issuing_authority": ISSUER}
            for layout in LAYOUTS:
                for side, doc_type in enumerate(DOCS):
                    style = "ink_blue" if (i + side) % 2 == 0 else "ink_green"
                    path = out / f"{i:02d}_{layout}_{doc_type}.png"
                    r.render(f"{doc_type}_{layout}", vals, path, style=style)
                    gt = {k: vals.get(k) for k in vals}
                    gt |= {f"mrz_line_{k + 1}": vals["mrz"][k] for k in range(3)}
                    items.append(
                        {"id": path.stem, "layout": layout, "doc_type": doc_type, "path": path, "gt": gt}
                    )
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--out", type=Path, default=Path("var/spike/prompt_ab"))
    ap.add_argument("--summarize", action="store_true", help="only rebuild summary.md from results.jsonl")
    args = ap.parse_args()
    if args.summarize:
        rows = [json.loads(x) for x in (args.out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        (args.out / "summary.md").write_text(summarize(rows), encoding="utf-8")
        print(summarize(rows))
        return
    runtime = load_runtime(Settings(_env_file=None))
    labels = runtime.catalog.printed_labels()
    items = build(args.n, args.out / "images", args.seed)
    print(f"built {len(items)} images", flush=True)
    rows = []
    with (
        OllamaClient(timeout_s=300) as client,
        (args.out / "results.jsonl").open("w", encoding="utf-8") as fh,
    ):
        doc0 = runtime.catalog.documents[items[0]["doc_type"]]
        s, u, sch, _ = request("v0", doc0)
        client.chat_json(model=MODEL, system=s, user=u, schema=sch, images=[png(items[0]["path"])])  # warm-up
        for it in items:
            doc = runtime.catalog.documents[it["doc_type"]]
            image = png(it["path"])
            for variant in VARIANTS:
                s, u, sch, keymap = request(variant, doc)
                res = client.chat_json(
                    model=MODEL, system=s, user=u, schema=sch, images=[image], logprobs=False
                )
                vals = values_of(res.data, variant, keymap)
                scores = {}
                for f, spec in doc.fields.items():
                    pred = vals.get(f)
                    echo = pred is not None and label_echo(str(pred), labels) is not None
                    kept = None if pred is not None and (echo or garbled(str(pred))) else pred
                    scores[f] = {
                        "correct": score_field(it["gt"].get(f), pred, spec).correct,
                        "checked_correct": score_field(it["gt"].get(f), kept, spec).correct,
                        "echo": echo,
                        "pred": pred,
                        "exp": score_field(it["gt"].get(f), None, spec).expected,
                    }
                row = {
                    "id": it["id"],
                    "layout": it["layout"],
                    "doc_type": it["doc_type"],
                    "variant": variant,
                    "json_ok": res.ok,
                    "scores": scores,
                    "wall_s": res.metrics.get("wall_s"),
                    "output_tokens": res.metrics.get("output_tokens"),
                }
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            print(f"{it['id']} done", flush=True)
    (args.out / "summary.md").write_text(summarize(rows), encoding="utf-8")
    print(summarize(rows))


def summarize(rows: list[dict]) -> str:
    out = [
        "# Prompt A/B (clean renders)",
        "",
        "| variant | layout | images | JSON ok | exact | after checks | label echoes | issuing_authority | place_of_birth | p50 s | out tok |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for variant in VARIANTS:
        for layout in (*LAYOUTS, "ALL"):
            rs = [r for r in rows if r["variant"] == variant and (layout == "ALL" or r["layout"] == layout)]
            if not rs:
                continue
            sc = [s for r in rs for s in r["scores"].values() if s["exp"] is not None]

            def field(name: str, rs: list[dict] = rs) -> str:
                xs = [r["scores"][name]["correct"] for r in rs if name in r["scores"]]
                return f"{sum(xs)}/{len(xs)}" if xs else "-"

            out.append(
                f"| {variant} | {layout} | {len(rs)} | {sum(r['json_ok'] for r in rs)}/{len(rs)} | "
                f"{sum(s['correct'] for s in sc)}/{len(sc)} | {sum(s['checked_correct'] for s in sc)}/{len(sc)} | "
                f"{sum(bool(s['echo']) for s in sc)} | {field('issuing_authority')} | {field('place_of_birth')} | "
                f"{statistics.median(r['wall_s'] for r in rs):.1f} | "
                f"{statistics.fmean(r['output_tokens'] or 0 for r in rs):.0f} |"
            )
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
