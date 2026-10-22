"""Day-1 spike (ADR-006): can a local VLM on this laptop read our Arabic cards, with usable signals?

Renders N fictional SPECIMEN ID cards (front + back), makes one degraded copy of each
(blur / tilt / low JPEG / low light / glare that washes out one field), then runs every
candidate VLM and the CPU OCR engine on all images. Reports, per model and condition:
exact-field accuracy, character error rate, false fills (non-null where the field was washed
out), false nulls, JSON validity, MRZ check-digit pass rate, logprob usability and latency.

Usage:  uv run --group synth python tools/spike/day1.py --n 10
Results: var/spike/day1/{results.jsonl, summary.md}
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import statistics
from pathlib import Path

import cv2
import numpy as np

from qareena.checks.fields import garbled, label_echo
from qareena.checks.mrz import read_td1
from qareena.config import Settings, load_runtime
from qareena.eval.compare import score_field
from qareena.readers.ocr import ArabicOcr
from qareena.readers.ollama import OllamaClient
from qareena.readers.prompts import SYSTEM, field_schema, user_prompt
from qareena.synth import degrade
from qareena.synth.profiles import ISSUER, make_person
from qareena.synth.render import Renderer, write_ground_truth
from qareena.text import arabic

TEMPLATES = {
    "iq_national_card_front": "iq_national_card_front_a",
    "iq_national_card_back": "iq_national_card_back_a",
}
CONDITIONS = ["blur", "tilt", "jpeg_low", "low_light", "glare"]
REPLACEMENT_CHAR = chr(0xFFFD)
TODAY = dt.date(2026, 10, 22)


def ground_truth(doc_type: str, fields: list[str], vals: dict) -> dict:
    gt = {}
    for f in fields:
        if f.startswith("mrz_line_"):
            gt[f] = vals["mrz"][int(f[-1]) - 1]
        else:
            gt[f] = vals.get(f)
    return gt


def png_bytes(img: np.ndarray, long_side: int = 1600) -> bytes:
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def build_items(n: int, out: Path, runtime, seed: int) -> list[dict]:
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    items = []
    with Renderer() as r:
        for i in range(n):
            person = make_person(rng, today=TODAY)
            vals = person.as_dict() | {"issuing_authority": ISSUER}
            for side, (doc_type, tpl) in enumerate(TEMPLATES.items()):
                fields = list(runtime.catalog.documents[doc_type].fields)
                style = "ink_blue" if (i + side) % 2 == 0 else "ink_green"
                res = r.render(tpl, vals, out / f"{i:02d}_{doc_type}_clean.png", style=style)
                write_ground_truth(res)
                gt = ground_truth(doc_type, fields, vals)
                items.append(
                    {
                        "id": f"{i:02d}_{doc_type}_clean",
                        "doc_type": doc_type,
                        "condition": "clean",
                        "path": str(res.image),
                        "gt": gt,
                    }
                )
                img = cv2.imread(str(res.image))
                cond = CONDITIONS[(2 * i + side) % len(CONDITIONS)]
                gt_d = dict(gt)
                if cond == "blur":
                    deg = degrade.gaussian_blur(img, 2.2)
                elif cond == "tilt":
                    deg = degrade.perspective(img, nrng, max_shift=0.07)
                elif cond == "jpeg_low":
                    deg = degrade.jpeg(degrade.downscale(img, 0.45), 30)
                elif cond == "low_light":
                    deg = degrade.low_light(img, 0.42, 9.0, nrng)
                else:  # glare washes out one field -> expected null
                    candidates = [f for f, b in res.boxes.items() if b and not f.startswith("mrz")]
                    target = rng.choice(candidates)
                    b = res.boxes[target]
                    h, w = img.shape[:2]
                    box = (b[0] * w / 1000, b[1] * h / 1000, b[2] * w / 1000, b[3] * h / 1000)
                    deg, mask = degrade.glare_over(img, box)
                    if degrade.coverage(mask, box, threshold=0.9) >= 0.5:  # truly washed out
                        gt_d[target] = None
                path = out / f"{i:02d}_{doc_type}_{cond}.png"
                cv2.imwrite(str(path), deg)
                items.append(
                    {"id": path.stem, "doc_type": doc_type, "condition": cond, "path": str(path), "gt": gt_d}
                )
    return items


def run_vlm(client: OllamaClient, model: str, item: dict, runtime) -> dict:
    doc = runtime.catalog.documents[item["doc_type"]]
    img = cv2.imread(item["path"])
    res = client.chat_json(
        model=model, system=SYSTEM, user=user_prompt(doc), schema=field_schema(doc), images=[png_bytes(img)]
    )
    scores = {}
    labels = runtime.catalog.printed_labels()
    for f, spec in doc.fields.items():
        pred = res.data.get(f) if res.data else None
        s = score_field(item["gt"].get(f), pred, spec)
        # what the system keeps after its deterministic checks (label echo / garbled -> null)
        kept = None if pred is not None and (label_echo(str(pred), labels) or garbled(str(pred))) else pred
        s2 = score_field(item["gt"].get(f), kept, spec)
        scores[f] = {
            "exp": s.expected,
            "pred": s.predicted,
            "correct": s.correct,
            "cer": s.cer,
            "false_fill": s.false_fill,
            "false_null": s.false_null,
            "checked_correct": s2.correct,
            "checked_false_fill": s2.false_fill,
            "checked_false_null": s2.false_null,
        }
    mrz_valid = None
    if item["doc_type"] == "iq_national_card_back" and res.data:
        lines = [res.data.get(f"mrz_line_{k}") or "" for k in (1, 2, 3)]
        mrz_valid = read_td1(lines, today=TODAY)[0].valid
    tokens = [t.token for t in res.logprobs]
    return {
        "model": model,
        "id": item["id"],
        "doc_type": item["doc_type"],
        "condition": item["condition"],
        "json_ok": res.ok,
        "parse_error": res.parse_error,
        "scores": scores,
        "mrz_valid": mrz_valid,
        "metrics": res.metrics,
        "n_logprobs": len(tokens),
        "fffd_tokens": sum(REPLACEMENT_CHAR in t for t in tokens),
        "tokens_reconstruct": "".join(tokens) == res.content,
    }


def run_ocr(ocr: ArabicOcr, item: dict, runtime) -> dict:
    img = cv2.imread(item["path"])
    res = ocr.read(img)
    lines = [arabic.strict(line.text) for line in res.lines]
    joined = " ".join(lines)
    hits, reversed_hits, total = 0, 0, 0
    for f, v in item["gt"].items():
        if v is None or f.startswith("mrz"):
            continue
        spec = runtime.catalog.documents[item["doc_type"]].fields[f]
        target = arabic.strict(str(v))
        if spec.kind.value == "date":
            continue  # printed format differs from ISO ground truth; checked by the VLM path
        total += 1
        if target and target in joined:
            hits += 1
        elif target and target[::-1] in joined:
            reversed_hits += 1
    return {
        "id": item["id"],
        "condition": item["condition"],
        "seconds": res.seconds,
        "n_lines": len(res.lines),
        "hits": hits,
        "reversed_hits": reversed_hits,
        "total": total,
        "mean_score": statistics.fmean([ln.score for ln in res.lines]) if res.lines else 0.0,
    }


def summarize(vlm_rows: list[dict], ocr_rows: dict[str, list[dict]], vram: dict) -> str:
    out = ["# Day-1 spike results", "", f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}", ""]
    out += [
        "## VLM field extraction",
        "",
        "| model | condition | images | JSON ok | field exact | mean CER | false fills | false nulls | after checks: exact / false fills / nulls | MRZ valid | p50 s | p95 s | prompt tok |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in sorted({r["model"] for r in vlm_rows}):
        rows = [r for r in vlm_rows if r["model"] == model]
        for cond in ["clean", *CONDITIONS, "ALL"]:
            rs = rows if cond == "ALL" else [r for r in rows if r["condition"] == cond]
            if not rs:
                continue
            sc = [s for r in rs for s in r["scores"].values()]
            legible = [s for s in sc if s["exp"] is not None]
            unreadable = [s for s in sc if s["exp"] is None]
            mrz = [r["mrz_valid"] for r in rs if r["mrz_valid"] is not None]
            secs = sorted(r["metrics"]["wall_s"] for r in rs)
            p95 = secs[min(len(secs) - 1, round(0.95 * (len(secs) - 1)))]
            out.append(
                f"| {model} | {cond} | {len(rs)} | {sum(r['json_ok'] for r in rs)}/{len(rs)} | "
                f"{sum(s['correct'] for s in legible)}/{len(legible)} | "
                f"{statistics.fmean([s['cer'] for s in legible]):.3f} | "
                f"{sum(s['false_fill'] for s in unreadable)}/{len(unreadable)} | "
                f"{sum(s['false_null'] for s in legible)} | "
                f"{sum(s['checked_correct'] for s in legible)} / "
                f"{sum(s['checked_false_fill'] for s in unreadable)} / "
                f"{sum(s['checked_false_null'] for s in legible)} | "
                f"{(str(sum(mrz)) + '/' + str(len(mrz))) if mrz else '-'} | "
                f"{statistics.median(secs):.1f} | {p95:.1f} | "
                f"{statistics.fmean([r['metrics']['prompt_tokens'] for r in rs]):.0f} |"
            )
    out += [
        "",
        "## Logprobs",
        "",
        "| model | tokens/answer | answers with U+FFFD tokens | tokens rebuild content |",
        "|---|---|---|---|",
    ]
    for model in sorted({r["model"] for r in vlm_rows}):
        rs = [r for r in vlm_rows if r["model"] == model]
        out.append(
            f"| {model} | {statistics.fmean([r['n_logprobs'] for r in rs]):.0f} | "
            f"{sum(r['fffd_tokens'] > 0 for r in rs)}/{len(rs)} | {sum(r['tokens_reconstruct'] for r in rs)}/{len(rs)} |"
        )
    out += ["", "## VRAM residency (ollama ps)", ""]
    for model, info in vram.items():
        out.append(f"- {model}: {info}")
    out += [
        "",
        "## OCR (CPU) — ground-truth values found verbatim in OCR text",
        "",
        "| detector | condition | images | hit rate | reversed-order hits | mean s | mean line score |",
        "|---|---|---|---|---|---|---|",
    ]
    for det, rows in ocr_rows.items():
        for cond in ["clean", *CONDITIONS]:
            rs = [r for r in rows if r["condition"] == cond]
            if not rs:
                continue
            tot = sum(r["total"] for r in rs)
            out.append(
                f"| {det} | {cond} | {len(rs)} | {sum(r['hits'] for r in rs)}/{tot} | "
                f"{sum(r['reversed_hits'] for r in rs)} | {statistics.fmean([r['seconds'] for r in rs]):.2f} | "
                f"{statistics.fmean([r['mean_score'] for r in rs]):.2f} |"
            )
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--models", nargs="+", default=["qwen3.5:4b", "qwen3-vl:4b"])
    ap.add_argument("--out", type=Path, default=Path("var/spike/day1"))
    args = ap.parse_args()
    runtime = load_runtime(Settings(_env_file=None))
    args.out.mkdir(parents=True, exist_ok=True)
    items = build_items(args.n, args.out / "images", runtime, args.seed)
    print(f"built {len(items)} images", flush=True)

    ocr_rows: dict[str, list[dict]] = {}
    for det in ("mobile", "server"):
        ocr = ArabicOcr(detector=det)
        subset = items if det == "mobile" else [it for it in items if it["condition"] == "clean"]
        ocr_rows[det] = [run_ocr(ocr, it, runtime) for it in subset]
        print(f"OCR {det}: done", flush=True)

    vlm_rows, vram = [], {}
    results_path = args.out / "results.jsonl"
    with OllamaClient(timeout_s=300) as client, results_path.open("w", encoding="utf-8") as fh:
        for model in args.models:
            run_vlm(client, model, items[0], runtime)  # warm-up: load weights, exclude from timings
            vram[model] = [
                {k: m.get(k) for k in ("name", "size", "size_vram", "context_length")}
                for m in client.running()
            ]
            for it in items:
                row = run_vlm(client, model, it, runtime)
                vlm_rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            print(f"VLM {model}: done", flush=True)
    summary = summarize(vlm_rows, ocr_rows, vram)
    (args.out / "summary.md").write_text(summary, encoding="utf-8")
    (args.out / "ocr.json").write_text(json.dumps(ocr_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
