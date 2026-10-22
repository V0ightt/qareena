"""Can the VLM *verify* a candidate value it failed to *transcribe*? (spike, dev split only)

On dev, the OCR reading at a field's own label was right for 80 of the 117 names the VLM
misread (the VLM drops letters in runs of similar glyphs: «التميمي» -> «التيمي»), and a zoomed
re-read by the same VLM mostly repeats its own error. Verification is an easier task than
transcription, so ask the VLM, on a crop around the label: "is the value printed here exactly
«X»? yes / no", for the OCR candidate and for the VLM's own first answer.

The safety question is the false-yes rate on WRONG candidates: if it is not close to zero,
verification cannot be used as evidence. Measured on fields where the two readers disagree.

Usage:  uv run python tools/spike/verify_ab.py --run var/runs/dev --data var/data/dev
Results: var/spike/verify_ab/{results.jsonl, summary.md}
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from qareena.config import FieldKind, Settings, load_runtime
from qareena.eval.compare import score_field
from qareena.pipeline.grounding import map_labels
from qareena.pipeline.intake import load_image_bytes
from qareena.pipeline.loop import crop_near_label
from qareena.pipeline.quality import assess
from qareena.pipeline.run import encode_png
from qareena.readers.ocr import OcrLine
from qareena.readers.ollama import OllamaClient, OllamaError
from qareena.readers.prompts import SYSTEM

SCHEMA = {
    "type": "object",
    "properties": {"same": {"type": "boolean"}},
    "required": ["same"],
    "additionalProperties": False,
}


def ask(doc_label: str, label: str, candidate: str) -> str:
    return (
        f"This is a close-up crop of the document «{doc_label}» around the field «{label}».\n"
        f"Is the value printed in that field exactly «{candidate}», letter for letter? "
        "Compare every letter and every dot. Answer false if any letter differs, is missing or is "
        "extra, or if you cannot see the value clearly."
    )


def lines_of(doc: dict) -> list[OcrLine]:
    return [
        OcrLine(ln["text"], ln["score"], ((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])))
        for ln in doc.get("ocr_lines", [])
        for b in [ln["box"]]
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("var/runs/dev"))
    ap.add_argument("--data", type=Path, default=Path("var/data/dev"))
    ap.add_argument("--out", type=Path, default=Path("var/spike/verify_ab"))
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()
    rt = load_runtime(Settings(_env_file=None))
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (
        OllamaClient(timeout_s=120) as client,
        (args.out / "results.jsonl").open("w", encoding="utf-8") as fh,
    ):
        for trace_path in sorted(args.run.glob("dev-*.json")):
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            truth = json.loads((args.data / trace["packet_id"] / "truth.json").read_text(encoding="utf-8"))
            for doc_type, doc in trace["documents"].items():
                if not doc.get("fields"):
                    continue
                spec_doc = rt.catalog.documents[doc_type]
                lines = lines_of(doc)
                labels = map_labels(spec_doc, lines)
                region = None
                for name, rec in doc["fields"].items():
                    spec = spec_doc.fields[name]
                    tf = truth["documents"][doc_type]["fields"].get(name, {})
                    if spec.kind in (FieldKind.MRZ_LINE, FieldKind.DATE) or tf.get("state") != "legible":
                        continue
                    claimed = labels.claims.get(name, ())
                    ocr = " ".join(lines[j].text for j in claimed) if claimed else None
                    vlm = rec.get("raw")
                    if not ocr or not vlm or score_field(ocr, vlm, spec).correct:
                        continue  # only fields where the two readers disagree
                    if name not in labels.label_box:
                        continue
                    if region is None:
                        img = load_image_bytes(
                            (args.data / trace["packet_id"] / f"{doc_type}.png").read_bytes()
                        ).image
                        q = assess(img)
                        region = q.rectified if q.rectified is not None else img
                    crop = encode_png(crop_near_label(region, labels.label_box[name]), 800)
                    expected = truth["values"][doc_type].get(name)
                    for source, cand in (("ocr", ocr), ("vlm", vlm)):
                        try:
                            res = client.chat_json(
                                model=rt.settings.vlm_model,
                                system=SYSTEM,
                                user=ask(spec_doc.label_ar, spec.label_ar, cand),
                                schema=SCHEMA,
                                images=[crop],
                                logprobs=False,
                                num_predict=32,
                            )
                            said = bool((res.data or {}).get("same"))
                        except OllamaError:
                            said = None
                        row = {
                            "packet": trace["packet_id"],
                            "field": f"{doc_type}.{name}",
                            "kind": spec.kind.value,
                            "source": source,
                            "candidate": cand,
                            "correct": score_field(expected, cand, spec).correct,
                            "said_same": said,
                        }
                        rows.append(row)
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fh.flush()
                    if len(rows) >= args.limit:
                        break
    c = Counter((r["correct"], r["said_same"]) for r in rows)
    yes_right = c[(True, True)]
    right = yes_right + c[(True, False)]
    yes_wrong = c[(False, True)]
    wrong = yes_wrong + c[(False, False)]
    summary = (
        "# Verify, not transcribe (dev, fields where VLM and OCR disagree)\n\n"
        f"- candidates: {len(rows)}\n"
        f"- yes on a correct candidate: {yes_right}/{right}\n"
        f"- **yes on a wrong candidate: {yes_wrong}/{wrong}** (must be ~0 to count as evidence)\n"
        f"- errors: {c[(True, None)] + c[(False, None)]}\n"
    )
    (args.out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
