import time

import numpy as np

from qareena.config import Settings, load_runtime
from qareena.pipeline.loop import Budget, crop_near_label, label_box, needs_evidence, run_loop
from qareena.pipeline.records import DocumentRead, FieldRecord, Source
from qareena.readers.ocr import OcrLine

RT = load_runtime(Settings(_env_file=None))
DOC = "iq_national_card_front"
IMG = np.full((1080, 1712, 3), 240, np.uint8)
LINES = [
    {"text": "الجد باپیر", "score": 0.9, "box": [1500, 260, 1640, 290]},
    {"text": "الاسم ناو", "score": 0.9, "box": [1520, 120, 1640, 150]},
]


def record(value="حسين", critical=True, sources=()):
    rec = FieldRecord(doc_type=DOC, field="grandfather_name", raw=value, value=value, critical=critical)
    for s in sources:
        rec.add_source(s)
    rec.add_reason("NOT_GROUNDED")
    return rec


def doc_with(rec):
    return DocumentRead(doc_type=DOC, image_id="x", fields={"grandfather_name": rec}, ocr_lines=LINES)


def budget(n=2, seconds=60):
    return Budget(n, time.perf_counter() + seconds)


def test_label_is_found_by_arabic_or_kurdish_text():
    spec = RT.catalog.field(DOC, "grandfather_name")
    assert label_box(spec, LINES) == (1500, 260, 1640, 290)
    assert label_box(spec, [{"text": "باپیر", "score": 1, "box": [1, 2, 3, 4]}]) == (1, 2, 3, 4)


def test_crop_covers_the_area_below_and_beside_an_rtl_label():
    crop = crop_near_label(IMG, (1500, 260, 1640, 290))
    assert crop.shape[0] > 30
    assert crop.shape[1] > 140


def test_only_uncertain_critical_fields_need_evidence():
    assert needs_evidence(record())
    assert not needs_evidence(record(sources=[Source.MRZ]))
    assert not needs_evidence(record(critical=False))
    assert needs_evidence(record(sources=[Source.REREAD]))  # a same-model re-read is not independent


def test_reocr_match_adds_an_independent_source_and_stops():
    rec = record()
    calls = []
    actions = run_loop(
        doc_with(rec),
        IMG,
        RT.catalog,
        budget=budget(),
        reread=lambda *a: calls.append("reread") or "حسين",
        reocr=lambda img: [OcrLine("حسين", 0.95, ((0, 0), (9, 0), (9, 9), (0, 9)))],
    )
    assert Source.OCR in rec.sources
    assert "NOT_GROUNDED" not in rec.reasons
    assert [a["action"] for a in actions] == ["reocr"]
    assert calls == []  # no VLM time spent once independent evidence exists


def test_reread_agreement_is_a_signal_not_a_source_and_disagreement_is_flagged():
    agree = record()
    run_loop(
        doc_with(agree), IMG, RT.catalog, budget=budget(), reread=lambda *a: "حسين", reocr=lambda img: []
    )
    assert agree.sources == [Source.REREAD]
    assert needs_evidence(agree)
    differ = record()
    run_loop(
        doc_with(differ), IMG, RT.catalog, budget=budget(), reread=lambda *a: "حسن", reocr=lambda img: []
    )
    assert "REREAD_DISAGREES" in differ.reasons


def test_budget_is_respected():
    rec = record()
    actions = run_loop(
        doc_with(rec), IMG, RT.catalog, budget=budget(n=1), reread=lambda *a: "حسين", reocr=lambda img: []
    )
    assert len(actions) == 1
    late = record()
    assert (
        run_loop(
            doc_with(late),
            IMG,
            RT.catalog,
            budget=budget(seconds=-1),
            reread=lambda *a: "x",
            reocr=lambda img: [],
        )
        == []
    )
