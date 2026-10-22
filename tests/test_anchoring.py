"""Label-anchored grounding: values must be found where their own label puts them.

Boxes are taken from real OCR output on our SPECIMEN renders (tools/spike/anchor_eval.py).
"""

from qareena.config import FieldKind, Settings, load_runtime
from qareena.pipeline.grounding import ground_field, map_labels
from qareena.readers.ocr import OcrLine

RT = load_runtime(Settings(_env_file=None))
FRONT = RT.catalog.documents["iq_national_card_front"]
TAX = RT.catalog.documents["tax_card"]
CHAMBER = RT.catalog.documents["chamber_of_commerce_id"]


def ln(text, x0, y0, x1, y1, score=0.95):
    return OcrLine(text, score, ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))


# Layout A: two columns; a small Kurdish label beside the Arabic one; the value below them.
# The OCR detector split «عبد الزهرة» into two boxes and misread «الاسم» as «الاسه».
LAYOUT_A = [
    ln("باوك", 913, 262, 971, 302),
    ln("الأبـ", 979, 259, 1020, 299),
    ln("ناو", 1530, 268, 1563, 298),
    ln("الاسه", 1573, 260, 1649, 298),
    ln("الزهرة", 809, 300, 945, 369),
    ln("عبد", 938, 306, 1024, 367),
    ln("مصطفى", 1466, 292, 1653, 371),
    ln("نازناو", 894, 396, 952, 432),
    ln("اللقب", 964, 391, 1023, 428),
    ln("باپير", 1529, 400, 1571, 433),
    ln("الجد", 1586, 393, 1649, 431),
    ln("مرتضى", 1489, 430, 1652, 506),
    ln("الشمري", 866, 429, 1030, 508),
]

# Layout B: labels in a right-hand column (Kurdish above Arabic), values on the same row, left.
LAYOUT_B = [
    ln("ناو", 1237, 226, 1282, 265),
    ln("الاسم", 1199, 258, 1280, 302),
    ln("هدى", 886, 238, 978, 294),
    ln("باوك", 1225, 308, 1279, 342),
    ln("الأب", 1216, 330, 1282, 380),
    ln("سجاد", 866, 317, 979, 370),
    ln("باپير", 1230, 389, 1279, 430),
    ln("الجد", 1211, 415, 1282, 463),
    ln("أحمد", 883, 404, 966, 443),
]


def test_value_below_its_label_is_claimed_even_when_split_and_misread():
    lm = map_labels(FRONT, LAYOUT_A)
    assert "given_name" in lm.label_box  # «الاسه»: one misread letter is still the label
    g = ground_field("father_name", "عبد الزهرة", FieldKind.NAME_PART, LAYOUT_A, lm)
    assert g.exact  # both OCR pieces, joined in reading order
    assert g.anchored
    assert ground_field("given_name", "مصطفى", FieldKind.NAME_PART, LAYOUT_A, lm).anchored


def test_a_field_shift_is_caught_not_confirmed():
    """The day-1 failure: the VLM returned the grandfather's name as the father's."""
    lm = map_labels(FRONT, LAYOUT_A)
    g = ground_field("father_name", "مرتضى", FieldKind.NAME_PART, LAYOUT_A, lm)
    assert g.matches == 0
    assert g.foreign
    g = ground_field("grandfather_name", "الشمري", FieldKind.NAME_PART, LAYOUT_A, lm)  # laqab's value
    assert g.matches == 0
    assert g.foreign


def test_value_on_the_label_row_is_claimed_but_not_the_next_row():
    lm = map_labels(FRONT, LAYOUT_B)
    assert ground_field("given_name", "هدى", FieldKind.NAME_PART, LAYOUT_B, lm).anchored
    assert ground_field("father_name", "سجاد", FieldKind.NAME_PART, LAYOUT_B, lm).anchored
    assert ground_field("father_name", "أحمد", FieldKind.NAME_PART, LAYOUT_B, lm).foreign


def test_unknown_layout_falls_back_to_unclaimed_lines_and_says_so():
    lines = [ln("المسمى", 1300, 100, 1500, 140), ln("هدى", 900, 100, 1200, 140)]  # label unknown
    g = ground_field("given_name", "هدى", FieldKind.NAME_PART, lines, map_labels(FRONT, lines))
    assert g.exact
    assert not g.anchored  # a signal for the scorer, not independent evidence


def test_misread_kurdish_label_goes_to_the_longer_label_it_resembles():
    lines = [ln("باوى دايك", 815, 526, 923, 565), ln("أب الأم", 931, 516, 1026, 565)]
    lm = map_labels(FRONT, lines)
    assert lm.label_lines[0] == {"mother_father_name"}  # not «الأم» via its Kurdish «دایک»


def test_label_split_in_two_by_the_detector_is_recognised():
    lines = [ln("الدم", 881, 655, 950, 693), ln("فصيلة", 931, 652, 1021, 688), ln("A-", 954, 694, 1024, 749)]
    lm = map_labels(FRONT, lines)
    assert lm.label_lines.get(0) == lm.label_lines.get(1) == {"blood_type"}
    assert ground_field("blood_type", "A-", FieldKind.ENUM, lines, lm).anchored


def test_fuzzy_label_matching_rejects_a_different_word():
    # «اسم المحل» (shop name) is 2 edits from «اسم المكلف» (taxpayer): a different label
    lines = [ln("اسم المحل", 1736, 544, 1878, 586), ln("أسواق الشروق", 1159, 520, 1445, 601)]
    assert map_labels(TAX, lines).label_lines == {}
    # «غرفة» is part of a chamber's name, not the label «الغرفة»
    lines = [ln("غرفة", 1500, 100, 1600, 140)]
    assert map_labels(CHAMBER, lines).label_lines == {}


def test_rotated_watermark_fragments_are_transparent():
    tilted = OcrLine("PEC", 1.0, ((1171, 356), (1387, 262), (1417, 332), (1201, 425)))
    lines = [ln("الجد", 1586, 393, 1649, 431), tilted, ln("مرتضى", 1100, 380, 1300, 440)]
    lm = map_labels(FRONT, lines)
    assert 1 not in lm.owner  # never claimed
    assert ground_field("grandfather_name", "مرتضى", FieldKind.NAME_PART, lines, lm).anchored


def test_a_shorter_ocr_line_cannot_ground_a_longer_value():
    lines = [ln("اسم المكلف", 1700, 100, 1880, 140), ln("سجاد رعد كاظم", 1100, 95, 1500, 145)]
    g = ground_field(
        "taxpayer_name", "سجاد رعد كاظم الحسيني", FieldKind.FULL_NAME, lines, map_labels(TAX, lines)
    )
    assert g.matches == 0  # half a name is not the name


def test_arabic_read_backwards_is_recognised_at_its_label():
    """Seen on the business documents: «داجس عر مظا يدييزلا» for «سجاد رعد كاظم الزبيدي»."""
    from qareena.pipeline.grounding import reads_reversed

    lines = [ln("اسم المكلف", 1700, 100, 1880, 140), ln("سجاد رعد كاظم الزبيدي", 1100, 95, 1650, 150)]
    lm = map_labels(TAX, lines)
    assert reads_reversed("taxpayer_name", "داجس دعر مظاك يديبزلا", FieldKind.FULL_NAME, lines, lm)
    assert reads_reversed("taxpayer_name", "يديبزلا مظاك دعر داجس", FieldKind.FULL_NAME, lines, lm)
    assert not reads_reversed("taxpayer_name", "سجاد رعد كاظم الزبيدي", FieldKind.FULL_NAME, lines, lm)
    assert not reads_reversed("taxpayer_name", "علي حسين", FieldKind.FULL_NAME, lines, lm)  # just wrong
