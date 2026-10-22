import datetime as dt

from qareena.config import FieldSpec
from qareena.eval.compare import canonical, parse_date, score_field

DATE_YMD = FieldSpec(label_ar="t", kind="date", date_format="%Y/%m/%d")
DATE_DMY = FieldSpec(label_ar="t", kind="date", date_format="%d/%m/%Y")
DIGITS = FieldSpec(label_ar="n", kind="digits")
NAME = FieldSpec(label_ar="n", kind="name_part")
MRZ = FieldSpec(label_ar="m", kind="mrz_line")


def test_dates_parse_with_arabic_indic_digits_and_document_order():
    assert parse_date("٢٠٢٤/٠١/٠٢", "%Y/%m/%d") == dt.date(2024, 1, 2)
    assert parse_date("02/01/2024", "%d/%m/%Y") == dt.date(2024, 1, 2)
    assert parse_date("2024/13/02", "%Y/%m/%d") is None
    assert canonical("2024/01/02", DATE_YMD) == canonical(dt.date(2024, 1, 2), DATE_YMD)


def test_unparseable_date_is_never_correct():
    s = score_field("2024-01-02", "around 2024", DATE_YMD)
    assert not s.correct


def test_digits_ignore_spaces_and_digit_script():
    assert score_field("991234567890", "٩٩ 1234 567890", DIGITS).correct


def test_names_require_exact_transcription():
    assert score_field("فاطمة", "فاطمة", NAME).correct
    assert not score_field("فاطمة", "فاطمه", NAME).correct  # a spelling variant is a reading error
    assert score_field("فاطمة", "فاطمة" + chr(0x200F), NAME).correct  # invisible marks are not


def test_null_semantics():
    fill = score_field(None, "علي", NAME)
    assert fill.false_fill
    assert not fill.correct
    null_ok = score_field(None, None, NAME)
    assert null_ok.correct
    assert not null_ok.false_fill
    miss = score_field("علي", None, NAME)
    assert miss.false_null
    assert miss.cer == 1.0


def test_mrz_whitespace_is_ignored():
    assert score_field("ID<UTO<<", " id<uto << ", MRZ).correct
