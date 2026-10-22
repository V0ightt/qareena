import pytest

from qareena.checks.fields import format_error, label_echo
from qareena.config import REPO_ROOT, FieldSpec, load_catalog

LABELS = load_catalog(REPO_ROOT / "config" / "documents.yaml").printed_labels()


@pytest.mark.parametrize(
    "value",
    [
        "دەسەڵاتی دەرکردن",  # the Kurdish label itself
        "دهسهلتي دهركردن",  # the Kurdish label re-spelled with Arabic letters (seen in the spike)
        "محل الولادة شوتني لهدايكبوين",  # Arabic label followed by a garbled Kurdish label
        "نموذج تجريبي",
        "SPECIMEN",
    ],
)
def test_label_echo_is_detected(value):
    assert label_echo(value, LABELS) is not None


@pytest.mark.parametrize("value", ["علي", "نور", "حسين", "مديرية البطاقة النموذجية", "الحلة", "زينب"])
def test_real_values_are_not_flagged(value):
    assert label_echo(value, LABELS) is None


NID = FieldSpec(label_ar="n", kind="digits", pattern=r"^\d{12}$")
SEX = FieldSpec(label_ar="s", kind="enum", values=("ذكر", "أنثى"))
NAME = FieldSpec(label_ar="n", kind="name_part")


def test_format_checks():
    assert format_error("٩٩١٢٣٤٥٦٧٨٩٠", NID) is None  # Arabic-Indic digits are fine
    assert format_error("99123456789", NID) == "FORMAT_INVALID"  # 11 digits (seen in the spike)
    assert format_error("99A234567890", NID) == "FORMAT_NOT_DIGITS"
    assert format_error("ذكر", SEX) is None
    assert format_error("male", SEX) == "VALUE_NOT_ALLOWED"
    assert format_error("علي", NAME) is None
    assert format_error("<script>", NAME) == "NAME_NOT_ARABIC_LETTERS"


@pytest.mark.parametrize("value", ["د.ه.س.ل.ت.ن", "ل.ه.د.ا.ي.ك.ب.و.و.ن", "د.م.ه.ل.ن"])
def test_spelled_out_glyphs_are_garbled(value):
    from qareena.checks.fields import garbled

    assert garbled(value)


@pytest.mark.parametrize("value", ["علي", "مديرية البطاقة النموذجية", "A+", "O+", "عبد الله"])
def test_normal_values_are_not_garbled(value):
    from qareena.checks.fields import garbled

    assert not garbled(value)


@pytest.mark.parametrize("value", ["دايك", "باوك", "ناو"])
def test_kurdish_labels_respelled_in_arabic_are_echoes(value):
    assert label_echo(value, LABELS) is not None


DATE_YMD = FieldSpec(label_ar="d", kind="date", date_format="%Y/%m/%d")
DATE_DMY = FieldSpec(label_ar="d", kind="date", date_format="%d/%m/%Y")


@pytest.mark.parametrize(
    ("value", "spec", "expected"),
    [
        ("٢٠٣٠/٠٣/٢٦", DATE_YMD, None),  # as printed, Arabic-Indic digits
        ("2030/3/26", DATE_YMD, None),
        ("31/12/2026", DATE_DMY, None),
        ("?1995", DATE_YMD, "DATE_UNPARSEABLE"),  # glare hid most of it (spike run 2)
        ("2030/02/30", DATE_YMD, "DATE_UNPARSEABLE"),
        ("2030-03-26", DATE_YMD, "DATE_FORMAT_DIFFERS"),  # ISO: taken from the MRZ, not the field
        ("31-12-2026", DATE_DMY, "DATE_FORMAT_DIFFERS"),
        ("2026/12/31", DATE_DMY, "DATE_UNPARSEABLE"),  # year first cannot be read day-first
    ],
)
def test_dates_must_be_whole_and_copied_as_printed(value, spec, expected):
    assert format_error(value, spec) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("ال", "NAME_TRUNCATED"), ("الع", "NAME_TRUNCATED"), ("الجبوري", None), ("عز", None)],
)
def test_a_name_part_is_more_than_the_article(value, expected):
    assert format_error(value, NAME) == expected
