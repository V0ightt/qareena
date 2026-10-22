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
