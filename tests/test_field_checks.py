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
