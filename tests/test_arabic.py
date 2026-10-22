from hypothesis import given
from hypothesis import strategies as st

from qareena.text import arabic

ALEF, ALEF_HAMZA_ABOVE, ALEF_HAMZA_BELOW, ALEF_MADDA = "ا", "أ", "إ", "آ"
TA_MARBUTA, HA, ALEF_MAQSURA, YA = "ة", "ه", "ى", "ي"


def test_digits_arabic_indic_and_persian_to_ascii():
    assert arabic.normalize_digits("١٢٣") == "123"  # ١٢٣
    assert arabic.normalize_digits("۴۵۶") == "456"  # ۴۵۶ (Persian forms)
    assert arabic.normalize_digits("AB٠x") == "AB0x"


def test_strict_removes_invisible_differences_only():
    plain = "محمد"
    with_marks = "مُحَمَّد"
    with_tatweel = "محــمد"
    with_controls = "\u200f" + "محمد" + "\u200e\u200b"
    for variant in (with_marks, with_tatweel, with_controls):
        assert arabic.strict(variant) == plain
    assert arabic.strict("  علي \t  حسين ") == "علي حسين"


def test_strict_does_not_fold_letters():
    # spelling differences must survive the strict layer
    assert arabic.strict(ALEF_HAMZA_ABOVE + "حمد") != arabic.strict(ALEF + "حمد")
    assert arabic.strict("فاطمة") != arabic.strict("فاطمه")


def test_strict_folds_presentation_forms():
    # U+FEE3 U+FEA4 U+FEE4 U+FEAA are presentation forms of م ح م د
    assert arabic.strict("ﻣﺤﻤﺪ") == "محمد"


def test_spelling_folds():
    for v in (ALEF_HAMZA_ABOVE, ALEF_HAMZA_BELOW, ALEF_MADDA, "ٱ"):
        assert arabic.spelling(v + "حمد") == ALEF + "حمد"
    assert arabic.spelling("فاطم" + TA_MARBUTA) == "فاطم" + HA
    assert arabic.spelling("مصطف" + ALEF_MAQSURA) == "مصطف" + YA
    assert arabic.spelling("کریم") == "كريم"  # Persian kaf/ya


def test_sound_folds_are_separate_from_spelling():
    gaf, qaf = "گاطع", "قاطع"  # گاطع / قاطع
    assert arabic.spelling(gaf) != arabic.spelling(qaf)
    assert arabic.sound(gaf) == arabic.sound(qaf)


def test_sorani_detection():
    assert arabic.has_sorani("ڕەش")  # ڕەش
    assert not arabic.has_sorani("محمد")


def test_is_arabic_letters():
    assert arabic.is_arabic_letters("علي حسين")
    assert not arabic.is_arabic_letters("Ali")
    assert not arabic.is_arabic_letters("علي1")
    assert not arabic.is_arabic_letters("   ")
    assert not arabic.is_arabic_letters("<img src=x>")


_arabic_text = st.text(
    alphabet=st.characters(min_codepoint=0x0600, max_codepoint=0x06FF) | st.sampled_from(" \u200fـ"),
    max_size=40,
)


@given(_arabic_text)
def test_layers_are_idempotent(s):
    for f in (arabic.strict, arabic.spelling, arabic.sound):
        assert f(f(s)) == f(s)


@given(_arabic_text)
def test_layers_are_monotone(s):
    # anything equal at a stricter layer stays equal at a looser one
    assert arabic.spelling(arabic.strict(s)) == arabic.spelling(s)
    assert arabic.sound(arabic.spelling(s)) == arabic.sound(s)
