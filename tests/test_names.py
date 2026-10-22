import pytest

from qareena.text.names import Match, compare_laqab, compare_names, compare_part, tokens


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("حسين", "حسين", Match.SAME),
        ("حُسَين", "حسين", Match.SAME),  # diacritics are invisible differences
        ("فاطمة", "فاطمه", Match.VARIANT),
        ("أحمد", "احمد", Match.VARIANT),
        ("مصطفى", "مصطفي", Match.VARIANT),
        ("گاطع", "قاطع", Match.SOUND_ALIKE),
        ("حسين", "حسن", Match.CONFLICT),
        ("علي", "عمر", Match.CONFLICT),
    ],
)
def test_compare_part(a, b, expected):
    assert compare_part(a, b) == expected
    assert compare_part(b, a) == expected  # symmetric


def test_compound_names_are_one_part_and_spacing_is_a_variant():
    assert tokens("عبد الله علي حسين") == ["عبد الله", "علي", "حسين"]
    assert tokens("نور الدين كاظم") == ["نور الدين", "كاظم"]
    assert tokens("أبو بكر محمد") == ["أبو بكر", "محمد"]
    assert compare_part("عبد الله", "عبدالله") == Match.VARIANT


def test_laqab_definite_article_is_optional():
    assert compare_laqab("الجبوري", "جبوري") == Match.VARIANT
    assert compare_laqab("الجبوري", "الجبوري") == Match.SAME
    assert compare_laqab("الجبوري", "العبيدي") == Match.CONFLICT


def test_full_names_same_and_variant():
    assert compare_names("علي حسين كاظم", "علي حسين كاظم").match == Match.SAME
    r = compare_names("محمد عبدالله حسين", "محمد عبد الله حسين")
    assert r.match == Match.VARIANT


def test_conflict_in_grandfather_is_detected_despite_high_similarity():
    r = compare_names("علي حسين كاظم", "علي حسن كاظم")
    assert r.match == Match.CONFLICT
    assert r.similarity > 0.8  # similar strings, still a conflict: similarity never decides
    assert [p.index for p in r.parts if p.match == Match.CONFLICT] == [1]


def test_missing_laqab_is_partial():
    r = compare_names(["علي", "حسين", "كاظم", "الجبوري"], "علي حسين كاظم")
    assert r.match == Match.PARTIAL
    assert r.missing == ("laqab",)


def test_missing_grandfather_alignment():
    # license shows given + father + laqab
    r = compare_names(["علي", "حسين", "كاظم", "الجبوري"], "علي حسين الجبوري")
    assert r.match == Match.PARTIAL
    assert r.missing == ("grandfather",)


def test_partial_with_sound_alike_stays_partial_and_conflict_dominates():
    r = compare_names(["علي", "گاطع", "حسن", "الربيعي"], "علي قاطع حسن")
    assert r.match == Match.PARTIAL  # worst is PARTIAL (sound-alike is better than partial)
    assert any(p.match == Match.SOUND_ALIKE for p in r.parts)
    r2 = compare_names(["علي", "حسين", "كاظم", "الجبوري"], "عمر حسين")
    assert r2.match == Match.CONFLICT


def test_order_of_arguments_does_not_change_the_class():
    a, b = ["علي", "حسين", "كاظم", "الجبوري"], "علي حسين الجبوري"
    assert compare_names(a, b).match == compare_names(b, a).match


def test_empty_names_are_never_a_match():
    assert compare_names("", "علي").match == Match.CONFLICT
    assert compare_names([], []).match == Match.CONFLICT


def test_reasons_are_stable_codes():
    r = compare_names(["علي", "حسين", "كاظم", "الجبوري"], "علي حسين كاظم")
    assert r.reasons == ("NAME_PARTIAL", "MISSING_LAQAB")
