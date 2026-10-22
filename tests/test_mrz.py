import datetime as dt

import pytest
from hypothesis import given
from hypothesis import strategies as st

from qareena.checks.mrz import check_digit, make_td1, parse_td1

# ICAO Doc 9303 Part 5 specimen (fictional state UTO)
ICAO_SPECIMEN = (
    "I<UTOD231458907<<<<<<<<<<<<<<<",
    "7408122F1204159UTO<<<<<<<<<<<6",
    "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
)
TODAY = dt.date(2026, 10, 22)


def test_check_digit_known_values():
    assert check_digit("D23145890") == "7"
    assert check_digit("740812") == "2"
    assert check_digit("120415") == "9"
    assert check_digit("<<<<<<") == "0"


def test_check_digit_rejects_invalid_characters():
    with pytest.raises(ValueError, match="invalid MRZ character"):
        check_digit("AB-12")


def test_icao_specimen_is_valid():
    m = parse_td1(ICAO_SPECIMEN, today=TODAY)
    assert m.valid, (m.checks, m.errors)
    assert m.document_number == "D23145890"
    assert m.birth_date == dt.date(1974, 8, 12)
    assert m.expiry_date == dt.date(2012, 4, 15)
    assert (m.primary_name, m.secondary_names) == ("ERIKSSON", ("ANNA", "MARIA"))
    assert m.sex == "F"
    assert m.issuer == "UTO"


@pytest.mark.parametrize(("line", "pos", "char"), [(0, 8, "9"), (1, 3, "9"), (1, 10, "5"), (0, 20, "1")])
def test_single_character_change_breaks_a_check(line, pos, char):
    rows = list(ICAO_SPECIMEN)
    row = rows[line]
    rows[line] = row[:pos] + char + row[pos + 1 :]
    assert not parse_td1(rows, today=TODAY).valid


def test_malformed_input_is_reported_not_raised():
    m = parse_td1(["SHORT", "LINES"], today=TODAY)
    assert not m.valid
    assert any("expected 3 lines" in e for e in m.errors)
    m2 = parse_td1(["I<UTO" + "é" * 25, ICAO_SPECIMEN[1], ICAO_SPECIMEN[2]], today=TODAY)
    assert not m2.valid


def test_whitespace_and_case_are_tolerated():
    m = parse_td1(
        " " + ICAO_SPECIMEN[0].lower() + " \n" + ICAO_SPECIMEN[1] + "\n" + ICAO_SPECIMEN[2], today=TODAY
    )
    assert m.valid


def test_invalid_date_is_an_error():
    rows = list(ICAO_SPECIMEN)
    bad = "741312"  # month 13
    rows[1] = bad + check_digit(bad) + rows[1][7:]
    m = parse_td1(rows, today=TODAY)
    assert not m.valid
    assert any("birth date" in e for e in m.errors)


def test_generated_iraqi_style_card_round_trips():
    lines = make_td1(
        document_number="AB1234567",
        optional_1="199012345678",
        birth_date=dt.date(1990, 3, 7),
        sex="M",
        expiry_date=dt.date(2034, 5, 1),
        primary_name="ALJUBOURI",
        secondary_names=["ALI", "HUSSEIN"],
    )
    m = parse_td1(lines, today=TODAY)
    assert m.valid, (m.checks, m.errors)
    assert m.issuer == "UTO"
    assert m.nationality == "UTO"
    assert m.optional_1 == "199012345678"
    assert m.birth_date == dt.date(1990, 3, 7)
    assert m.expiry_date == dt.date(2034, 5, 1)


@given(
    doc=st.from_regex(r"[A-Z]{2}[0-9]{7}", fullmatch=True),
    nid=st.from_regex(r"[0-9]{12}", fullmatch=True),
    dob=st.dates(min_value=dt.date(1940, 1, 1), max_value=dt.date(2008, 12, 31)),
    exp=st.dates(min_value=dt.date(2020, 1, 1), max_value=dt.date(2040, 12, 31)),
    sex=st.sampled_from(["M", "F"]),
)
def test_round_trip_property(doc, nid, dob, exp, sex):
    m = parse_td1(
        make_td1(
            document_number=doc,
            optional_1=nid,
            birth_date=dob,
            sex=sex,
            expiry_date=exp,
            primary_name="TEST",
            secondary_names=["A"],
        ),
        today=TODAY,
    )
    assert m.valid
    assert (m.document_number, m.optional_1, m.birth_date, m.expiry_date) == (doc, nid, dob, exp)


def test_canonicalize_fixes_miscounted_fillers_only():
    from qareena.checks.mrz import canonicalize_fillers

    l1, l2, l3 = ICAO_SPECIMEN
    sloppy = (l1 + "<<", l2[:18] + "<" * 9 + l2[-1], l3.replace("<<ANNA", "<<<ANNA") + "<<<")
    fixed, notes = canonicalize_fillers(sloppy)
    assert fixed == ICAO_SPECIMEN
    assert len(notes) == 4
    assert parse_td1(fixed, today=TODAY).valid


def test_canonicalize_never_changes_content():
    from qareena.checks.mrz import canonicalize_fillers

    l1, l2, l3 = ICAO_SPECIMEN
    # a missing digit inside line 1 is a content error: the line must stay invalid
    broken = (l1[:10] + l1[11:], l2, l3)
    fixed, _ = canonicalize_fillers(broken)
    assert not parse_td1(fixed, today=TODAY).valid
    # non-filler optional data in line 2 is left alone
    odd = (l1, l2[:18] + "AB" + "<" * 8 + l2[-1], l3)
    fixed2, notes2 = canonicalize_fillers(odd)
    assert fixed2[1] == odd[1]
    assert not notes2
    # already-canonical input is untouched
    assert canonicalize_fillers(ICAO_SPECIMEN) == (ICAO_SPECIMEN, ())


def test_generator_refuses_bad_input():
    with pytest.raises(ValueError, match="fit"):
        make_td1(
            document_number="AB1234567",
            optional_1="1" * 16,
            birth_date=dt.date(1990, 1, 1),
            sex="M",
            expiry_date=dt.date(2030, 1, 1),
            primary_name="X",
            secondary_names=[],
        )
    with pytest.raises(ValueError, match="sex"):
        make_td1(
            document_number="AB1234567",
            optional_1="",
            birth_date=dt.date(1990, 1, 1),
            sex="X",
            expiry_date=dt.date(2030, 1, 1),
            primary_name="X",
            secondary_names=[],
        )


def test_class_repair_fixes_o_zero_confusions_only_when_check_digits_pass():
    from qareena.checks.mrz import read_td1

    lines = make_td1(
        document_number="OE9159457",
        optional_1="995258598991",
        birth_date=dt.date(1990, 3, 7),
        sex="M",
        expiry_date=dt.date(2034, 5, 1),
        primary_name="ALZUBAIDI",
        secondary_names=["RAAD"],
    )
    misread = (lines[0].replace("UTO", "UT0").replace("OE9", "0E9"), lines[1] + "<<", lines[2] + "<")
    m, notes = read_td1(misread, today=TODAY)
    assert m.valid
    assert m.document_number == "OE9159457"
    assert m.issuer == "UTO"
    assert any("check digit" in n for n in notes)


def test_class_repair_cannot_rescue_a_wrong_digit():
    from qareena.checks.mrz import read_td1

    rows = list(
        make_td1(
            document_number="AB1234567",
            optional_1="991234567890",
            birth_date=dt.date(1990, 3, 7),
            sex="M",
            expiry_date=dt.date(2034, 5, 1),
            primary_name="X",
            secondary_names=["Y"],
        )
    )
    rows[1] = rows[1][:2] + ("7" if rows[1][2] != "7" else "6") + rows[1][3:]  # a real digit misread
    m, _ = read_td1(rows, today=TODAY)
    assert not m.valid
