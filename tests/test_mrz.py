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
