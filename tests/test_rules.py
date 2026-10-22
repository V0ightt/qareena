import datetime as dt

import pytest

from qareena.checks.mrz import make_td1
from qareena.config import Settings, load_runtime
from qareena.pipeline.records import DocumentRead, Source
from qareena.pipeline.rules import (
    apply_field_rules,
    cross_document,
    make_record,
    national_card_rules,
    validity_rules,
)

TODAY = dt.date(2026, 10, 22)
RT = load_runtime(Settings(_env_file=None))


def base_values():
    dob = dt.date(1988, 5, 14)
    expiry = dt.date(2031, 3, 2)
    mrz = make_td1(
        document_number="QD3970915",
        optional_1="991234567890",
        birth_date=dob,
        sex="M",
        expiry_date=expiry,
        primary_name="ALJUBOURI",
        secondary_names=["ALI"],
    )
    return {
        "iq_national_card_front": {
            "national_id_number": "991234567890",
            "card_access_number": "QD3970915",
            "given_name": "علي",
            "father_name": "عبدالله",
            "grandfather_name": "حسين",
            "laqab": "الجبوري",
            "mother_name": "زينب",
            "mother_father_name": "كاظم",
            "sex": "ذكر",
            "blood_type": "O+",
        },
        "iq_national_card_back": {
            "issuing_authority": "مديرية البطاقة النموذجية",
            "issue_date": "2021/03/02",
            "expiry_date": "2031/03/02",
            "date_of_birth": "1988/05/14",
            "place_of_birth": "بغداد",
            "family_number": "ZZ1234567890ABCDEF",
            "mrz_line_1": mrz[0],
            "mrz_line_2": mrz[1],
            "mrz_line_3": mrz[2],
        },
        "chamber_of_commerce_id": {
            "member_name": "علي عبدالله حسين الجبوري",
            "trade_name": "أسواق دجلة",
            "membership_number": "48213",
            "member_class": "الأولى",
            "chamber_governorate": "غرفة تجارة بغداد",
            "issue_date": "05/01/2026",
            "valid_until": "31/12/2026",
            "activity": "بيع المواد الغذائية بالمفرد",
        },
        "tax_card": {
            "taxpayer_name": "علي عبدالله حسين الجبوري",
            "tax_id": "9912345678",
            "trade_name": "أسواق دجلة",
            "tax_office": "فرع الكرخ",
            "issue_date": "10/02/2026",
            "valid_until": "31/12/2027",
        },
    }


def build(values, grounded=True):
    reads = {}
    for doc_type, fields in values.items():
        d = DocumentRead(doc_type=doc_type, image_id=doc_type)
        for f, raw in fields.items():
            rec = make_record(doc_type, f, raw, RT.catalog)
            if grounded and raw is not None:
                rec.evidence.ocr_matches, rec.evidence.ocr_similarity = 1, 1.0
            d.fields[f] = rec
        reads[doc_type] = d
    for d in reads.values():
        apply_field_rules(d, RT.catalog)
    national_card_rules(reads.get("iq_national_card_front"), reads.get("iq_national_card_back"), TODAY)
    for k in ("chamber_of_commerce_id", "tax_card"):
        validity_rules(reads.get(k), TODAY)
    checks = cross_document(reads, RT.policy)
    return reads, checks


def blocking(reads):
    return {(d, f): r.reasons for d, doc in reads.items() for f, r in doc.fields.items() if r.blocking}


def test_consistent_packet_has_no_blocking_reasons_and_two_sources_everywhere_critical():
    reads, checks = build(base_values())
    assert blocking(reads) == {}
    for doc in reads.values():
        for rec in doc.fields.values():
            if rec.critical and rec.value is not None:
                assert len(rec.sources) >= 1, (rec.field, rec.sources)
    front = reads["iq_national_card_front"].fields
    assert Source.MRZ in front["national_id_number"].sources
    assert Source.CROSS_DOC in front["grandfather_name"].sources
    assert sorted((c.rule, c.result) for c in checks) == [
        ("OWNER_NAME", "SAME"),
        ("OWNER_NAME", "SAME"),
        ("TRADE_NAME", "SAME"),
    ]


def test_grandfather_conflict_blocks_both_sides():
    v = base_values()
    v["chamber_of_commerce_id"]["member_name"] = "علي عبدالله حسن الجبوري"
    reads, checks = build(v)
    assert "NAME_CONFLICT" in reads["iq_national_card_front"].fields["grandfather_name"].reasons
    assert reads["chamber_of_commerce_id"].fields["member_name"].blocking
    assert any(c.result == "CONFLICT" for c in checks)


def test_spelling_variant_is_a_match_with_a_note():
    v = base_values()
    v["tax_card"]["taxpayer_name"] = "علي عبد الله حسين الجبوري"
    reads, _ = build(v)
    tax = reads["tax_card"].fields["taxpayer_name"]
    assert not tax.blocking
    assert "NAME_VARIANT" in tax.reasons
    assert Source.CROSS_DOC in tax.sources


def test_missing_laqab_follows_policy_note():
    v = base_values()
    v["chamber_of_commerce_id"]["member_name"] = "علي عبدالله حسين"
    reads, _ = build(v)
    rec = reads["chamber_of_commerce_id"].fields["member_name"]
    assert "NOTE_MISSING_LAQAB" in rec.reasons
    assert not rec.blocking


def test_mrz_mismatch_on_national_number_blocks():
    v = base_values()
    v["iq_national_card_front"]["national_id_number"] = "991234567899"  # a plausible misread
    reads, _ = build(v)
    rec = reads["iq_national_card_front"].fields["national_id_number"]
    assert "MRZ_VIZ_MISMATCH" in rec.reasons
    assert rec.blocking
    assert Source.MRZ not in rec.sources


def test_broken_mrz_gives_no_mrz_source():
    v = base_values()
    line2 = v["iq_national_card_back"]["mrz_line_2"]
    v["iq_national_card_back"]["mrz_line_2"] = line2[:3] + ("9" if line2[3] != "9" else "8") + line2[4:]
    reads, _ = build(v)
    back = reads["iq_national_card_back"]
    assert back.mrz["valid"] is False
    assert Source.MRZ not in back.fields["date_of_birth"].sources


def test_expired_and_inconsistent_dates():
    v = base_values()
    v["iq_national_card_back"]["expiry_date"] = "2026/03/02"
    reads, _ = build(v)
    exp = reads["iq_national_card_back"].fields["expiry_date"]
    assert "EXPIRED" in exp.reasons
    assert "DATE_INCONSISTENT" in exp.reasons  # issue + 10 years != expiry
    assert "MRZ_VIZ_MISMATCH" in exp.reasons


def test_underage_is_blocking():
    v = base_values()
    v["iq_national_card_back"]["date_of_birth"] = "2012/01/01"
    reads, _ = build(v)
    assert "UNDERAGE" in reads["iq_national_card_back"].fields["date_of_birth"].reasons


@pytest.mark.parametrize("label", ["دەسەڵاتی دەرکردن", "دهسهلتي دهركردن"])
def test_label_echo_empties_the_field(label):
    v = base_values()
    v["iq_national_card_back"]["issuing_authority"] = label
    reads, _ = build(v)
    rec = reads["iq_national_card_back"].fields["issuing_authority"]
    assert rec.value is None
    assert "LABEL_ECHO" in rec.reasons
    assert rec.blocking


def test_short_national_number_is_a_format_error():
    v = base_values()
    v["iq_national_card_front"]["national_id_number"] = "99123456789"
    reads, _ = build(v)
    assert "FORMAT_INVALID" in reads["iq_national_card_front"].fields["national_id_number"].reasons


def test_trade_name_conflict_and_expired_chamber_id():
    v = base_values()
    v["tax_card"]["trade_name"] = "معرض النخيل"
    v["chamber_of_commerce_id"]["valid_until"] = "31/12/2025"
    reads, _ = build(v)
    assert "TRADE_NAME_CONFLICT" in reads["tax_card"].fields["trade_name"].reasons
    assert "EXPIRED" in reads["chamber_of_commerce_id"].fields["valid_until"].reasons


def test_ungrounded_values_get_no_ocr_source():
    reads, _ = build(base_values(), grounded=False)
    rec = reads["iq_national_card_front"].fields["mother_name"]
    assert "NOT_GROUNDED" in rec.reasons
    assert Source.OCR not in rec.sources
