"""The written rulebook: deterministic per-document and cross-document checks.

Rules never accept a field. They add reason codes (blocking ones force human review) and
record which *independent sources* agree with the VLM's reading. A critical field may only be
accepted later if at least one such source exists (ADR-004: two agreeing sources).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping

from qareena.checks.fields import format_error, garbled, label_echo
from qareena.checks.mrz import read_td1
from qareena.config import Catalog, FieldKind, Policy
from qareena.eval.compare import canonical
from qareena.pipeline.records import CrossCheck, DocumentRead, FieldRecord, Source
from qareena.text import names
from qareena.text.names import Match

FRONT, BACK = "iq_national_card_front", "iq_national_card_back"
CHAMBER, TAX = "chamber_of_commerce_id", "tax_card"
ID_NAME_PARTS = ("given_name", "father_name", "grandfather_name", "laqab")
OCR_EXACT = 0.95
CARD_VALIDITY_YEARS = 10
VALIDITY_TOLERANCE_DAYS = 3
ADULT_AGE = 18


# --- per field ----------------------------------------------------------------------------


def apply_field_rules(doc: DocumentRead, catalog: Catalog) -> None:
    """Label echo, format, required, and OCR corroboration for every field of one document."""
    labels = catalog.printed_labels()
    spec_doc = catalog.documents[doc.doc_type]
    for name, rec in doc.fields.items():
        spec = spec_doc.fields[name]
        if rec.raw is not None and (label_echo(rec.raw, labels) is not None or garbled(rec.raw)):
            rec.value = None  # a printed label or spelled-out glyphs are not a value: treat as unread
            rec.add_reason(
                "LABEL_ECHO" if label_echo(rec.raw, labels) else "GARBLED_TEXT", blocking=spec.required
            )
            rec.evidence.ocr_matches = 0
        if rec.value is None:
            if spec.required:
                rec.add_reason("REQUIRED_FIELD_MISSING", blocking=True)
            continue
        if rec.raw is not None and (err := format_error(rec.raw, spec)) is not None:
            rec.add_reason(err, blocking=True)
        ev = rec.evidence
        if ev.ocr_matches >= 1 and ev.ocr_similarity >= OCR_EXACT:
            # Only a match at the field's own printed label is independent evidence; on a layout
            # whose labels we cannot find, a match elsewhere is a signal for the scorer only.
            rec.add_source(Source.OCR if ev.ocr_anchored else Source.OCR_UNANCHORED)
            if ev.ocr_matches > 1:
                rec.add_reason("OCR_AMBIGUOUS_BOX")
        elif ev.ocr_foreign:  # the value is printed, but at another field's label: a field shift
            rec.add_reason("VALUE_AT_OTHER_LABEL", blocking=spec.critical)
        else:
            rec.add_reason("NOT_GROUNDED")


# --- national card (front + back + MRZ) --------------------------------------------------


def national_card_rules(front: DocumentRead | None, back: DocumentRead | None, today: dt.date) -> None:
    if back is not None:
        issue, expiry, dob = (_date(back, f) for f in ("issue_date", "expiry_date", "date_of_birth"))
        if expiry is not None and expiry <= today:
            back.fields["expiry_date"].add_reason("EXPIRED", blocking=True)
        if issue is not None and issue > today:
            back.fields["issue_date"].add_reason("DATE_IN_FUTURE", blocking=True)
        if dob is not None and issue is not None and dob >= issue:
            back.fields["date_of_birth"].add_reason("DATE_ORDER", blocking=True)
        if dob is not None and _age(dob, today) < ADULT_AGE:
            back.fields["date_of_birth"].add_reason("UNDERAGE", blocking=True)
        if issue is not None and expiry is not None:
            expected = _add_years(issue, CARD_VALIDITY_YEARS)
            if abs((expiry - expected).days) > VALIDITY_TOLERANCE_DAYS:
                for f in ("issue_date", "expiry_date"):
                    back.fields[f].add_reason("DATE_INCONSISTENT")  # one of the two is probably misread
        _mrz_rules(front, back, today)


def _mrz_rules(front: DocumentRead | None, back: DocumentRead, today: dt.date) -> None:
    raw = [back.fields.get(f"mrz_line_{k}") for k in (1, 2, 3)]
    if any(r is None or r.raw is None for r in raw):
        return
    mrz, notes = read_td1([r.raw for r in raw], today=today)  # type: ignore[union-attr]
    back.mrz = {
        "valid": mrz.valid,
        "checks": mrz.checks,
        "errors": list(mrz.errors),
        "notes": list(notes),
        "document_number": mrz.document_number,
        "national_number": mrz.optional_1,
        "birth_date": mrz.birth_date.isoformat() if mrz.birth_date else None,
        "expiry_date": mrz.expiry_date.isoformat() if mrz.expiry_date else None,
        "sex": mrz.sex,
    }
    if not mrz.valid:
        for r in raw:
            r.add_reason("MRZ_CHECK_FAILED")  # type: ignore[union-attr]
        return
    for r, line in zip(raw, mrz.lines, strict=True):
        r.add_source(Source.MRZ)  # type: ignore[union-attr]
        if "NOT_GROUNDED" in r.reasons:  # type: ignore[union-attr]
            r.reasons.remove("NOT_GROUNDED")  # type: ignore[union-attr]  # check digits outrank OCR here
        if r.value != line:  # type: ignore[union-attr]
            # filler count or O/0-type repairs, kept only because every check digit passes
            r.value = line  # type: ignore[union-attr]
            r.add_reason("NOTE_MRZ_REPAIRED")  # type: ignore[union-attr]
    pairs = [
        (back, "date_of_birth", mrz.birth_date.isoformat() if mrz.birth_date else None, True),
        (back, "expiry_date", mrz.expiry_date.isoformat() if mrz.expiry_date else None, True),
    ]
    if front is not None:
        pairs += [
            (front, "national_id_number", mrz.optional_1 or None, True),
            (front, "card_access_number", mrz.document_number or None, False),
        ]
    for doc, field, mrz_value, critical in pairs:
        rec = doc.fields.get(field)
        if rec is None or rec.value is None or mrz_value is None:
            continue
        if rec.value == mrz_value:
            rec.add_source(Source.MRZ)
        else:
            rec.add_reason("MRZ_VIZ_MISMATCH", blocking=critical)
    if front is not None and (sex := front.fields.get("sex")) is not None and sex.value:
        expected = {"M": "ذكر", "F": "أنثى"}.get(mrz.sex)  # ذكر / أنثى
        if expected and sex.value != expected:
            sex.add_reason("MRZ_VIZ_MISMATCH")


# --- business documents ----------------------------------------------------------------


def validity_rules(doc: DocumentRead | None, today: dt.date) -> None:
    if doc is None:
        return
    issue, valid_until = _date(doc, "issue_date"), _date(doc, "valid_until")
    if valid_until is not None and valid_until < today:
        doc.fields["valid_until"].add_reason("EXPIRED", blocking=True)
    if issue is not None and issue > today:
        doc.fields["issue_date"].add_reason("DATE_IN_FUTURE", blocking=True)
    if issue is not None and valid_until is not None and valid_until < issue:
        doc.fields["valid_until"].add_reason("DATE_ORDER", blocking=True)


# --- cross-document --------------------------------------------------------------------


def cross_document(reads: Mapping[str, DocumentRead], policy: Policy) -> list[CrossCheck]:
    checks: list[CrossCheck] = []
    front = reads.get(FRONT)
    if front is not None:
        id_parts = [(p, front.fields[p]) for p in ID_NAME_PARTS if p in front.fields]
        for doc_type, field in ((CHAMBER, "member_name"), (TAX, "taxpayer_name")):
            other = reads.get(doc_type)
            if other is not None and field in other.fields:
                checks.append(_owner_name(id_parts, other.fields[field], doc_type, field, policy))
    chamber, tax = reads.get(CHAMBER), reads.get(TAX)
    if chamber is not None and tax is not None:
        checks.append(_same_text("TRADE_NAME", chamber.fields["trade_name"], tax.fields["trade_name"]))
    return checks


def _owner_name(
    id_parts: list[tuple[str, FieldRecord]], full: FieldRecord, doc_type: str, field: str, policy: Policy
) -> CrossCheck:
    members = [(FRONT, p) for p, _ in id_parts] + [(doc_type, field)]
    given = [r for _, r in id_parts if r.field != "laqab"]
    if full.value is None or any(r.value is None for r in given):
        return CrossCheck(
            rule="OWNER_NAME", members=members, result="MISSING", reasons=["CROSS_CHECK_INCOMPLETE"]
        )
    left = [r.value for _, r in id_parts if r.value]
    cmp = names.compare_names(left, full.value)  # type: ignore[arg-type]
    reasons = list(cmp.reasons)
    present = [(p, r) for p, r in id_parts if r.value]
    for part in cmp.parts:
        rec = present[part.index][1] if part.index < len(present) else None
        if part.index == 3 and len(present) == 3:  # skip-grandfather alignment compares laqab slot
            rec = None
        if rec is None:
            continue
        if part.match in (Match.SAME, Match.VARIANT):
            rec.add_source(Source.CROSS_DOC)
        elif part.match == Match.SOUND_ALIKE:
            rec.add_reason("NAME_SOUND_ALIKE")
        elif part.match == Match.CONFLICT:
            rec.add_reason("NAME_CONFLICT", blocking=True)
    if cmp.match in (Match.SAME, Match.VARIANT) or (
        cmp.match == Match.PARTIAL and all(p.match <= Match.VARIANT for p in cmp.parts)
    ):
        full.add_source(Source.CROSS_DOC)
    if cmp.match == Match.VARIANT:
        full.add_reason("NAME_VARIANT")
    elif cmp.match == Match.SOUND_ALIKE:
        full.add_reason("NAME_SOUND_ALIKE")
    elif cmp.match == Match.PARTIAL:
        for m in cmp.missing:
            rule = {"laqab": policy.names.missing_laqab, "grandfather": policy.names.missing_grandfather}.get(
                m, "check"
            )
            full.add_reason(f"NAME_MISSING_{m.upper()}" if rule == "check" else f"NOTE_MISSING_{m.upper()}")
    elif cmp.match == Match.CONFLICT:
        full.add_reason("NAME_CONFLICT", blocking=True)
    return CrossCheck(
        rule="OWNER_NAME",
        members=members,
        result=cmp.match.name,
        reasons=reasons,
        detail={
            "similarity": round(cmp.similarity, 3),
            "missing": list(cmp.missing),
            "parts": [(p.index, p.left, p.right, p.match.name) for p in cmp.parts],
        },
    )


def _same_text(rule: str, a: FieldRecord, b: FieldRecord) -> CrossCheck:
    members = [(a.doc_type, a.field), (b.doc_type, b.field)]
    if a.value is None or b.value is None:
        return CrossCheck(rule=rule, members=members, result="MISSING", reasons=["CROSS_CHECK_INCOMPLETE"])
    m = names.compare_part(a.value, b.value)
    if m in (Match.SAME, Match.VARIANT):
        a.add_source(Source.CROSS_DOC)
        b.add_source(Source.CROSS_DOC)
    elif m == Match.SOUND_ALIKE:
        for r in (a, b):
            r.add_reason(f"{rule}_SOUND_ALIKE")
    else:
        for r in (a, b):
            r.add_reason(f"{rule}_CONFLICT", blocking=True)
    return CrossCheck(rule=rule, members=members, result=m.name, reasons=[f"{rule}_{m.name}"])


# --- helpers ---------------------------------------------------------------------------------


def make_record(doc_type: str, field: str, raw: str | None, catalog: Catalog) -> FieldRecord:
    spec = catalog.documents[doc_type].fields[field]
    value = canonical(raw, spec) if raw is not None else None
    if value is not None and spec.kind == FieldKind.DATE and value.startswith("?"):
        rec = FieldRecord(
            doc_type=doc_type,
            field=field,
            raw=raw,
            value=None,
            critical=spec.critical,
            required=spec.required,
        )
        rec.add_reason("DATE_UNPARSEABLE", blocking=spec.required)
        return rec
    return FieldRecord(
        doc_type=doc_type, field=field, raw=raw, value=value, critical=spec.critical, required=spec.required
    )


def _date(doc: DocumentRead, field: str) -> dt.date | None:
    rec = doc.fields.get(field)
    if rec is None or rec.value is None:
        return None
    try:
        return dt.date.fromisoformat(rec.value)
    except ValueError:
        return None


def _age(dob: dt.date, today: dt.date) -> int:
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def _add_years(d: dt.date, years: int) -> dt.date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(year=d.year + years, day=28)
