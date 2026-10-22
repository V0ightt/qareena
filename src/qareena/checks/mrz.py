"""ICAO Doc 9303 Part 5 TD1 machine-readable zone (3 lines x 30 characters).

Used two ways:
* to *validate* a read MRZ: every check digit must pass before any MRZ value is trusted;
* to *generate* synthetic MRZs for our fictional documents (issuer "UTO", ICAO's own
  fictional state), so test cards can never collide with a real Iraqi card.

Layout (1-based positions, as in the standard):
  line 1:  1-2 document code | 3-5 issuing state | 6-14 document number | 15 check |
           16-30 optional data (Iraqi cards: the 12-digit national number)
  line 2:  1-6 birth date YYMMDD | 7 check | 8 sex | 9-14 expiry YYMMDD | 15 check |
           16-18 nationality | 19-29 optional data | 30 composite check
  line 3:  name: PRIMARY<<SECONDARY<NAMES, filler "<"
Composite check covers line 1 positions 6-30, line 2 positions 1-7, 9-15 and 19-29.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

LINE_LEN = 30
_ALLOWED = re.compile(r"^[A-Z0-9<]{30}$")
_VALUES = {**{str(d): d for d in range(10)}, **{chr(65 + i): 10 + i for i in range(26)}, "<": 0}
_WEIGHTS = (7, 3, 1)


def check_digit(data: str) -> str:
    """ICAO 7-3-1 check digit of ``data`` (characters 0-9, A-Z, '<')."""
    total = 0
    for i, ch in enumerate(data):
        try:
            total += _VALUES[ch] * _WEIGHTS[i % 3]
        except KeyError:
            raise ValueError(f"invalid MRZ character {ch!r}") from None
    return str(total % 10)


@dataclass(frozen=True)
class TD1:
    document_code: str
    issuer: str
    document_number: str
    optional_1: str
    birth_date: dt.date | None
    sex: str
    expiry_date: dt.date | None
    nationality: str
    optional_2: str
    primary_name: str
    secondary_names: tuple[str, ...]
    checks: dict[str, bool] = field(default_factory=dict)
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """All check digits pass, every date parses, and the lines were well formed."""
        return not self.errors and bool(self.checks) and all(self.checks.values())


def parse_td1(lines: list[str] | tuple[str, ...] | str, *, today: dt.date | None = None) -> TD1:
    """Parse and validate a TD1 MRZ. Never raises on bad content; problems go in ``errors``.

    Whitespace is removed and letters upper-cased, nothing else: we do not "repair" OCR
    confusions here, because a repaired value must still be proven by its check digit.
    """
    today = today or dt.date.today()
    if isinstance(lines, str):
        lines = lines.splitlines()
    rows = ["".join(r.split()).upper() for r in lines if r.strip()]
    errors: list[str] = []
    if len(rows) != 3:
        errors.append(f"expected 3 lines, got {len(rows)}")
        rows = (rows + ["<" * LINE_LEN] * 3)[:3]
    for n, r in enumerate(rows, 1):
        if not _ALLOWED.match(r):
            errors.append(f"line {n} is not 30 characters of A-Z, 0-9, '<'")
    l1, l2, l3 = (r.ljust(LINE_LEN, "<")[:LINE_LEN] for r in rows)
    safe = [_safe(x) for x in (l1, l2, l3)]
    l1, l2, l3 = safe

    doc_number, doc_cd_field, optional_1 = l1[5:14], l1[14], l1[15:30]
    if doc_cd_field == "<" and optional_1.strip("<"):  # long document number (overflow)
        overflow = optional_1.split("<", 1)[0]
        doc_number, doc_cd_field = doc_number + overflow[:-1], overflow[-1:]
        optional_1 = optional_1[len(overflow) :]
    checks = {
        "document_number": _cd_ok(doc_number, doc_cd_field),
        "birth_date": _cd_ok(l2[0:6], l2[6]),
        "expiry_date": _cd_ok(l2[8:14], l2[14]),
        "composite": _cd_ok(l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29], l2[29]),
    }
    birth = _parse_date(l2[0:6], today, kind="birth")
    expiry = _parse_date(l2[8:14], today, kind="expiry")
    if birth is None:
        errors.append("birth date is not a valid YYMMDD date")
    if expiry is None:
        errors.append("expiry date is not a valid YYMMDD date")
    primary, _, secondary = l3.partition("<<")
    return TD1(
        document_code=l1[0:2].rstrip("<"),
        issuer=l1[2:5],
        document_number=doc_number.rstrip("<"),
        optional_1=optional_1.rstrip("<"),
        birth_date=birth,
        sex=l2[7],
        expiry_date=expiry,
        nationality=l2[15:18],
        optional_2=l2[18:29].rstrip("<"),
        primary_name=primary.replace("<", " ").strip(),
        secondary_names=tuple(p for p in secondary.split("<") if p),
        checks=checks,
        errors=tuple(errors),
    )


def make_td1(
    *,
    document_number: str,
    optional_1: str,
    birth_date: dt.date,
    sex: str,
    expiry_date: dt.date,
    primary_name: str,
    secondary_names: list[str] | tuple[str, ...],
    issuer: str = "UTO",
    nationality: str = "UTO",
    document_code: str = "ID",
    optional_2: str = "",
) -> tuple[str, str, str]:
    """Build a valid TD1 MRZ. Defaults to ICAO's fictional state UTO: never a real card."""
    if len(document_number) > 9:
        raise ValueError("document numbers longer than 9 characters are not generated")
    if sex not in ("M", "F", "<"):
        raise ValueError("sex must be M, F or <")
    doc = _field(document_number, 9)
    opt1 = _field(optional_1, 15)
    l1 = _field(document_code, 2) + _field(issuer, 3) + doc + check_digit(doc) + opt1
    dob, exp = birth_date.strftime("%y%m%d"), expiry_date.strftime("%y%m%d")
    opt2 = _field(optional_2, 11)
    l2_body = dob + check_digit(dob) + sex + exp + check_digit(exp) + _field(nationality, 3) + opt2
    composite = check_digit(l1[5:30] + l2_body[0:7] + l2_body[8:15] + l2_body[18:29])
    l2 = l2_body + composite
    name = _clean(primary_name)
    if secondary_names:
        name += "<<" + "<".join(_clean(n) for n in secondary_names)
    l3 = name[:LINE_LEN].ljust(LINE_LEN, "<")  # ICAO allows truncating long names
    for line in (l1, l2, l3):
        if len(line) != LINE_LEN or not _ALLOWED.match(line):
            raise ValueError(f"generated MRZ line is malformed: {line!r}")
    return l1, l2, l3


def canonicalize_fillers(lines: list[str] | tuple[str, ...]) -> tuple[tuple[str, str, str], tuple[str, ...]]:
    """Fix miscounted '<' filler runs without touching any letter or digit.

    Readers (human or model) often miscount long runs of '<'. Fillers carry no information
    except position, so re-padding them is safe; every value is still proven by its check
    digit afterwards. Only these cases are handled, anything else is returned unchanged:

    * line 1: positions 1-15 intact, then optional data followed only by fillers
    * line 2: 18 fixed characters, an all-filler optional block, then the composite digit
    * line 3: trailing fillers, and runs of 3+ '<' between name parts (-> '<<')

    Returns the canonical lines and a description of each change, for the evidence trace.
    """
    rows = ["".join(r.split()).upper() for r in lines]
    rows = ([*rows, "", "", ""])[:3]
    notes: list[str] = []
    l1, l2, l3 = rows

    if len(l1) != LINE_LEN and len(l1) >= 15:
        head, tail = l1[:15], l1[15:]
        content = tail.rstrip("<")
        if "<" not in content and len(content) <= 15:
            l1 = head + content.ljust(15, "<")
            notes.append(f"line 1 filler re-padded ({len(rows[0])} -> 30)")

    if len(l2) != LINE_LEN and len(l2) >= 19:
        head, middle, last = l2[:18], l2[18:-1], l2[-1]
        if set(middle) <= {"<"} and last.isdigit():
            l2 = head + "<" * 11 + last
            notes.append(f"line 2 optional-data filler re-padded ({len(rows[1])} -> 30)")

    stripped = l3.rstrip("<")
    name = re.sub(r"(?<=[A-Z])<{3,}(?=[A-Z])", "<<", stripped)
    if name != stripped:
        notes.append("line 3 name separator collapsed to '<<'")
    if len(name) <= LINE_LEN:
        new3 = name.ljust(LINE_LEN, "<")
        if len(l3) != LINE_LEN:
            notes.append(f"line 3 filler re-padded ({len(l3)} -> 30)")
        l3 = new3
    return (l1, l2, l3), tuple(notes)


# --- internals ------------------------------------------------------------------------


def _safe(line: str) -> str:
    return "".join(ch if ch in _VALUES else "<" for ch in line)


def _cd_ok(data: str, digit: str) -> bool:
    return digit.isdigit() and check_digit(data) == digit


def _parse_date(yymmdd: str, today: dt.date, *, kind: str) -> dt.date | None:
    if not yymmdd.isdigit():
        return None
    yy, mm, dd = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    if kind == "birth":  # the most recent century that is not in the future
        year = 2000 + yy if 2000 + yy <= today.year else 1900 + yy
    else:  # expiry: within the next 50 years, else last century
        year = 2000 + yy if 2000 + yy <= today.year + 50 else 1900 + yy
    try:
        value = dt.date(year, mm, dd)
    except ValueError:
        return None
    if kind == "birth" and value > today:
        try:
            value = dt.date(year - 100, mm, dd)
        except ValueError:
            return None
    return value


def _clean(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "<", text.upper()).strip("<")


def _field(value: str, width: int) -> str:
    cleaned = _clean(value)
    if len(cleaned) > width:
        raise ValueError(f"{value!r} does not fit in {width} MRZ characters")
    return cleaned.ljust(width, "<")
