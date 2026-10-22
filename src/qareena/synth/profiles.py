"""Fictional Iraqi-style identities for synthetic documents.

Every value is invented and seeded. Numbers live in ranges chosen to be obviously fictional
(national numbers start with "99", family numbers with "ZZ", the MRZ issuer is ICAO's
fictional state "UTO"), so a synthetic card can never be mistaken for or collide with a real
one. Names are common Iraqi names combined at random; any resemblance to a real person is
coincidental, and the cards say SPECIMEN on their face.
"""

from __future__ import annotations

import datetime as dt
import random
import string
from dataclasses import asdict, dataclass

from qareena.checks.mrz import make_td1

# (Arabic, Latin transliteration for the MRZ)
MALE = [
    ("علي", "ALI"),
    ("حسين", "HUSSEIN"),
    ("محمد", "MOHAMMED"),
    ("عمر", "OMAR"),
    ("مصطفى", "MUSTAFA"),
    ("كرار", "KARRAR"),
    ("عباس", "ABBAS"),
    ("حسن", "HASSAN"),
    ("أحمد", "AHMED"),
    ("يوسف", "YOUSIF"),
    ("عبد الله", "ABDULLAH"),
    ("عبد الأمير", "ABDULAMEER"),
    ("عبد الزهرة", "ABDULZAHRA"),
    ("نور الدين", "NOORALDEEN"),
    ("كاظم", "KADHIM"),
    ("جاسم", "JASIM"),
    ("گاطع", "GATEA"),
    ("سجاد", "SAJJAD"),
    ("مهدي", "MAHDI"),
    ("ليث", "LAITH"),
    ("حيدر", "HAIDER"),
    ("مرتضى", "MURTADHA"),
    ("فلاح", "FALAH"),
    ("ستار", "SATTAR"),
    ("رعد", "RAAD"),
]
FEMALE = [
    ("زينب", "ZAINAB"),
    ("فاطمة", "FATIMA"),
    ("مريم", "MARYAM"),
    ("نور", "NOOR"),
    ("سارة", "SARA"),
    ("رقية", "RUQAYA"),
    ("هدى", "HUDA"),
    ("آمنة", "AMNA"),
    ("بتول", "BATOOL"),
    ("زهراء", "ZAHRAA"),
]
LAQAB = [
    ("الجبوري", "ALJUBOURI"),
    ("الربيعي", "ALRUBAIE"),
    ("التميمي", "ALTAMIMI"),
    ("العبيدي", "ALOBAIDI"),
    ("الساعدي", "ALSAADI"),
    ("الدليمي", "ALDULAIMI"),
    ("الخفاجي", "ALKHAFAJI"),
    ("الموسوي", "ALMOUSAWI"),
    ("الشمري", "ALSHAMMARI"),
    ("الزبيدي", "ALZUBAIDI"),
]
CITIES = [
    "بغداد",
    "البصرة",
    "النجف",
    "كربلاء",
    "الحلة",
    "الديوانية",
    "الناصرية",
    "العمارة",
    "الكوت",
    "الموصل",
]
BLOOD = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
TRADE_WORDS = ["أسواق", "مكتب", "معرض", "محلات", "شركة"]
TRADE_KINDS = ["الرافدين", "دجلة", "النخيل", "الفرات", "بابل", "سومر", "الأمانة", "النور", "الواحة", "الشروق"]
ACTIVITIES = [
    "بيع المواد الغذائية بالمفرد",
    "بيع الهواتف وملحقاتها",
    "بيع الملابس",
    "مواد إنشائية",
    "قرطاسية",
]
CHAMBER_CLASSES = ["ممتازة", "الأولى", "الثانية", "الثالثة", "الرابعة", "الخامسة"]
TAX_OFFICES = ["فرع الكرخ", "فرع الرصافة", "فرع البصرة", "فرع النجف", "فرع كربلاء"]
ISSUER = "مديرية البطاقة النموذجية"  # fictional issuing authority


@dataclass(frozen=True)
class Person:
    given_name: str
    father_name: str
    grandfather_name: str
    laqab: str | None
    mother_name: str
    mother_father_name: str
    sex: str  # ذكر | أنثى
    blood_type: str
    date_of_birth: dt.date
    place_of_birth: str
    national_id_number: str
    card_access_number: str
    family_number: str
    issue_date: dt.date
    expiry_date: dt.date
    mrz: tuple[str, str, str]

    @property
    def full_name(self) -> str:
        parts = [self.given_name, self.father_name, self.grandfather_name]
        return " ".join(parts + ([self.laqab] if self.laqab else []))

    def as_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, dt.date):
                d[k] = v.isoformat()
        d["mrz"] = list(self.mrz)
        return d


@dataclass(frozen=True)
class Business:
    trade_name: str
    activity: str
    chamber_governorate: str
    membership_number: str
    member_class: str
    chamber_issue_date: dt.date
    chamber_valid_until: dt.date
    tax_id: str
    tax_office: str
    tax_issue_date: dt.date
    tax_valid_until: dt.date


def make_person(rng: random.Random, *, today: dt.date) -> Person:
    male = rng.random() < 0.7
    given, given_lat = rng.choice(MALE if male else FEMALE)
    father, father_lat = rng.choice(MALE)
    grandfather, _ = rng.choice([m for m in MALE if m[0] != father])
    laqab, laqab_lat = rng.choice(LAQAB) if rng.random() < 0.8 else (None, None)
    mother, _ = rng.choice(FEMALE)
    mother_father, _ = rng.choice(MALE)
    dob = _random_date(rng, dt.date(1960, 1, 1), dt.date(today.year - 19, 12, 31))
    # cards exist since 2016 and last 10 years: issue late enough that the card is still valid
    earliest = max(dt.date(2016, 1, 1), _add_years(today, -10) + dt.timedelta(days=60))
    issue = _random_date(rng, earliest, today - dt.timedelta(days=30))
    expiry = _add_years(issue, 10)
    nid = "99" + "".join(rng.choice(string.digits) for _ in range(10))
    access = "".join(rng.choice(string.ascii_uppercase) for _ in range(2)) + "".join(
        rng.choice(string.digits) for _ in range(7)
    )
    family = "ZZ" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(16))
    sex = "ذكر" if male else "أنثى"
    mrz = make_td1(
        document_number=access,
        optional_1=nid,
        birth_date=dob,
        sex="M" if male else "F",
        expiry_date=expiry,
        primary_name=laqab_lat or father_lat,
        secondary_names=[given_lat],
    )
    return Person(
        given_name=given,
        father_name=father,
        grandfather_name=grandfather,
        laqab=laqab,
        mother_name=mother,
        mother_father_name=mother_father,
        sex=sex,
        blood_type=rng.choice(BLOOD),
        date_of_birth=dob,
        place_of_birth=rng.choice(CITIES),
        national_id_number=nid,
        card_access_number=access,
        family_number=family,
        issue_date=issue,
        expiry_date=expiry,
        mrz=mrz,
    )


def make_business(rng: random.Random, *, today: dt.date) -> Business:
    # chamber membership is annual: issued this year, valid until 31 December
    jan1 = dt.date(today.year, 1, 1)
    c_issue = _random_date(rng, jan1, max(jan1, today - dt.timedelta(days=10)))
    t_issue = _random_date(rng, dt.date(today.year - 1, 1, 1), today - dt.timedelta(days=10))
    return Business(
        trade_name=f"{rng.choice(TRADE_WORDS)} {rng.choice(TRADE_KINDS)}",
        activity=rng.choice(ACTIVITIES),
        chamber_governorate=f"غرفة {rng.choice(CITIES)} النموذجية",  # fictional chamber name
        membership_number=str(rng.randint(1000, 99999)),
        member_class=rng.choice(CHAMBER_CLASSES),
        chamber_issue_date=c_issue,
        chamber_valid_until=dt.date(c_issue.year, 12, 31),
        tax_id="99" + "".join(rng.choice(string.digits) for _ in range(8)),
        tax_office=rng.choice(TAX_OFFICES),
        tax_issue_date=t_issue,
        tax_valid_until=dt.date(t_issue.year + 1, 12, 31),
    )


def _random_date(rng: random.Random, lo: dt.date, hi: dt.date) -> dt.date:
    return lo + dt.timedelta(days=rng.randint(0, (hi - lo).days))


def _add_years(d: dt.date, years: int) -> dt.date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 February
        return d.replace(year=d.year + years, day=28)
