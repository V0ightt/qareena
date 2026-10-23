"""Everything the user reads, in Arabic: Iraqi Arabic for applicants, plain MSA-leaning Arabic
for reviewers. Every message must be reviewed by a native Iraqi speaker before a pilot.

Reason codes stay in the trace and the audit log; people see these sentences. An unknown code
falls back to the code itself, so nothing is ever hidden.
"""

from __future__ import annotations

# Applicant: one message per retake reason (DESIGN §8.2).
RETAKE: dict[str, str] = {
    "BLUR": "الصورة مغبّشة. ثبّت إيدك شوية وصوّر مرة لخ.",
    "GLARE": "اكو لمعة ضوه على البطاقة. ميّلها شوية أو بعّدها عن الضوه.",
    "CUT_CORNER": "زاوية البطاقة مگصوصة. خلّي البطاقة كلها داخل الإطار.",
    "FINGER": "إصبعك مغطّي جزء من البطاقة. شيله وصوّر مرة لخ.",
    "TOO_DARK": "الصورة ظلمة هواية. صوّر بمكان بيه ضوه أكثر.",
    "TOO_BRIGHT": "الصورة فاتحة هواية والكتابة مو واضحة. بعّد الموبايل عن الضوه القوي.",
    "TOO_FAR": "البطاقة صغيرة بالصورة. قرّب الموبايل شوية.",
    "TOO_SMALL": "الصورة صغيرة هواية. صوّر بدقة أعلى.",
    "TILTED": "البطاقة مايلة. خلّي الموبايل فوگها عدل.",
    "WRONG_DOCUMENT": "هاي مو الوثيقة المطلوبة. صوّر الوثيقة الصحيحة رجاءً.",
    "DOC_MISSING": "هاي الوثيقة بعدها ما وصلتنا. صوّرها ودزها رجاءً.",
    "UNSUPPORTED_FORMAT": "نوع الملف مو مدعوم. دز صورة JPG أو PNG أو ملف PDF.",
    "FILE_TOO_LARGE": "الملف كبير هواية. دز صورة أصغر.",
    "CORRUPT_IMAGE": "ما گدرنا نفتح الصورة. صوّرها مرة لخ.",
    "NOT_A_DOCUMENT": "ما لگينا وثيقة بالصورة. صوّر الوثيقة نفسها.",
}

OUTCOME_APPLICANT: dict[str, str] = {
    "PRE_CHECKED": "وصلتنا كل الوثائق وهي واضحة. موظفنا راح يراجعها ويأكدها.",
    "NEEDS_REVIEW": "وصلتنا الوثائق. موظفنا راح يراجعها، وإذا احتجنا شي راح نتواصل وياك.",
    "AWAITING_RETAKE": "نحتاج نعيد تصوير وثيقة أو أكثر. شوف التفاصيل جوه.",
}

OUTCOME: dict[str, str] = {
    "PRE_CHECKED": "جاهز للموافقة",
    "NEEDS_REVIEW": "يحتاج مراجعة",
    "AWAITING_RETAKE": "بانتظار إعادة تصوير",
}

BAND: dict[str, str] = {"accept": "مطابق", "check": "تأكد", "review": "راجع"}

STATUS: dict[str, str] = {
    "QUEUED": "بالانتظار",
    "PROCESSING": "جاري الفحص",
    "DONE": "اكتمل",
    "FAILED": "تعذّر الفحص",
}

DOCUMENT: dict[str, str] = {
    "iq_national_card_front": "البطاقة الوطنية (الوجه)",
    "iq_national_card_back": "البطاقة الوطنية (الظهر)",
    "chamber_of_commerce_id": "هوية غرفة التجارة",
    "tax_card": "البطاقة الضريبية",
}

# Reviewer: what each reason code means and what to do about it.
REASON: dict[str, str] = {
    "LABEL_ECHO": "القارئ أعاد اسم الحقل بدل قيمته",
    "GARBLED_TEXT": "القراءة حروف متقطعة غير مفهومة",
    "READ_REVERSED": "القراءة معكوسة الاتجاه (من اليسار لليمين)",
    "REQUIRED_FIELD_MISSING": "حقل مطلوب لم يُقرأ",
    "FORMAT_INVALID": "الصيغة لا تطابق الصيغة المتوقعة",
    "FORMAT_NOT_DIGITS": "يجب أن يحتوي أرقاماً فقط",
    "VALUE_NOT_ALLOWED": "قيمة غير مسموحة لهذا الحقل",
    "NAME_NOT_ARABIC_LETTERS": "الاسم يحتوي رموزاً غير عربية",
    "NAME_TRUNCATED": "الاسم مقروء جزئياً",
    "VALUE_TRUNCATED": "القيمة مقروءة جزئياً",
    "DATE_UNPARSEABLE": "التاريخ غير مكتمل أو غير صالح",
    "DATE_FORMAT_DIFFERS": "التاريخ مكتوب بصيغة غير المطبوعة (ربما أُخذ من مكان آخر)",
    "DATE_IN_FUTURE": "تاريخ الإصدار في المستقبل",
    "DATE_ORDER": "ترتيب التواريخ غير منطقي",
    "DATE_INCONSISTENT": "مدة الصلاحية لا تطابق القاعدة",
    "EXPIRED": "الوثيقة منتهية الصلاحية",
    "UNDERAGE": "العمر أقل من ١٨ سنة",
    "NOT_GROUNDED": "القارئ الثاني (OCR) لم يجد القيمة عند هذا الحقل",
    "OCR_AMBIGUOUS_BOX": "القيمة مكتوبة بأكثر من مكان؛ تأكد من الموضع",
    "VALUE_AT_OTHER_LABEL": "القيمة موجودة لكن تحت حقل آخر (قد تكون من حقل مختلف)",
    "LABEL_NOT_FOUND": "لم نجد اسم الحقل المطبوع على الوثيقة",
    "REREAD_DISAGREES": "القراءة المكبّرة أعطت قيمة مختلفة",
    "MRZ_CHECK_FAILED": "أرقام التحقق في السطر الآلي لا تتطابق",
    "MRZ_VIZ_MISMATCH": "القيمة لا تطابق السطر الآلي (MRZ)",
    "NAME_CONFLICT": "الاسم يختلف بين الوثائق",
    "NAME_SOUND_ALIKE": "الاسم متشابه لفظاً بين الوثائق لكن بحروف مختلفة",
    "NAME_VARIANT": "اختلاف إملائي بسيط في الاسم بين الوثائق",
    "TRADE_NAME_CONFLICT": "الاسم التجاري يختلف بين الوثائق",
    "SUSPICIOUS_TEXT": "على الوثيقة نص يشبه تعليمات للنظام؛ تم تجاهله",
    "WRONG_DOCUMENT": "يبدو أنها ليست الوثيقة المطلوبة",
    "READ_FAILED": "تعذّرت قراءة الوثيقة آلياً",
    "SYSTEM_DEGRADED": "أحد القارئين متوقف؛ كل الحقول تحتاج مراجعة",
    "KILL_SWITCH": "الفحص الآلي موقوف من صفحة النظام",
    "POLICY_NOT_CERTIFIED": "سياسة الثقة الحالية غير معتمدة؛ لا شيء يُقبل آلياً",
    "CROSS_CHECK_OPEN": "مطابقة بين الوثائق تحتاج قراراً",
    "CROSS_CHECK_INCOMPLETE": "مطابقة بين الوثائق غير مكتملة",
    "NOTE_MRZ_REPAIRED": "صُحّحت قراءة السطر الآلي وأكدتها أرقام التحقق",
    "NOTE_RECOVERED_FROM_CROP": "أُعيدت القراءة من صورة مكبّرة للحقل",
    "NOTE_OCR_CONSENSUS": "اعتُمدت قراءة القارئ الثاني: متطابقة حرفياً على الوثيقتين، والقارئ الأول قرأ غيرها",
    "NAME_MISSING_LAQAB": "اللقب غير موجود في إحدى الوثائق",
    "NAME_MISSING_GRANDFATHER": "اسم الجد غير موجود في إحدى الوثائق",
    "NOTE_MISSING_LAQAB": "اللقب غير مذكور في إحدى الوثائق (مقبول)",
    "NOTE_MISSING_GRANDFATHER": "اسم الجد غير مذكور في إحدى الوثائق (مقبول)",
}

# Cross-document checks: "<RULE>_<RESULT>" codes.
CROSS_RESULT: dict[str, str] = {
    "SAME": "متطابق",
    "VARIANT": "اختلاف إملائي",
    "SOUND_ALIKE": "تشابه لفظي بحروف مختلفة",
    "PARTIAL": "تطابق جزئي",
    "CONFLICT": "تعارض",
    "MISSING": "ناقص",
}

NOT_CHECKED = "ما تحققنا من أصالة الوثائق ولا من مطابقة الوجه."


def retake(code: str) -> str:
    """The applicant's message for a retake code, naming the document when we know what it is."""
    base, _, detail = code.partition(":")
    if base == "WRONG_DOCUMENT" and detail in DOCUMENT:
        return f"هاي تبين {DOCUMENT[detail]}، مو الوثيقة المطلوبة هنا. صوّر الوثيقة الصحيحة رجاءً."
    return RETAKE.get(base, code)


def reason(code: str) -> str:
    base = code.split(":", 1)[0]
    if base.startswith("QUALITY_"):
        return "جودة الصورة: " + RETAKE.get(base.removeprefix("QUALITY_"), base)
    if base.startswith(("OPEN_FIELDS", "RETAKES_EXHAUSTED", "NOTE_RETRIED_RESIZED")):
        return {
            "OPEN_FIELDS": "حقول تحتاج قراراً",
            "RETAKES_EXHAUSTED": "استُنفدت محاولات إعادة التصوير",
            "NOTE_RETRIED_RESIZED": "أُعيدت القراءة بحجم صورة آخر",
        }[base] + (f" ({code.split(':', 1)[1]})" if ":" in code else "")
    for rule in ("OWNER_NAME_", "TRADE_NAME_"):
        if base.startswith(rule) and base.removeprefix(rule) in CROSS_RESULT:
            return CROSS_RESULT[base.removeprefix(rule)]
    return REASON.get(base, code)
