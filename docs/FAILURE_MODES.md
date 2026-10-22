# Observed failure modes

Every entry here was seen in our own runs, not imagined. Each one lists the evidence, what the system does about it now, and what risk remains. The planned red-team catalogue is in `DESIGN.md` §12. This file records what actually happened. Numbers are from fictional SPECIMEN documents only (`docs/spike/`, dev split). No test split has been opened.

## FM-1 · Field shift: one field's value returned under another field's key

- **Seen.** In the day-1 spike, 4 of 40 cards had a critical name field holding another field's value; one was a clean render:
  - grandfather → father,
  - laqab → grandfather,
  - mother's father → father.

  On the first end-to-end tax card, the issue and valid-until dates came back swapped.
- **Why it is dangerous.** The value *is* printed on the card. A check of the form "the OCR read it somewhere" confirms it, and the two-source rule is quietly satisfied by a wrong answer.
- **Now.**
  - OCR evidence counts only on lines that follow the field's *own* printed label (ADR-015).
  - A value found only at another field's label is `VALUE_AT_OTHER_LABEL`, blocking for critical fields.
  - On the tax card, both swapped dates were caught this way.
  - The label mapper claimed a wrong field's value line 0 times on 1,072 fields over 11 packets and 3 layouts.
- **Remaining risk.**
  - On a layout whose labels we cannot find, OCR is only a signal. Critical fields then need MRZ or cross-document evidence, which costs coverage on unfamiliar layouts. This is deliberate.
  - A shift between two fields that hold the same value is invisible, and harmless.

## FM-2 · The Kurdish label returned as the value

- **Seen.** With English JSON keys, the VLM returned «دەسەڵاتی دەرکردن» ("issuing authority") as the issuing authority 20 times out of 20, and «شوێنی لەدایکبوون» ("place of birth") as the place of birth. It matched the key to the Kurdish label by meaning.
- **Now.**
  - Printed Arabic labels are the JSON keys (prompt v3; exact 237 → 256 of 342).
  - Label echoes are detected and never kept. They get one crop re-read at the label, and the new answer goes through every rule.
- **Remaining risk.** On layout A the echo still happens for those two fields. Each recovery costs about 2 s of VLM time.

## FM-3 · Arabic read left to right, and columns swapped

- **Seen.**
  - On wider business documents read at 1600 px, Arabic sometimes came back character-reversed («داجس عر مظا يدييزلا» for «سجاد رعد كاظم الزبيدي»), with the left and right columns swapped.
  - At 1600 px: 4 of 36 documents reversed, 3 with swapped dates, and 1 generation stopped in a repetition loop.
  - At 1280 px and 1024 px: none.
- **Now.**
  - The input size is certified policy per document type: chamber 1280 px, tax card 1024 px (exact 151 → 212 of 252).
  - A reading that the OCR at the label matches *backwards* is `READ_REVERSED`. It gets a crop re-read and is blocked if it stays reversed.
  - A looping generation is retried once at another size.
- **Remaining risk.** Phone photos have not been measured yet.

## FM-4 · A finger hides the end of a value, and both readers agree on the fragment

- **Seen.** On the first dev run, a finger covered the end of the trade name on two documents. The VLM, the OCR, and the comparison between documents all agreed on the visible fragment «محلا». That gave a truncated value two "independent" sources and no reason to look twice. «الك» (from «الكوت») was the same pattern.
- **Why it is dangerous.** Independence of readers does not help when both see the same occluded image.
- **Now.**
  - The quality gate asks for a retake when a saturated, skin-toned blob enters the document from its edge (`FINGER`). A face photo printed inside an ID card does not trigger it.
  - On a fresh dev seed the gate matched the construction labels on 120 of 120 documents, with 0 false retakes on 160 clean renders.
  - Text cut down to the article is `VALUE_TRUNCATED`.
- **Remaining risk.** A finger the gate misses (unusual skin tone or lighting) still produces agreeing fragments. Phone photos with real fingers are needed to measure this.

## FM-5 · The model "corrects" the document from what it knows about real ones

- **Seen.** On layout B, the VLM read «مديرية البطاقة النموذجية» (our SPECIMEN text) as «مديرية البطاقة الوطنية», the real Iraqi wording. It read the prior, not the page.
- **Now.** Anchored OCR disagrees (`NOT_GROUNDED`), so the field cannot be accepted.
- **Remaining risk.** The same prior could apply to real documents that deviate from the usual wording. This is why OCR agreement is required.

## FM-6 · Dropped letters in runs of similar glyphs

- **Seen.**
  - On dev, names were the weakest kind: 72% exact.
  - Most errors were dropped letters in runs of connected, similar-looking glyphs:
    - «التميمي» → «التيمي»
    - «الزبيدي» → «الزيدي»
    - «مرتضى» → «مرضى»
  - «ظ» was read as «ز».
  - At the field's own label, the OCR read names correctly 84% of the time vs 72% for the VLM. A zoomed re-read by the same VLM usually repeated its own error.
- **Now.**
  - These fields are not grounded, so they go to a person.
  - The workbench shows the OCR's reading next to the VLM's, with one-click "use this reading".
- **Tested and rejected: letting the VLM verify the OCR's candidate** (`tools/spike/verify_ab.py`, 614 candidates where the readers disagreed).
  - Asked whether the crop shows exactly that value, the 4B VLM said yes to 12 of 226 correct candidates and to 11 of 388 wrong ones.
  - It rejects almost everything, and its rare yes is right only about half the time. That is no evidence at all.
  - So the system still never replaces a value on its own. The OCR's reading stays a suggestion for the reviewer.
