# Day-1 spike: stack decision memo (22 Oct 2026)

**Question (ADR-006):** can a local VLM on this laptop read our Arabic SPECIMEN cards, with usable confidence signals, at acceptable latency? This is the riskiest assumption, so it was tested before building the data factory.

**Setup:**
- 10 fictional cards (front and back), each with one degraded copy (blur, tilt, low JPEG, low light, or glare washing out one field), giving 40 images.
- RTX 4060 Laptop 8 GB, Ollama 0.40.8, RapidOCR on CPU.
- Tool: `tools/spike/day1.py`. Raw results: `var/spike/` (not committed).
- These are digital renders only. Phone photos come later (shot list), so **these numbers say nothing yet about real photos.**

## Run 1: candidate VLMs (prompt v1)

| | qwen3.5:4b | qwen3-vl:4b |
|---|---|---|
| Valid JSON | **40/40** | 2/40 |
| Output tokens per answer | ~164 | ~1,020 (hits the cap: repetition loop) |
| Latency p50 / p95 | **5.1 s / 6.7 s** | 21.5 s / 24.8 s |
| Field exact (all conditions) | 238/366 (65%) | 6/366 |
| GPU residency | 100% GPU, 3.36 GB at 8k context | 100% GPU, 4.18 GB |
| Logprobs with an image | returned; tokens rebuild the answer 40/40; no U+FFFD tokens | — |

**Decision: `qwen3.5:4b`. `qwen3-vl:4b` is rejected** (repetition loops, 4× slower, no usable output).

## Error analysis (qwen3.5:4b, run 1): what was wrong, and the fix

| Error class | Evidence | Fix (where) |
|---|---|---|
| **Kurdish label echo**, by far the largest | `issuing_authority` 0/20, returned as a dotted spelling of the Kurdish label «دەسەڵاتی دەرکردن». Name fields returned «ناو», «باوك», «دايك» | The prompt lists every printed label as never-a-value (catalog-driven). Label-echo detection ignores punctuation. A garbled-text detector nulls spelled-out glyphs (`checks/fields.py`) |
| MRZ filler miscount | `mrz_line_2`/`3` 0/20: two or three extra `<` | `canonicalize_fillers` re-pads filler only |
| O↔0 in class-fixed MRZ positions | issuer «UT0», access number «0E…» | `repair_types` (Iraqi TD1 profile), kept **only if every check digit passes** (`read_td1`) |
| Plausible wrong digits under blur | `996345494950` read as `997345496950` | Not fixable by reading. Caught by the MRZ cross-check, because critical fields need two sources (ADR-004) |
| Labelling bug in the spike itself | "false fills" on glare, but the model had read through glare that did not hide the text | Unreadable is now defined by true visibility (≥ 90% wash-out over ≥ 50% of the field) |
| OCR server detector | 7–30 s per image, no accuracy gain over mobile (93 vs 94/125 hits) | PP-OCRv5 **mobile** detector |
| PP-OCRv6 | RapidOCR refuses to route it for Arabic | PP-OCRv5 Arabic recogniser |

## Run 2: qwen3.5:4b with the fixes

Same 40 images, `tools/spike/day1.py --out var/spike/day1b`.

| | Run 1 | Run 2 |
|---|---|---|
| Field exact (raw VLM answer) | 238/366 (65%) | 235/366 (64%) |
| After the deterministic checks (label echo, garbled text → null) | — | 235/366: **52 wrong answers became nulls, 0 right answers lost** |
| False fills on truly unreadable fields | 14/14 raw | 4/14 after checks (see below) |
| MRZ valid after `read_td1` | lines 2–3 0/20 raw | **20/20** |
| Latency p50 / p95 | 5.1 / 6.7 s | 5.3 / 6.4 s |

**What did not work:** listing every printed label as never-a-value did not change the reading (235 vs 238). `issuing_authority` and `place_of_birth` were still returned as their Kurdish label 20/20 times.

**A new, dangerous error class: field shift.** In 4 of 40 images (one clean) a critical name field held *another field's* value: father ← grandfather, grandfather ← laqab, father ← mother's father. The value is printed on the card, so grounding against "any OCR line" would have confirmed it. See "Label-anchored grounding" below.

**The 4 false fills left after the checks**, and what now catches them:

| Field | Answer | Caught by |
|---|---|---|
| laqab | «ال» (glare hid the rest) | `NAME_TRUNCATED` (new) |
| date_of_birth | «?1995» | `DATE_UNPARSEABLE` (new) |
| blood_type | «AB» (sign washed out) | `VALUE_NOT_ALLOWED` (pipeline enum rule) |
| expiry_date | «2030-03-26»: ISO, taken from the MRZ, not the printed «2030/03/26» | `DATE_FORMAT_DIFFERS` (new) |

## Prompt A/B: how to ask so the model stops returning labels

`tools/spike/prompt_ab.py`: 6 fictional people × layouts A, B and the **held-out C** × front and back = 36 clean images, 342 legible fields.

| Variant | Exact | Label echoes | Held-out layout C | p50 |
|---|---|---|---|---|
| v0: English keys + label list (Run 2) | 237 (69%) | 26 | 70/114 | 4.7 s |
| v1: Arabic printed labels as JSON keys | 235 | 24 | 70/114 | 5.3 s |
| v2: English keys, {label, value} pairs | 256 (75%) | 4 | 77/114 | 8.6 s |
| **v3: v1 + one paragraph on how bilingual labels look** | **256 (75%)** | 13 | **83/114** | **5.4 s** |

**Decision: v3** (now `readers/prompts.py`). Same accuracy as v2 at 63% of the latency, and the best on the unseen layout. The remaining echoes are nulled by the checks and recovered by a crop re-read at the label (`Pipeline._recover`).

Errors still seen with v3, all caught downstream:
- The model "corrected" «مديرية البطاقة النموذجية» to the real-world «مديرية البطاقة الوطنية» (a prior, not a reading). Anchored OCR disagrees.
- «النجف» read as «النصف». Anchored OCR disagrees.
- A dropped digit in the national ID (11 digits). Format rule.
- O↔0 in codes. Check digits for the MRZ; OCR disagreement for the family number.

## Label-anchored grounding (no VLM involved)

`tools/spike/anchor_eval.py` renders packets in every layout, runs the OCR reader, maps each field's printed label to the value lines after it (`grounding.map_labels`), and compares with the true value boxes. "Stolen" = a field's true value line claimed by a *different* field, which is exactly the case that would confirm a field shift.

Rules were tuned on 3 packets (seed 7), then checked once on 8 fresh packets (seed 2026):

| | Fields | Claimed right | Stolen |
|---|---|---|---|
| Tuning (3 packets × 3 layouts × 4 documents) | 314 | 310 (98.7%) | **0** |
| Check (8 packets × 3 layouts × 4 documents) | 758 | 755 (99.6%) | **0** |

On the held-out layout C most labels are reworded and are *not* found. Those fields fall back to unanchored matching, which the rules treat as a signal only: on an unfamiliar layout a critical field needs MRZ or cross-document evidence.

## Quality gate on the new layouts

Clean renders of layouts B and C first gave false retakes:
- CUT_CORNER: a full-width band (a header strip, the MRZ zone) has four corners on the frame.
- GLARE: layout C prints the MRZ on pure white.

Fixes:
- An outline with four corners on the frame spans it, so the gate reads the whole image.
- A clipped blob that encloses dark ink is print, not glare. Glare measurements on real hotspots are unchanged.

Result: **0/124 false retakes** on clean renders of all layouts.
