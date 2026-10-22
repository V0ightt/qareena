# Decision log (ADRs)

Short records of the decisions that shape Qareena. Each one gives the context, the decision, the alternatives we rejected, and what would make us revisit it.

The format follows lightweight Architecture Decision Records. New decisions are appended, and superseded ones are marked, never deleted.

---

## ADR-001 · Local-first reading; the cloud is only a benchmark reference
- **Context.** Identity documents are the most sensitive data a wallet holds. Iraq has no comprehensive data-protection law, but CBI E-Payment Regulation 2/2024 requires customer-record confidentiality. Free cloud tiers may use inputs to improve their products. Free limits can also run out mid-demo, as the organisers warn.
- **Decision.** The live path runs on one machine with open-weight models. A Gemini model is used only in the offline benchmark, on synthetic data, to measure how far our local stack is from a strong cloud model.
- **Rejected.**
  - Cloud-first: privacy risk, rate limits and a network dependency in the live demo.
  - Hybrid crop-to-cloud: crops still carry PII.
- **Revisit if** the measured local-vs-cloud gap on the validation split is large *and* ZainCash offers a no-training, in-country endpoint.

## ADR-002 · Two independent readers; evidence boxes come from OCR
- **Context.** No reader is reliable on Arabic alone. In arXiv 2610.25235, Arabic print / handwriting CER was:
  - Qwen3-VL-8B: 0.36 / 0.50
  - PaddleOCR-VL: 0.33 / 0.44
  - Surya: 0.20 / 0.25

  The same paper rated Qwen3-VL Arabic line boxes poor on 12 of 14 pages.
- **Decision.**
  - A VLM extracts fields semantically, by label.
  - A classic OCR engine returns Arabic lines with boxes and scores. It runs on CPU so the GPU stays free for the VLM.
  - Every VLM value must be found in the OCR text. The matched line is the evidence box, and agreement between the readers is a confidence signal.
- **Rejected.** Taking boxes from the VLM, which is unreliable for Arabic. Template coordinates, which break on unseen layouts such as judge input.
- **Revisit if** the VLM's own boxes prove accurate on our validation photos (IoU ≥ 0.5 on ≥ 90% of fields).

## ADR-003 · A deterministic workflow with one bounded loop; no agent framework
- **Context.** Anthropic's *Building effective agents*, which the organisers recommend, says to use the simplest solution and add agency only where the number of steps is unknown. The organisers also say to start with plain function calling and add a framework only when needed.
- **Decision.**
  - A plain-Python state machine with a fixed workflow.
  - The only loop gathers evidence for uncertain fields. Its read-only actions are reread, re-OCR and escalate. It runs under a budget: ≤ 2 actions per field, and the whole packet finishes in ≤ 90 s.
  - In the MVP, the next action is chosen by a written policy table, not by an LLM planner.
  - We call this an *agentic workflow*, not an agent.
  - *Amended 22 Oct after review:* a retake is a durable packet state (`AWAITING_RETAKE`), not a loop action. The applicant answers minutes or hours later, and the packet resumes for that one document.
- **Rejected.**
  - LangGraph or another framework: more surface, no new capability.
  - A free-form tool-using agent: compounding errors and hard to audit.
- **Revisit if** an ablation shows an LLM planner resolves more uncertain fields at equal safety.

## ADR-004 · Decision-critical fields need two agreeing sources
- **Context.** Field-level and packet-level targets have to be consistent. If each of 20 accepted fields were wrong 2% of the time, about 33% of pre-checked packets would contain an error. Even at 1% with 10 critical fields, it would be about 10%.
- **Decision.**
  - About 11 fields per packet are *decision-critical*: the four name parts, ID number, date of birth, ID expiry, trade name, chamber membership number, license expiry and tax ID.
  - A critical field can be auto-accepted only when **two independent sources agree**:
    - front ↔ MRZ, which carries check digits (ID number, DOB, expiry)
    - ID ↔ license ↔ tax card (names, trade name)
    - VLM ↔ OCR, plus agreement on a zoomed re-read (membership number, tax ID)
  - The calibrated threshold for critical fields is certified separately and more strictly than for other fields.
  - Errors in non-critical fields (for example blood type or place of birth) are reported, but they never block a pass on their own.
- **Revisit if** ZainCash's reviewers name a different set of fields as decisive (interview, week 1).

## ADR-005 · Threshold by fixed-sequence testing with Clopper–Pearson bounds
- **Context.** We need a threshold whose error rate among accepted fields holds with high confidence on unseen data. Labels are cheap for digital packets, but every packet costs GPU time.
- **Decision.**
  - Scan a pre-registered grid of thresholds from strictest to loosest. At each one, compute the one-sided 95% Clopper–Pearson upper bound on the error among accepted fields in the calibration split.
  - Stop at the first threshold whose bound exceeds α, and use the last one that passed. Fixed-sequence testing (as in Learn-Then-Test) controls the family-wise error without a Bonferroni correction.
  - α is 1% for critical fields and 2% for the others.
  - If no threshold passes, everything goes to a human. That is a valid result.
- **Honest limit.**
  - The scorer is fitted on the development split, and calibration (150 packets, ~1,650 critical fields) only certifies τ.
  - At α = 1%, 1,150 accepted fields tolerate 5 errors, but 600 tolerate only 1.
  - The grid starts at the strictest τ that still accepts ≥ 300 fields.
  - The bound is pooled across tiers, so per-tier error is reported, not certified.
  - We cannot certify 0.1%, and we say so.

## ADR-006 · Fail fast: test the riskiest assumption on day 1
- **Context.** The riskiest assumption is that a local model on an 8 GB laptop GPU reads Arabic ID fields well enough, with usable signals (valid JSON, logprobs, OCR agreement), at acceptable latency. Everything else depends on it.
- **Decision.**
  - Day 1 is a spike: one quick template, 10 renders and 5 phone photos.
  - Run the candidate VLMs and OCR engines, and measure accuracy, null behaviour, JSON validity, logprobs, VRAM and latency.
  - Write a go/no-go memo before building the data factory.
  - Day 2: a skeleton CLI (photos → VLM → rules → HTML). Day 4: the first end-to-end run with OCR grounding, deepened afterwards.
- **Rejected.** Building all templates first and running the bake-off on day 5.

## ADR-007 · Web stack: FastAPI + Jinja2 + HTMX (RTL)
- **Context.** We need an Arabic RTL reviewer screen with box overlays and a phone capture page, and the stack must be safe against XSS from extracted text.
- **Decision.** FastAPI, server-rendered Jinja2 with autoescape on, HTMX, Tailwind with logical properties and `dir="rtl"`, and a CSP of `default-src 'self'`.
- **Rejected.**
  - Streamlit: no native RTL; its camera input can't select the rear camera.
  - Gradio: rigid layout; its annotation component is pinned to old versions.
  - React SPA: adds a build chain a solo developer doesn't need.
- **Revised while building (22 Oct).**
  - HTMX and Tailwind were dropped for one hand-written stylesheet and about 100 lines of vanilla JavaScript: polling, keyboard shortcuts and box highlighting. The pages need no more, and there is nothing to vendor or build. Every page also works without JavaScript.
  - Progress is polled every 2 s instead of streamed over SSE. A case takes 30–60 s, so polling is simpler and just as responsive.
  - `Referrer-Policy` is `same-origin`, not `no-referrer`: with `no-referrer` browsers send `Origin: null` on form posts, which the cross-site guard must refuse. A real-browser test caught this.

## ADR-008 · Synthetic templates that are unmistakably fake
- **Context.** Realistic synthetic IDs could be misused, and forgery is a crime under Iraqi Penal Code 111/1969.
- **Decision.** Every template:
  - is our own design
  - carries a permanent SPECIMEN watermark (نموذج تجريبي — غير صالح)
  - uses ICAO's fictional issuer "UTO" in the MRZ
  - uses invalid number ranges and generated faces (or none)
  - has no emblems, real stamps, security features or official layouts

## ADR-009 · A person signs off every file; "pre-checked" replaces "auto-pass"
- **Context.** A compliance reviewer would reject "auto-pass with a 5% audit": about 95% of merchant files would get no human sign-off. Approving while a conflict is open also invites automation bias.
- **Decision.**
  - The best outcome is `PRE_CHECKED`: every critical field is confirmed by two sources, and the reviewer approves in one click.
  - Approve stays disabled while any `check` or `review` item is open.
  - Fully automatic approval is a configuration flag, off by default (with a 5% audit sample when on), for ZainCash's compliance team to decide.
  - Reviewers keep sanctions/PEP screening, checking that the business exists, and activity eligibility.
- **Value claim.** Measured as fields a reviewer must touch and seconds per file, not as files nobody looked at.

## ADR-010 · Two sealed test slices, labels by construction, a second author for attacks
- **Context.** The MVP asks for "first results on your test set", and the progress review asks for results on a held-out set "you did not tune on". A single locked test cannot serve both. A hash alone proves the files did not change, not that nobody looked.
- **Decision.**
  - **Test-A** (30 packets) is opened at the MVP. **Test-B** (100 packets, with one held-out layout per document type) is opened once, at the progress review.
  - Both are sealed on 29 Oct (digital) and 1 Nov (photos) via committed SHA-256 manifests and an encrypted archive, whose key is offered to the organisers.
  - Unreadable, ambiguous and legible labels are computed from known occlusion masks. Phone photos follow a shot list, with corner clicks and a rubric.
  - A second person writes the adversarial cases.

## ADR-011 · MVP document set = the brief's; ZainCash's own list comes next
- **Context.** The brief names an ID, a business license and a tax card. ZainCash's published merchant-wallet list is different. It has the national ID plus the housing card, a Chamber of Commerce ID or practice certificate, a lease, and a stamped bank letter, and it **does not list a tax card**. The housing card is required for every wallet, personal ones included, and has no chip.
- **Decision.**
  - The MVP covers the brief's set: ID front and back, Chamber of Commerce ID, and tax card.
  - The **housing card is the first extension** after the MVP, then the bank letter (the account holder's name gives one more cross-check).
  - Document types are configuration (schema + rulebook + template), so each addition is mostly data work.
- **Revisit if** the ZainCash interview names a different priority.

## ADR-012 · Qwen3.5-4B on GPU + RapidOCR (PP-OCRv5 Arabic) on CPU
- **Context.** We have an 8 GB laptop GPU, about 10 GB of free disk, and Windows 10.
  - The multimodal 9B reportedly needs about 8.1 GB of VRAM at 8K context.
  - PP-OCRv6 has no Arabic.
  - Paddle on Windows has known oneDNN problems.
  - Surya is impractical on CPU and its weights carry licence limits.
- **Decision.**
  - `qwen3.5:4b` (3.3 GB, Apache-2.0) via Ollama ≥ 0.40, with `qwen3-vl:4b` as the control.
  - RapidOCR with the PP-OCRv5 Arabic ONNX recogniser on CPU, so GPU and CPU work in parallel.
  - Tesseract 5 `ara` as the fallback.
  - Our own MRZ check-digit code.
- **Revisit if** the day-1 spike shows the 4B under-reading Arabic fields that the larger VLM handles at 100% GPU, or RapidOCR's Arabic order or quality fails.

## ADR-013 · Policy is a release, not a toggle; flags and policy live apart
- **Context.** Big products have admin panels full of switches. Most of Qareena's switches are safety policy. Loosening α or turning off the two-source rule from a screen would silently void the certified error bound.
- **Decision.**
  - Policy lives in versioned, content-hashed `config/policy.yaml`. The router only pre-checks under a policy whose thresholds were certified, which is the certification gate.
  - Engineering feature flags live in typed settings. They double as the ablation switches.
  - The only runtime control is a kill switch that sends every file to human review.
  - The workbench gets a read-only «النظام» page.
  - Every decision carries a fingerprint: policy hash, schema hash, models and flags.
- **Rejected.** A general feature-flag service or UI. It adds surface and lets safety policy change without re-certification.

## ADR-014 · Our own Arabic normaliser instead of CAMeL Tools
- **Context.** We need about 40 lines of explicit folding tables (strict / spelling / sound). CAMeL Tools brings heavy dependencies and folds by its own conventions.
- **Decision.**
  - `qareena.text.arabic` has three explicit, tested layers.
  - Code points are written as escapes, and a source-hygiene test blocks invisible bidi characters (Trojan Source).
- **Revisit if** we need morphology or dialect identification. We do not plan to.

## ADR-015 · OCR evidence counts only at the field's own printed label
- **Context.** In the day-1 spike, 4 of 40 cards (one clean) had a critical name field holding *another* field's value: the grandfather's name returned as the father's. That value is printed on the card, so "the OCR also read it somewhere" would have confirmed a wrong answer and quietly broken the two-source rule (ADR-004).
- **Decision.**
  - `grounding.map_labels` finds each field's printed label in the OCR output: whole tokens, longest label first, an edit budget for OCR misreads, and labels split across two boxes. Each label then claims the text that follows it in Arabic reading order: to its left on the row, else the first row below.
  - Only a match on a line claimed by that field is `Source.OCR`.
  - A value found only at another field's label is `VALUE_AT_OTHER_LABEL`, blocking for critical fields.
  - If the label is not found (an unfamiliar layout), a match on an unclaimed line is `Source.OCR_UNANCHORED`: a scorer feature, never independent evidence.
  - The evidence loop's crop re-OCR is anchored the same way, because a crop can contain the next row's value.
- **Evidence.** 0 stolen lines on 1,072 fields over 11 packets and 3 layouts; 99.6% of value lines claimed by the right field on fresh packets (`docs/spike/2026-10-22-day1-stack-decision.md`).
- **Cost.** Fewer acceptances on layouts whose labels we do not know: their critical fields need MRZ or cross-document evidence. This is deliberate.
- **Rejected.** Template coordinates per layout (they do not survive a new layout or a phone photo), and global one-to-one assignment of OCR lines to fields (legitimately repeated names, e.g. a father and grandfather both called علي, break it).

## ADR-016 · Printed Arabic labels are the JSON keys; label echoes get one crop re-read
- **Context.** With English keys the model matched `issuing_authority` to the Kurdish label «دەسەڵاتی دەرکردن» (which means "issuing authority") 20 times out of 20. Listing labels as never-values did not help.
- **Decision.**
  - JSON keys are the printed Arabic labels, plus one paragraph on how bilingual labels look (variant v3 in the A/B: exact 237 → 256 of 342, same latency, best on the held-out layout).
  - A required field answered with a label or with spelled-out glyphs is read once more, from a crop at its label (`Pipeline._recover`, at most 4 per document).
  - The new answer has no logprobs, so it scores lower. It goes through every rule, and is marked `NOTE_RECOVERED_FROM_CROP` with the first answer kept for the trace.
- **Rejected.** {label, value} pairs (v2). Same accuracy, 60% slower.

## ADR-017 · The input size is certified policy, chosen per document type
- **Context.** The day-1 spike fixed 1600 px for ID cards. The business documents read far worse at that size: 151/252 exact vs 212/252 at 1024 px. Every right-to-left reversal and column swap happened at 1600 px, and so did a repetition loop.
- **Decision.**
  - `policy.reader` sets the long side sent to the VLM, per document type: chamber 1280 px, tax card 1024 px, ID cards 1600 px (232/336 exact; 1280 px equal, 1024 px 208).
  - A looping or unparseable answer gets one retry at the other size.
  - Because size changes the readings, it lives in the content-hashed policy: changing it voids the certificate (ADR-013).
- **Rejected.** Choosing size per layout (the layout is unknown at read time), and a prompt sentence about RTL order (no measurable effect).
