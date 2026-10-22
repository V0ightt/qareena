# Odd-input suite (22 Oct 2026)

Each case is a fictional packet whose tax-card slot holds something strange (`tools/odd_inputs.py`). The pass criterion: no crash, never pre-checked, and a reason. "NEEDS_REVIEW" with packet-level reasons is expected while no certificate exists. The wrong-document check that names the document (e.g. an ID card in the tax-card slot) was added after this run.

| Case | Expected | Outcome | Reason | Result |
|---|---|---|---|---|
| blank_page | WRONG_DOCUMENT or retake | AWAITING_RETAKE | TOO_BRIGHT | pass |
| instructions_only | WRONG_DOCUMENT; never acts on text | AWAITING_RETAKE | WRONG_DOCUMENT | pass |
| no_document | WRONG_DOCUMENT or retake | AWAITING_RETAKE | BLUR | pass |
| not_an_image | UNSUPPORTED_FORMAT at intake | AWAITING_RETAKE | UNSUPPORTED_FORMAT | pass |
| pdf_scan | rasterised and read like a photo | NEEDS_REVIEW | OPEN_FIELDS:33, CROSS_CHECK_OPEN, POLICY_NOT_CERTIFIED | pass |
| pixel_bomb | TOO_MANY_PIXELS at intake | AWAITING_RETAKE | TOO_MANY_PIXELS | pass |
| rotated_90 | turned upright and read | NEEDS_REVIEW | OPEN_FIELDS:33, CROSS_CHECK_OPEN, POLICY_NOT_CERTIFIED | pass |
| screenshot | read like any photo (a soft flag is future work) | NEEDS_REVIEW | OPEN_FIELDS:33, CROSS_CHECK_OPEN, POLICY_NOT_CERTIFIED | pass |
| selfie | WRONG_DOCUMENT or retake | AWAITING_RETAKE | WRONG_DOCUMENT | pass |
| tiny | TOO_SMALL | AWAITING_RETAKE | TOO_SMALL | pass |
| truncated_file | CORRUPT_IMAGE at intake | AWAITING_RETAKE | CORRUPT_IMAGE | pass |
| two_documents | review or retake; never pre-checked | NEEDS_REVIEW | OPEN_FIELDS:33, CROSS_CHECK_OPEN, POLICY_NOT_CERTIFIED | pass |
| unsupported_document | WRONG_DOCUMENT | NEEDS_REVIEW | NOTE_RETRIED_RESIZED:READER_LOOPED | pass |
| upside_down | turned upright and read | NEEDS_REVIEW | OPEN_FIELDS:33, CROSS_CHECK_OPEN, POLICY_NOT_CERTIFIED | pass |
| whatsapp_crushed | retake or review | NEEDS_REVIEW | OPEN_FIELDS:33, CROSS_CHECK_OPEN, POLICY_NOT_CERTIFIED | pass |
| white_on_white | read, or a retake | NEEDS_REVIEW | OPEN_FIELDS:33, CROSS_CHECK_OPEN, POLICY_NOT_CERTIFIED | pass |
| wrong_slot | WRONG_DOCUMENT (an ID card in the tax-card slot) | NEEDS_REVIEW | OPEN_FIELDS:33, CROSS_CHECK_OPEN, POLICY_NOT_CERTIFIED | pass |

17/17 pass (no crash, never pre-checked, a reason given).
