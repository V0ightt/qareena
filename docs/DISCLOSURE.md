# Disclosure: models, tools, data and AI assistants

The challenge rules require disclosing every model, tool, data source and AI coding assistant. This file is updated whenever something is added or removed. Last update: **22 Oct 2026** (build stage).

## AI assistants used to build this project

| Assistant | Used for |
|---|---|
| Claude Code (Anthropic; Claude Opus 5.5) | Research synthesis, design brief, technical design, code |
| ChatGPT / OpenAI Codex (OpenAI) | Earlier planning and the two-slide proposal |

### Models

| Model | Role | Where it runs | License | Status |
|---|---|---|---|---|
| Qwen3.5-4B (Alibaba Qwen), `qwen3.5:4b`, Q4_K_M | VLM reader | Local, Ollama 0.40.8, GPU | Apache-2.0 | **In use** (chosen in the day-1 spike) |
| Qwen3-VL-4B-Instruct, `qwen3-vl:4b` | VLM control baseline | Local, Ollama, GPU | Apache-2.0 | Tested in the day-1 spike, **rejected** (repetition loops) |
| PP-OCRv5 mobile detector + `arabic_PP-OCRv5_rec_mobile` (PaddleOCR models, run via RapidOCR / ONNX Runtime) | OCR reader, label anchoring, evidence boxes | Local, CPU | Apache-2.0 | **In use** |
| Tesseract 5.5 `ara` | OCR fallback | Local, CPU | Apache-2.0 | Planned, not used yet |
| Gemini Flash-class model (Google) | Cloud reference in the offline benchmark only, on synthetic data | Google AI Studio | Google terms | Planned, not used yet |

No model has been trained or fine-tuned. The confidence scorer is a small logistic regression that we fit on our own synthetic development split (`qareena.eval.fit`).

## Libraries and tools in use

- **Runtime:** Python 3.12, NumPy, SciPy, OpenCV (headless), Pillow, RapidOCR, ONNX Runtime, rapidfuzz, httpx, Pydantic and pydantic-settings, PyYAML, Ollama.
- **PDF intake:** pypdfium2 (PDFium, first page only, no scripts).
- **Web app:** FastAPI (Starlette), Uvicorn, python-multipart, Jinja2, SQLite (Python standard library), cryptography (Fernet for data at rest). The stylesheet and the little JavaScript are our own; no CDN, no front-end framework.
- **Synthetic documents and screenshots:** Jinja2 templates, Playwright driving the locally installed Microsoft Edge (headless).
- **Development:** pytest, hypothesis, ruff, uv, GitHub Actions, gitleaks.
- **Fonts in the synthetic documents** (all SIL Open Font License, licence texts in `src/qareena/synth/fonts/`): Noto Naskh Arabic, Amiri, Aref Ruqaa, IBM Plex Sans Arabic, IBM Plex Mono.

**Planned, not used yet:** pillow-heif (iPhone photos). HTMX and Tailwind CSS were planned and then dropped (ADR-007).

The design brief is hand-written HTML/CSS, rendered to PDF with Microsoft Edge (headless). Its fonts are IBM Plex Sans, IBM Plex Sans Arabic and IBM Plex Mono (SIL OFL).

Our own code, not a library:
- Arabic normalisation (`qareena.text.arabic`; ADR-014, not CAMeL Tools).
- MRZ parsing and check digits (`qareena.checks.mrz`).
- Image degradations (`qareena.synth.degrade`). Augraphy and albumentations were considered but not needed.

## Data

- **Synthetic only.** Fictional people and businesses are drawn at random from curated lists in our own code (`qareena.synth.profiles`: common Iraqi given names, laqabs, governorates, trade-name words). National ID and tax numbers start with "99" and family numbers with "ZZ", so they are recognisably fictional; membership and card numbers are random. No LLM or Faker is used to generate them.
- Documents are rendered from our own templates. They are clearly marked SPECIMEN, use ICAO's fictional issuer "UTO", and carry no real logos, stamps or security features.
- Table backgrounds are procedural (generated noise and gradients), not photographs.
- No real identity documents, customer data or personal data are used, ours or anyone else's.
- Physical photos, when taken, are of printed **synthetic** documents only.

# External facts cited in the design

The sources are listed in `docs/RESEARCH.md`. Items marked *preprint* are not peer-reviewed.
