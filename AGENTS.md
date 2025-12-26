# Repository Guidelines

## Project Structure & Module Organization
- `ocr_processor.py` is the main entry point and contains the OCR pipeline (API calls, PDF validation, PDF generation, batch mode).
- `batch_ocr.py.example` is a reference notebook/script from Mistral’s cookbook; it is not used by the app runtime.
- `pyproject.toml` defines dependencies and Python version (3.13+).
- Outputs are written next to inputs by default: `*_ocr.json` and `*_with_ocr.pdf`.

## Build, Test, and Development Commands
- `uv sync` installs dependencies with uv (recommended).
- `python ocr_processor.py input.pdf ./output` runs a single-file OCR pass (synchronous).
- `python ocr_processor.py ./pdfs ./output --batch` runs batch OCR for a directory (cost-effective, async).
- `python ocr_processor.py ./pdfs ./output --batch --job-id batch_123abc` resumes a batch job.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation.
- Prefer descriptive function names like `process_directory_batch` and `validate_pdf_file`.
- Keep logging consistent via the module-level `logger`.
- No formatter or linter is configured in this repo; keep changes small and readable.

## Testing Guidelines
- No automated tests are present. If you add tests, place them under a `tests/` directory and name files `test_*.py`.
- Focus on validating file I/O and OCR response handling. Example: `tests/test_validate_pdf.py`.

## Commit & Pull Request Guidelines
- Git history is minimal (only “initial commit”), so no established convention yet.
- Use short, imperative commit subjects (e.g., “Add batch retry logging”).
- PRs should describe the OCR behavior change, include sample commands used, and note any API cost impacts.

## Configuration & Runtime Notes
- Set `MISTRAL_API_KEY` in your environment or a `.env` file in the repo root before running.
- Batch mode creates a temporary JSONL file and may run for hours; be explicit about `--check-interval` and `--max-wait-hours` when testing.
