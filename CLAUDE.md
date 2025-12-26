# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an OCR (Optical Character Recognition) processor that uses the Mistral AI OCR API to extract text from PDFs. It supports two processing modes:
- **Synchronous mode**: Real-time processing with immediate results
- **Batch mode**: Cost-effective batch processing (50% cost savings) using the Mistral batch API

The processor extracts content from PDFs, converts it to structured markdown/JSON, and creates enhanced PDFs with formatted OCR results appended.

## Setup and Dependencies

**Python Version**: 3.13+

**Using uv (recommended)**:
```bash
uv sync
```

**Using pip**:
```bash
pip install -r requirements.txt
```

**Environment Configuration**:
1. Copy `.env.example` to `.env`
2. Add your Mistral API key: `MISTRAL_API_KEY=your_api_key_here`
3. Get an API key from: https://console.mistral.ai/

## Commands

**Run synchronous OCR processing** (real-time, slower):
```bash
python ocr_processor.py ./pdfs ./output
```

**Run batch OCR processing** (async, cost-effective, 50% savings):
```bash
python ocr_processor.py ./pdfs ./output --batch
```

**Resume an existing batch job**:
```bash
python ocr_processor.py ./pdfs ./output --batch --job-id batch_123abc
```

**Common options**:
- `--check-interval` (int): Seconds between batch status checks (default: 5)
- `--max-wait-hours` (int): Maximum hours to wait for batch completion (default: 24)

## Architecture and Key Components

### Main Entry Point: `ocr_processor.py`

The module is organized into several functional layers:

1. **API Integration Layer** (`encode_pdf`, `process_pdf_ocr`, batch operations)
   - Handles communication with Mistral OCR API
   - Manages file uploads and batch job creation
   - Monitors batch job status with polling

2. **OCR Processing Layer** (`process_pdf`, `process_directory`)
   - Synchronous PDF processing for single/multiple files
   - Direct API calls with immediate results
   - Used when real-time response is required

3. **Batch Processing Layer** (`process_directory_batch`, `create_batch_jsonl`, `process_batch_results`)
   - Handles large-scale asynchronous processing
   - Creates JSONL batch format for API submission
   - Monitors long-running jobs and retrieves results
   - Allows resuming interrupted jobs via `job_id`

4. **Output Generation Layer** (`create_ocr_pdf`, `merge_pdfs`)
   - Converts OCR results (markdown + tables + images) into formatted PDFs
   - Uses ReportLab for PDF generation with styled text
   - Extracts base64-encoded images from OCR response
   - Merges original PDF with generated OCR results PDF

5. **Utility Layer** (`save_ocr_results`, `extract_image_from_base64`)
   - Persists OCR results as JSON files
   - Handles base64 image extraction and temporary storage

### Key Data Flow

```
PDF File
  ↓
[encode_pdf] → base64
  ↓
[API Call: mistral-ocr-latest]
  ↓
OCR Response {
  pages: [
    {
      markdown: "extracted text",
      tables: ["html"],
      images: [{image_base64: "..."}],
      header/footer: "text"
    }
  ]
}
  ↓
[save_ocr_results] → JSON file
[create_ocr_pdf] → formatted PDF
  ↓
[merge_pdfs] → Original + OCR Results
  ↓
*_with_ocr.pdf
```

### Batch Processing Workflow

The batch API mode offers 50% cost savings but requires asynchronous handling:

1. **Batch Creation Phase**: Multiple PDFs are collected into a JSONL file
2. **Upload Phase**: Batch file is uploaded to Mistral; returns `batch_file_id`
3. **Job Creation Phase**: Batch job is created; returns `job_id`
4. **Monitoring Phase**: Poll job status every `check_interval` seconds until completion
5. **Results Download**: Results file contains JSONL with all processed outputs
6. **Processing Phase**: Results are matched with source PDFs and final PDFs are generated

The `job_id` can be saved and used with `--job-id` flag to resume interrupted processing.

## Configuration and Error Handling

**Logging**: All operations are logged to console with timestamps and levels (INFO, ERROR, WARNING)

**API Key**: Required via environment variable `MISTRAL_API_KEY`. Missing key will raise `ValueError` at module load time.

**File Paths**:
- Output files are named `{original_name}_with_ocr.pdf`
- OCR JSON results are stored as `{original_name}_ocr.json`
- Default output directory is same as input directory (overridable via arguments)

## External Dependencies

- **mistralai**: Mistral API client (v1.10.0+)
- **PyPDF2**: PDF reading and merging (v3.0.1+)
- **reportlab**: PDF generation with formatted layouts (v4.4.7+)
- **Pillow**: Image processing (v12.0.0+)
- **python-dotenv**: Environment variable loading (v1.0.0+)

