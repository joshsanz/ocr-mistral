#!/usr/bin/env python3
"""
OCR Processor: Send PDFs to Mistral OCR API, save results, and append to PDFs with formatted layout.
"""

import os
import json
import base64
import logging
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from io import BytesIO

from mistralai import Mistral
from PyPDF2 import PdfMerger, PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, Preformatted
)
from reportlab.lib import colors
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Mistral client
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not set")

client = Mistral(api_key=api_key)


def encode_pdf(pdf_path: str) -> str:
    """Encode PDF to base64."""
    with open(pdf_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode('utf-8')


def create_batch_jsonl(pdf_files: List[str], output_file: str) -> None:
    """Create a JSONL batch file for Mistral batch API."""
    logger.info(f"Creating batch file with {len(pdf_files)} PDFs")
    with open(output_file, 'w') as f:
        for idx, pdf_path in enumerate(pdf_files):
            base64_pdf = encode_pdf(pdf_path)
            entry = {
                "custom_id": str(idx),
                "body": {
                    "model": "mistral-ocr-latest",
                    "document": {
                        "type": "document_url",
                        "document_url": f"data:application/pdf;base64,{base64_pdf}"
                    },
                    "table_format": "html",
                    "include_image_base64": True
                }
            }
            f.write(json.dumps(entry) + '\n')
    logger.info(f"Batch file created: {output_file}")


def upload_batch_file(batch_file_path: str) -> str:
    """Upload batch file to Mistral API and return file ID."""
    logger.info(f"Uploading batch file: {batch_file_path}")
    with open(batch_file_path, 'rb') as f:
        batch_data = client.files.upload(
            file={
                "file_name": os.path.basename(batch_file_path),
                "content": f
            },
            purpose="batch"
        )
    logger.info(f"Batch file uploaded with ID: {batch_data.id}")
    return batch_data.id


def monitor_batch_job(job_id: str, check_interval: int = 5, max_wait_hours: int = 24) -> bool:
    """Monitor batch job until completion."""
    max_wait_seconds = max_wait_hours * 3600
    elapsed = 0

    while elapsed < max_wait_seconds:
        job = client.batch.jobs.get(job_id=job_id)
        total = job.total_requests
        succeeded = job.succeeded_requests
        failed = job.failed_requests
        completed = succeeded + failed

        percent = round((completed / total) * 100, 2) if total > 0 else 0
        logger.info(
            f"Batch Job {job_id}: {job.status} | "
            f"Completed: {completed}/{total} ({percent}%) | "
            f"Succeeded: {succeeded}, Failed: {failed}"
        )

        if job.status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            logger.info(f"Batch job {job_id} finished with status: {job.status}")
            return job.status == "SUCCEEDED"

        time.sleep(check_interval)
        elapsed += check_interval

    logger.error(f"Batch job {job_id} did not complete within {max_wait_hours} hours")
    return False


def process_batch_results(
    job_id: str,
    pdf_files: List[str],
    output_dir: Optional[str] = None
) -> Tuple[int, int]:
    """Download and process batch results."""
    logger.info(f"Processing results for batch job {job_id}")

    # Download results file
    results_content = client.files.download(file_id=job_id)

    # Parse results - the API returns JSONL format
    results_lines = results_content.decode('utf-8').strip().split('\n')
    results_by_id = {}
    for line in results_lines:
        if line.strip():
            result = json.loads(line)
            custom_id = result.get('custom_id')
            results_by_id[custom_id] = result

    successful = 0
    failed = 0

    # Process each PDF with its corresponding OCR result
    for idx, pdf_path in enumerate(pdf_files):
        custom_id = str(idx)
        if custom_id not in results_by_id:
            logger.warning(f"No result found for PDF {idx}: {pdf_path}")
            failed += 1
            continue

        try:
            result = results_by_id[custom_id]

            # Extract the OCR response from the batch result
            if result.get('status') != 'succeeded':
                logger.error(f"OCR failed for {pdf_path}: {result.get('error')}")
                failed += 1
                continue

            # Parse the response body - it's a JSON string in batch responses
            response_body = result.get('result', {}).get('body', {})
            if isinstance(response_body, str):
                ocr_response = json.loads(response_body)
            else:
                ocr_response = response_body

            # Save OCR results
            save_ocr_results(pdf_path, ocr_response)

            # Create PDF from OCR results
            ocr_pdf_buffer = create_ocr_pdf(ocr_response)

            # Determine output path
            if output_dir is None:
                output_dir_path = os.path.dirname(pdf_path)
            else:
                output_dir_path = output_dir

            pdf_filename = os.path.basename(pdf_path)
            output_pdf_path = os.path.join(output_dir_path, pdf_filename.replace('.pdf', '_with_ocr.pdf'))

            # Merge PDFs
            merge_pdfs(pdf_path, ocr_pdf_buffer, output_pdf_path)

            logger.info(f"Successfully processed: {pdf_path}")
            successful += 1

        except Exception as e:
            logger.error(f"Error processing result for {pdf_path}: {e}")
            failed += 1

    return successful, failed


def process_pdf_ocr(pdf_path: str) -> Dict[str, Any]:
    """Call Mistral OCR API and return response as dict."""
    logger.info(f"Calling OCR API for: {pdf_path}")
    base64_pdf = encode_pdf(pdf_path)

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{base64_pdf}"
        },
        table_format="html",
        include_image_base64=True
    )

    # Convert response object to dict
    return ocr_response.model_dump() if hasattr(ocr_response, 'model_dump') else ocr_response


def save_ocr_results(pdf_path: str, ocr_response: Dict[str, Any]) -> str:
    """Save OCR response as JSON next to the PDF."""
    json_path = pdf_path.replace('.pdf', '_ocr.json')

    with open(json_path, 'w') as f:
        json.dump(ocr_response, f, indent=2)

    logger.info(f"Saved OCR results to: {json_path}")
    return json_path


def extract_image_from_base64(base64_str: str, temp_dir: str = "/tmp") -> Optional[str]:
    """Extract base64 image and save to temporary file."""
    try:
        # Remove data URI prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]

        image_data = base64.b64decode(base64_str)
        temp_path = f"{temp_dir}/ocr_image_{hash(base64_str) % 10000}.png"

        with open(temp_path, 'wb') as f:
            f.write(image_data)

        return temp_path
    except Exception as e:
        logger.warning(f"Failed to extract image from base64: {e}")
        return None


def create_ocr_pdf(ocr_response: Dict[str, Any]) -> BytesIO:
    """Create a PDF from OCR results with formatted layout."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=HexColor('#1f4788'),
        spaceAfter=12,
        borderPadding=5
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=HexColor('#2e5c8a'),
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        spaceAfter=6,
        alignment=0
    )

    # Add OCR content for each page
    pages = ocr_response.get('pages', [])
    for page_idx, page in enumerate(pages):
        if page_idx > 0:
            story.append(PageBreak())

        # Add page header
        page_num = page.get('index', page_idx) + 1
        story.append(Paragraph(f"OCR Results - Page {page_num}", title_style))
        story.append(Spacer(1, 0.1*inch))

        # Add header if present
        if page.get('header'):
            story.append(Paragraph(f"<b>Header:</b> {page['header']}", body_style))
            story.append(Spacer(1, 0.05*inch))

        # Add main markdown content
        markdown_content = page.get('markdown', '')
        if markdown_content:
            # Simple markdown parsing for common patterns
            lines = markdown_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 0.05*inch))
                elif line.startswith('# '):
                    story.append(Paragraph(line[2:], heading_style))
                elif line.startswith('## '):
                    story.append(Paragraph(line[3:], styles['Heading3']))
                elif line.startswith('- ') or line.startswith('* '):
                    story.append(Paragraph(f"• {line[2:]}", body_style))
                else:
                    story.append(Paragraph(line, body_style))

        # Add tables if present
        tables = page.get('tables', [])
        for table_data in tables:
            if isinstance(table_data, str):
                # HTML table - add as formatted text for now
                story.append(Spacer(1, 0.1*inch))
                story.append(Paragraph("<b>Table:</b>", heading_style))
                story.append(Preformatted(table_data, styles['Code']))
            story.append(Spacer(1, 0.1*inch))

        # Add images if present
        images = page.get('images', [])
        for img_data in images:
            if isinstance(img_data, dict) and 'image_base64' in img_data:
                img_path = extract_image_from_base64(img_data['image_base64'])
                if img_path:
                    try:
                        # Get image dimensions
                        pil_img = Image.open(img_path)
                        img_width, img_height = pil_img.size

                        # Scale image to fit page width
                        max_width = 6 * inch
                        scale = min(1.0, max_width / img_width) if img_width > 0 else 1.0

                        story.append(Spacer(1, 0.1*inch))
                        rl_image = RLImage(
                            img_path,
                            width=img_width * scale,
                            height=img_height * scale
                        )
                        story.append(rl_image)
                        story.append(Spacer(1, 0.1*inch))
                    except Exception as e:
                        logger.warning(f"Failed to add image: {e}")

        # Add footer if present
        if page.get('footer'):
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(f"<b>Footer:</b> {page['footer']}", body_style))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


def merge_pdfs(original_pdf_path: str, ocr_pdf_buffer: BytesIO, output_pdf_path: str) -> None:
    """Merge original PDF with OCR results PDF."""
    merger = PdfMerger()

    try:
        # Add original PDF
        merger.append(original_pdf_path)

        # Add OCR PDF
        merger.append(ocr_pdf_buffer)

        # Write result
        with open(output_pdf_path, 'wb') as output_file:
            merger.write(output_file)

        logger.info(f"Created merged PDF: {output_pdf_path}")
    finally:
        merger.close()


def process_pdf(pdf_path: str, output_dir: Optional[str] = None) -> bool:
    """Process a single PDF: call OCR, save results, and merge."""
    try:
        # Get OCR results
        ocr_response = process_pdf_ocr(pdf_path)

        # Save OCR results
        save_ocr_results(pdf_path, ocr_response)

        # Create PDF from OCR results
        ocr_pdf_buffer = create_ocr_pdf(ocr_response)

        # Determine output path
        if output_dir is None:
            output_dir = os.path.dirname(pdf_path)

        pdf_filename = os.path.basename(pdf_path)
        output_pdf_path = os.path.join(output_dir, pdf_filename.replace('.pdf', '_with_ocr.pdf'))

        # Merge PDFs
        merge_pdfs(pdf_path, ocr_pdf_buffer, output_pdf_path)

        logger.info(f"Successfully processed: {pdf_path}")
        return True

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return False


def process_directory(directory_path: str, output_dir: Optional[str] = None, recursive: bool = True) -> None:
    """Process all PDFs in a directory using synchronous API calls."""
    dir_path = Path(directory_path)

    if not dir_path.exists():
        logger.error(f"Directory not found: {directory_path}")
        return

    # Find all PDF files
    if recursive:
        pdf_files = list(dir_path.rglob("*.pdf"))
    else:
        pdf_files = list(dir_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {directory_path}")
        return

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process (synchronous mode)")

    successful = 0
    failed = 0

    for pdf_path in pdf_files:
        if process_pdf(str(pdf_path), output_dir):
            successful += 1
        else:
            failed += 1

    logger.info(f"Processing complete: {successful} successful, {failed} failed")


def process_directory_batch(
    directory_path: str,
    output_dir: Optional[str] = None,
    recursive: bool = True,
    check_interval: int = 5,
    max_wait_hours: int = 24,
    job_id: Optional[str] = None
) -> None:
    """
    Process all PDFs in a directory using Mistral batch API (50% cost savings).

    Args:
        directory_path: Path to directory containing PDFs
        output_dir: Optional output directory for _with_ocr.pdf files
        recursive: Search subdirectories
        check_interval: Seconds between batch status checks
        max_wait_hours: Maximum hours to wait for batch completion
        job_id: If provided, check status of existing batch job instead of creating new one
    """
    dir_path = Path(directory_path)

    if not dir_path.exists():
        logger.error(f"Directory not found: {directory_path}")
        return

    # Find all PDF files
    if recursive:
        pdf_files = list(dir_path.rglob("*.pdf"))
    else:
        pdf_files = list(dir_path.glob("*.pdf"))

    pdf_files = [str(f) for f in pdf_files]

    if not pdf_files:
        logger.warning(f"No PDF files found in {directory_path}")
        return

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process (batch mode)")

    # If job_id is provided, skip batch creation and go straight to monitoring
    if job_id:
        logger.info(f"Resuming batch job {job_id}")
        batch_file_id = job_id
        is_complete = monitor_batch_job(job_id, check_interval, max_wait_hours)
    else:
        # Create batch file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            batch_file = f.name

        try:
            create_batch_jsonl(pdf_files, batch_file)

            # Upload batch file
            batch_file_id = upload_batch_file(batch_file)

            # Create batch job
            logger.info(f"Creating batch job for {len(pdf_files)} PDFs")
            batch_job = client.batch.jobs.create(
                input_files=[batch_file_id],
                model="mistral-ocr-latest",
                endpoint="/v1/ocr"
            )
            logger.info(f"Batch job created with ID: {batch_job.id}")

            # Monitor batch job
            is_complete = monitor_batch_job(batch_job.id, check_interval, max_wait_hours)
            job_id = batch_job.id
        finally:
            # Clean up temporary batch file
            if os.path.exists(batch_file):
                os.remove(batch_file)

    if not is_complete:
        logger.error("Batch processing did not complete successfully")
        logger.info(f"To resume later, use: python ocr_processor.py {directory_path} {output_dir or '.'} --batch --job-id {job_id}")
        return

    # Process results
    successful, failed = process_batch_results(job_id, pdf_files, output_dir)
    logger.info(f"Batch processing complete: {successful} successful, {failed} failed")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Process PDFs with Mistral OCR API and append results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synchronous mode (real-time, slower but immediate results)
  python ocr_processor.py ./pdfs ./output

  # Batch mode (cost-effective, 50% savings, asynchronous)
  python ocr_processor.py ./pdfs ./output --batch

  # Resume a batch job
  python ocr_processor.py ./pdfs ./output --batch --job-id batch_123abc
        """
    )

    parser.add_argument("directory", help="Directory containing PDFs to process")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output directory for *_with_ocr.pdf files (default: same as input)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch API mode (50%% cost savings, asynchronous)"
    )
    parser.add_argument(
        "--job-id",
        type=str,
        help="Resume existing batch job (requires --batch)"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=5,
        help="Seconds between batch status checks (default: 5)"
    )
    parser.add_argument(
        "--max-wait-hours",
        type=int,
        default=24,
        help="Maximum hours to wait for batch completion (default: 24)"
    )

    args = parser.parse_args()

    if args.job_id and not args.batch:
        logger.error("--job-id requires --batch flag")
        sys.exit(1)

    if args.batch:
        process_directory_batch(
            args.directory,
            args.output_dir,
            recursive=True,
            check_interval=args.check_interval,
            max_wait_hours=args.max_wait_hours,
            job_id=args.job_id
        )
    else:
        process_directory(args.directory, args.output_dir, recursive=True)
