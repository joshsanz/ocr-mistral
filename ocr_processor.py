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
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from io import BytesIO

from mistralai import Mistral
from dotenv import load_dotenv
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

# Configure logging with colors
import coloredlogs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add colored logging
coloredlogs.install(
    level='INFO',
    logger=logging.getLogger(),
    fmt='%(asctime)s - %(levelname)s - %(message)s',
    level_styles={
        'debug': {'color': 'cyan'},
        'info': {'color': 'green'},
        'warning': {'color': 'yellow'},
        'error': {'color': 'red', 'bold': True},
        'critical': {'color': 'red', 'background': 'white'}
    },
    field_styles={
        'asctime': {'color': 'blue'},
        'levelname': {'bold': True}
    }
)

logger = logging.getLogger(__name__)


def retry_api_call(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 4.0,
    retryable_exceptions: tuple = (Exception,),
    **kwargs
) -> Any:
    """
    Retry a function call with exponential backoff.

    Args:
        func: Function to call
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay between retries
        retryable_exceptions: Tuple of exception types that should trigger retries
        **kwargs: Arguments to pass to the function

    Returns:
        Result of the function call

    Raises:
        Exception: The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(**kwargs)
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay * 0.1)  # Add 10% jitter
                total_delay = delay + jitter

                logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}. "
                    f"Retrying in {total_delay:.2f} seconds..."
                )
                time.sleep(total_delay)
            continue

    logger.error(f"All {max_retries} retry attempts failed")
    raise last_exception


class FailureTracker:
    """Track failed files and retry attempts for comprehensive reporting."""

    def __init__(self):
        self.failed_files = []
        self.retry_attempts = {}

    def report_failure(self, file_path: str, error: Exception, attempt: int = 0):
        """Report a failure for a specific file."""
        failure_info = {
            'file_path': file_path,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'attempt': attempt,
            'timestamp': time.time()
        }

        # Track retry attempts
        if file_path not in self.retry_attempts:
            self.retry_attempts[file_path] = []

        self.retry_attempts[file_path].append(failure_info)

        # Only add to final failed list if this is the last attempt
        if attempt >= 3:  # Assuming max 3 retries
            self.failed_files.append(failure_info)

    def generate_report(self) -> Dict:
        """Generate a comprehensive failure report."""
        return {
            'total_failed_files': len(self.failed_files),
            'failed_files': self.failed_files,
            'retry_statistics': {
                file_path: len(attempts)
                for file_path, attempts in self.retry_attempts.items()
            }
        }

    def save_report(self, report_path: str):
        """Save failure report to JSON file."""
        report = self.generate_report()
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Failure report saved to: {report_path}")

    def log_summary(self):
        """Log a summary of failures."""
        if not self.failed_files:
            logger.info("All files processed successfully!")
            return

        logger.error(f"Processing completed with {len(self.failed_files)} failed files:")
        for failure in self.failed_files:
            logger.error(f"  - {failure['file_path']}: {failure['error_type']} - {failure['error_message']}")

# Load environment variables from .env file if present
# Uses explicit path to work regardless of current working directory
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Initialize Mistral client
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    raise ValueError(
        "Mistral API key not found. Please set MISTRAL_API_KEY environment variable "
        "or add 'MISTRAL_API_KEY=your_key' to .env file in the project root."
    )

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
                    "document": {
                        "document_url": f"data:application/pdf;base64,{base64_pdf}"
                    }
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
    output_dir: Optional[str] = None,
    failure_tracker: Optional[FailureTracker] = None
) -> Tuple[int, int]:
    """Download and process batch results."""
    logger.info(f"Processing results for batch job {job_id}")

    # Initialize failure tracker if not provided
    if failure_tracker is None:
        failure_tracker = FailureTracker()

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
        
        # Validate PDF before processing
        if not validate_pdf_file(pdf_path):
            error_msg = f"PDF validation failed for {pdf_path}. Skipping to avoid wasting API credits."
            logger.error(error_msg)
            failure_tracker.report_failure(pdf_path, ValueError(error_msg), attempt=0)
            failed += 1
            continue
            
        if custom_id not in results_by_id:
            logger.warning(f"No result found for PDF {idx}: {pdf_path}")
            failure_tracker.report_failure(pdf_path, Exception("No result found in batch response"), attempt=0)
            failed += 1
            continue

        try:
            result = results_by_id[custom_id]

            # Extract the OCR response from the batch result
            if result.get('status') != 'succeeded':
                error_msg = result.get('error', 'Unknown batch processing error')
                logger.error(f"OCR failed for {pdf_path}: {error_msg}")
                failure_tracker.report_failure(pdf_path, Exception(error_msg), attempt=0)
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
            failure_tracker.report_failure(pdf_path, e, attempt=0)
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
    response_dict = ocr_response.model_dump() if hasattr(ocr_response, 'model_dump') else ocr_response
    
    # Validate the OCR response
    validate_ocr_response(response_dict, pdf_path)
    
    return response_dict


def process_pdf_ocr_with_retry(pdf_path: str, failure_tracker: FailureTracker) -> Optional[Dict[str, Any]]:
    """Call Mistral OCR API with retry logic and failure tracking."""
    try:
        return retry_api_call(
            process_pdf_ocr,
            max_retries=3,
            base_delay=1.0,
            max_delay=4.0,
            pdf_path=pdf_path
        )
    except Exception as e:
        failure_tracker.report_failure(pdf_path, e, attempt=3)
        return None


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


def process_pdf(pdf_path: str, output_dir: Optional[str] = None, failure_tracker: Optional[FailureTracker] = None) -> bool:
    """Process a single PDF: call OCR, save results, and merge."""
    try:
        # Initialize failure tracker if not provided
        if failure_tracker is None:
            failure_tracker = FailureTracker()

        # Validate PDF before processing
        if not validate_pdf_file(pdf_path):
            error_msg = f"PDF validation failed for {pdf_path}. Skipping to avoid wasting API credits."
            logger.error(error_msg)
            failure_tracker.report_failure(pdf_path, ValueError(error_msg), attempt=0)
            return False

        # Get OCR results with retry logic
        ocr_response = process_pdf_ocr_with_retry(pdf_path, failure_tracker)
        
        if ocr_response is None:
            # OCR failed even after retries
            return False

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
        if failure_tracker:
            failure_tracker.report_failure(pdf_path, e, attempt=0)
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

    # Filter out invalid PDFs before processing
    valid_pdf_files = []
    invalid_pdf_files = []
    
    for pdf_path in pdf_files:
        if validate_pdf_file(str(pdf_path)):
            valid_pdf_files.append(pdf_path)
        else:
            invalid_pdf_files.append(pdf_path)
            logger.warning(f"Skipping invalid PDF: {pdf_path}")
    
    logger.info(f"Found {len(pdf_files)} PDF file(s), {len(valid_pdf_files)} valid, {len(invalid_pdf_files)} invalid")

    # Initialize failure tracker
    failure_tracker = FailureTracker()
    
    successful = 0
    failed = 0

    for pdf_path in valid_pdf_files:
        if process_pdf(str(pdf_path), output_dir, failure_tracker):
            successful += 1
        else:
            failed += 1

    # Generate and log failure report
    failure_tracker.log_summary()
    
    # Save failure report
    report_path = os.path.join(directory_path, "failure_report.json")
    failure_tracker.save_report(report_path)
    
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

    # Filter out invalid PDFs before processing
    valid_pdf_files = []
    invalid_pdf_files = []
    
    for pdf_path in pdf_files:
        if validate_pdf_file(pdf_path):
            valid_pdf_files.append(pdf_path)
        else:
            invalid_pdf_files.append(pdf_path)
            logger.warning(f"Skipping invalid PDF: {pdf_path}")
    
    logger.info(f"Found {len(pdf_files)} PDF file(s), {len(valid_pdf_files)} valid, {len(invalid_pdf_files)} invalid")

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
            create_batch_jsonl(valid_pdf_files, batch_file)

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

    # Initialize failure tracker
    failure_tracker = FailureTracker()
    
    # Process results
    successful, failed = process_batch_results(job_id, valid_pdf_files, output_dir, failure_tracker)
    
    # Generate and log failure report
    failure_tracker.log_summary()
    
    # Save failure report
    report_path = os.path.join(directory_path, "failure_report.json")
    failure_tracker.save_report(report_path)
    
    logger.info(f"Batch processing complete: {successful} successful, {failed} failed")


def is_valid_pdf_file(path: str) -> bool:
    """Check if path is a valid PDF file."""
    try:
        return Path(path).is_file() and path.lower().endswith('.pdf')
    except Exception:
        return False


def validate_pdf_file(pdf_path: str) -> bool:
    """
    Validate that a PDF file can be opened and read.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        True if PDF is valid and can be opened, False otherwise
        
    Raises:
        Exception: If there are issues reading the file
    """
    try:
        # Check if file exists and has PDF extension
        if not is_valid_pdf_file(pdf_path):
            logger.warning(f"File {pdf_path} is not a valid PDF file")
            return False
            
        # Try to open the PDF with PyPDF2 to validate it
        with open(pdf_path, 'rb') as pdf_file:
            try:
                # Try to read the PDF - this will fail if PDF is corrupted
                reader = PdfReader(pdf_file)
                
                # Check if we can get basic info (this validates the PDF structure)
                if len(reader.pages) == 0:
                    logger.warning(f"PDF {pdf_path} appears to be empty or have no pages")
                    return False
                    
                return True
                
            except Exception as e:
                logger.warning(f"Failed to read PDF {pdf_path}: {str(e)}")
                return False
                
    except Exception as e:
        logger.warning(f"Error validating PDF {pdf_path}: {str(e)}")
        return False


def validate_ocr_response(ocr_response: Dict[str, Any], pdf_path: str) -> None:
    """
    Validate OCR response to ensure it contains meaningful data.
    
    Args:
        ocr_response: Response from OCR API
        pdf_path: Path to the original PDF file
        
    Raises:
        ValueError: If OCR response is invalid or empty
    """
    if not ocr_response:
        raise ValueError(f"OCR API returned empty response for {pdf_path}")
        
    # Check for empty pages array
    pages = ocr_response.get('pages', [])
    if not pages or len(pages) == 0:
        usage_info = ocr_response.get('usage_info', {})
        doc_size = usage_info.get('doc_size_bytes', 'unknown')
        pages_processed = usage_info.get('pages_processed', 0)
        
        error_msg = (
            f"OCR API returned empty results for {pdf_path} "
            f"(pages: [], document_size: {doc_size} bytes, "
            f"pages_processed: {pages_processed}). "
            f"This typically indicates a corrupted or unreadable PDF."
        )
        raise ValueError(error_msg)
        
    # Check if document_annotation is None (another sign of failure)
    if ocr_response.get('document_annotation') is None:
        error_msg = (
            f"OCR API returned null document_annotation for {pdf_path}. "
            f"This typically indicates the PDF could not be processed."
        )
        raise ValueError(error_msg)
        
    # Check for zero pages processed in usage info
    usage_info = ocr_response.get('usage_info', {})
    if usage_info.get('pages_processed', 0) == 0:
        error_msg = (
            f"OCR API processed 0 pages for {pdf_path} "
            f"(document_size: {usage_info.get('doc_size_bytes', 'unknown')} bytes). "
            f"This typically indicates a corrupted or unreadable PDF."
        )
        raise ValueError(error_msg)


def is_valid_directory(path: str) -> bool:
    """Check if path is a valid directory."""
    try:
        return Path(path).is_dir()
    except Exception:
        return False


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Process PDFs with Mistral OCR API and append results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single PDF file (synchronous)
  python ocr_processor.py document.pdf ./output

  # Process directory of PDFs (synchronous)
  python ocr_processor.py ./pdfs ./output

  # Process directory with batch mode (cost-effective, 50% savings)
  python ocr_processor.py ./pdfs ./output --batch

  # Process single file with batch mode
  python ocr_processor.py document.pdf ./output --batch

  # Resume a batch job
  python ocr_processor.py ./pdfs ./output --batch --job-id batch_123abc
        """
    )

    parser.add_argument(
        "input_path",
        help="File or directory to process (PDF file or directory containing PDFs)"
    )
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

    input_path = args.input_path

    if is_valid_pdf_file(input_path):
        # Process single file
        logger.info(f"Processing single PDF file: {input_path}")
        
        # Initialize failure tracker
        failure_tracker = FailureTracker()
        
        if args.batch:
            # For batch mode with single file, we need to create a batch with just this file
            pdf_files = [input_path]
            
            # Validate PDF before processing
            if not validate_pdf_file(input_path):
                error_msg = f"PDF validation failed for {input_path}. Skipping to avoid wasting API credits."
                logger.error(error_msg)
                failure_tracker.report_failure(input_path, ValueError(error_msg), attempt=0)
                failure_tracker.log_summary()
                report_path = os.path.join(os.path.dirname(input_path) or '.', "failure_report.json")
                failure_tracker.save_report(report_path)
                sys.exit(1)
            
            # Create batch file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                batch_file = f.name

            try:
                create_batch_jsonl(pdf_files, batch_file)

                # Upload batch file
                batch_file_id = upload_batch_file(batch_file)

                # Create batch job
                logger.info(f"Creating batch job for single PDF")
                batch_job = client.batch.jobs.create(
                    input_files=[batch_file_id],
                    model="mistral-ocr-latest",
                    endpoint="/v1/ocr"
                )
                logger.info(f"Batch job created with ID: {batch_job.id}")

                # Monitor batch job
                is_complete = monitor_batch_job(batch_job.id, args.check_interval, args.max_wait_hours)
                
                if is_complete:
                    # Process results
                    successful, failed = process_batch_results(
                        batch_job.id, 
                        pdf_files, 
                        args.output_dir, 
                        failure_tracker
                    )
                    
                    # Generate and log failure report
                    failure_tracker.log_summary()
                    
                    # Save failure report
                    report_path = os.path.join(os.path.dirname(input_path) or '.', "failure_report.json")
                    failure_tracker.save_report(report_path)
                    
                    logger.info(f"Single file batch processing complete: {successful} successful, {failed} failed")
                else:
                    logger.error("Single file batch processing did not complete successfully")
            finally:
                # Clean up temporary batch file
                if os.path.exists(batch_file):
                    os.remove(batch_file)
        else:
            # Synchronous processing for single file
            if process_pdf(input_path, args.output_dir, failure_tracker):
                logger.info(f"Successfully processed single file: {input_path}")
                failure_tracker.log_summary()
            else:
                logger.error(f"Failed to process single file: {input_path}")
                failure_tracker.log_summary()
                
                # Save failure report
                report_path = os.path.join(os.path.dirname(input_path) or '.', "failure_report.json")
                failure_tracker.save_report(report_path)

    elif is_valid_directory(input_path):
        # Process directory
        logger.info(f"Processing directory: {input_path}")
        if args.batch:
            process_directory_batch(
                input_path,
                args.output_dir,
                recursive=True,
                check_interval=args.check_interval,
                max_wait_hours=args.max_wait_hours,
                job_id=args.job_id
            )
        else:
            process_directory(input_path, args.output_dir, recursive=True)
    else:
        logger.error(f"Invalid input: {input_path} is neither a PDF file nor a directory")
        sys.exit(1)
