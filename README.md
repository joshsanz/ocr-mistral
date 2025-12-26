# OCR Processor

A Python tool for processing PDF files with OCR (Optical Character Recognition) using Mistral AI's API. This tool can process single PDF files or batches of PDFs, generating JSON output with extracted text and annotated PDFs.

## Features

- Single file OCR processing
- Batch processing for multiple PDF files
- PDF validation
- JSON output with OCR results
- Annotated PDF generation with OCR text
- Resume capability for batch jobs

## Requirements

- Python 3.13+
- Mistral AI API key
- UV package manager (recommended)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ocr.git
   cd ocr
   ```

2. Install dependencies using UV:
   ```bash
   uv sync
   ```

3. Set up your Mistral AI API key:
   - Create a `.env` file in the project root:
     ```bash
     cp .env.example .env
     ```
   - Add your API key to the `.env` file:
     ```
     MISTRAL_API_KEY=your_api_key_here
     ```

## Usage

### Single File Processing

Process a single PDF file (synchronous mode):
```bash
python ocr_processor.py input.pdf ./output
```

**Note**: The output directory is optional. If not specified, outputs will be saved in the same directory as the input file.

This will:
- Process `input.pdf` with OCR
- Generate `input_ocr.json` with the OCR results
- Generate `input_with_ocr.pdf` with annotated text
- Save outputs in the `./output` directory

### Batch Processing

Process multiple PDF files in a directory (asynchronous, 50% cost savings):
```bash
python ocr_processor.py ./pdfs ./output --batch
```

**Note**: Batch mode provides significant cost savings (50%) compared to synchronous processing.

This will:
- Process all PDF files in the `./pdfs` directory
- Create a temporary JSONL file for tracking batch progress
- Process files asynchronously for better efficiency
- Generate OCR JSON and annotated PDFs for each file
- Save all outputs in the `./output` directory

### Resuming a Batch Job

If a batch job was interrupted, you can resume it:
```bash
python ocr_processor.py ./pdfs ./output --batch --job-id batch_123abc
```

Replace `batch_123abc` with the actual job ID from your previous batch run.

### Single File with Batch Mode

You can also process a single file using batch mode for cost savings:
```bash
python ocr_processor.py document.pdf ./output --batch
```

### Additional Options

- `--check-interval`: Set the interval (in seconds) for checking batch job status (default: 5)
- `--max-wait-hours`: Set the maximum wait time (in hours) for batch job completion (default: 24)

Example with custom options:
```bash
python ocr_processor.py ./pdfs ./output --batch --check-interval 15 --max-wait-hours 8
```

## Input/Output Behavior

### Input Files

- Supports single PDF files or directories containing multiple PDF files
- PDF files should not be password-protected
- Files should have `.pdf` extension

### Output Files

For each input PDF file, the following outputs are generated:

1. `*_ocr.json`: JSON file containing the OCR results with:
   - Extracted text
   - Page information
   - Confidence scores
   - Bounding box coordinates

2. `*_with_ocr.pdf`: Annotated PDF with:
   - Original document content
   - OCR text overlaid as annotations
   - Searchable text layer

### File Naming Conventions

- Input: `document.pdf`
- OCR JSON: `document_ocr.json`
- Annotated PDF: `document_with_ocr.pdf`

### Output Directory

- If output directory is specified, all files are saved there
- If not specified, files are saved in the same directory as the input
- The tool creates the output directory if it doesn't exist

### Batch Processing Files

- Temporary JSONL file is created for batch job tracking
- This file contains job status and progress information
- The file is automatically cleaned up after job completion

## Configuration

The tool uses the following environment variables:

- `MISTRAL_API_KEY`: Your Mistral AI API key (required)

## Cost Considerations

- **Synchronous mode**: Standard pricing for immediate processing
- **Batch mode**: 50% cost savings, but processing is asynchronous and may take longer
- Batch jobs can be resumed if interrupted, avoiding duplicate costs

## Development

### Running Tests

Currently, there are no automated tests. If you add tests, place them in a `tests/` directory and name files `test_*.py`.

### Coding Style

- Follow PEP 8 guidelines
- Use 4-space indentation
- Prefer descriptive function names
- Keep logging consistent via the module-level logger

### Adding New Features

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Implement your changes

3. Test thoroughly

4. Commit with a descriptive message:
   ```bash
   git commit -m "Add feature description"
   ```

5. Push and create a pull request

## Troubleshooting

### API Key Issues

If you get authentication errors:
- Verify your `MISTRAL_API_KEY` is set correctly
- Check that the key is valid and has not expired
- Ensure there are no typos in the `.env` file

### PDF Processing Errors

If PDF processing fails:
- Verify the input PDF is not corrupted
- Check that the PDF is not password-protected
- Ensure you have read permissions for the input file

### Batch Processing Issues

If batch processing hangs:
- Check your internet connection
- Verify the Mistral API is operational
- Try reducing the batch size
- Use the `--max-wait-hours` option to limit wait time

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please follow the existing code style and submit pull requests with clear descriptions of your changes.

## Support

For issues or questions, please open a GitHub issue.