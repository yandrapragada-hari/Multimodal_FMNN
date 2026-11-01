---
name: doc-pdf-creator
description: >-
  Generates publication-quality PDF documents and Word (.docx) files from Markdown blueprints,
  technical specifications, research proposals, and reports.
---

# Document & PDF Creator Skill

## Capabilities
- Converts any Markdown file or technical blueprint into clean, formatted **PDF** documents using `fpdf2` and `reportlab`.
- Generates Microsoft Word (**DOCX**) files with formatted headings, code blocks, tables, and bullet points using `python-docx`.

## Available Helpers
- `scripts/md_to_pdf.py`: Converts markdown text/file directly to PDF.
- `scripts/md_to_docx.py`: Converts markdown text/file to structured Word document.

## Usage Guide

### 1. Generating a PDF from Markdown
```bash
python ~/.gemini/config/skills/doc-pdf-creator/scripts/md_to_pdf.py --input "path/to/doc.md" --output "path/to/doc.pdf" --title "Document Title"
```

### 2. Generating a Word (.docx) Document from Markdown
```bash
python ~/.gemini/config/skills/doc-pdf-creator/scripts/md_to_docx.py --input "path/to/doc.md" --output "path/to/doc.docx"
```
