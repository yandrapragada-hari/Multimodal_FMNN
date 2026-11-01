"""Universal Markdown to Word (.docx) document generator using python-docx."""

import argparse
import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor


def convert_md_to_docx(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc = Document()

    in_code_block = False
    code_lines = []

    for line in lines:
        line_clean = line.rstrip("\n")

        # Code block handling
        if line_clean.startswith("```"):
            if in_code_block:
                # Flush code block
                code_text = "\n".join(code_lines)
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.25)
                run = p.add_run(code_text)
                run.font.name = "Consolas"
                run.font.size = Pt(8.5)
                run.font.color.rgb = RGBColor(60, 60, 60)
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(line_clean)
            continue

        # Header 1
        if line_clean.startswith("# "):
            doc.add_heading(line_clean[2:], level=1)
        # Header 2
        elif line_clean.startswith("## "):
            doc.add_heading(line_clean[3:], level=2)
        # Header 3
        elif line_clean.startswith("### "):
            doc.add_heading(line_clean[4:], level=3)
        # Bullet list item
        elif line_clean.startswith("- ") or line_clean.startswith("* "):
            text = line_clean[2:]
            text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
            doc.add_paragraph(text, style="List Bullet")
        # Regular text
        else:
            if line_clean.strip():
                clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", line_clean)
                clean_text = re.sub(r"`(.*?)`", r"\1", clean_text)
                doc.add_paragraph(clean_text)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc.save(output_path)
    print(f"Successfully generated DOCX: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Markdown to Word DOCX")
    parser.add_argument("--input", required=True, help="Input Markdown file path")
    parser.add_argument("--output", required=True, help="Output DOCX file path")
    args = parser.parse_args()

    convert_md_to_docx(args.input, args.output)
