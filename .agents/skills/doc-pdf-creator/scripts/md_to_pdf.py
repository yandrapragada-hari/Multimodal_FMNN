"""Universal Markdown to PDF converter using FPDF2 with Unicode sanitization."""

import argparse
import os
import re
from fpdf import FPDF
from fpdf.enums import XPos, YPos


def sanitize_text(text: str) -> str:
    """Replaces Unicode characters with ASCII equivalents to support core fonts."""
    replacements = {
        "\u2500": "-",
        "\u2502": "|",
        "\u251c": "|-",
        "\u2514": "`-",
        "\u252c": "+",
        "\u2534": "+",
        "\u253c": "+",
        "\u2192": "->",
        "\u2190": "<-",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2260": "!=",
        "\u2208": "in",
        "\u2211": "sum",
        "\u221e": "inf",
        "\u22c5": "*",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2014": "--",
        "\u2013": "-",
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    # Remove any remaining non-latin1 characters safely
    return text.encode("latin-1", "replace").decode("latin-1")


class MarkdownPDF(FPDF):
    def __init__(self, title="Document"):
        super().__init__()
        self.doc_title = title
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, self.doc_title, align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def convert_md_to_pdf(input_path: str, output_path: str, title: str = "Document"):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pdf = MarkdownPDF(title=sanitize_text(title))
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)

    in_code_block = False

    for line in lines:
        line_clean = sanitize_text(line.rstrip("\n"))

        # Code block boundary
        if line_clean.startswith("```"):
            in_code_block = not in_code_block
            pdf.ln(2)
            continue

        if in_code_block:
            pdf.set_font("Courier", size=8)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 5, "  " + line_clean[:85], fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            continue

        # Header 1
        if line_clean.startswith("# "):
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 15)
            pdf.set_text_color(20, 40, 120)
            pdf.cell(0, 9, line_clean[2:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
        # Header 2
        elif line_clean.startswith("## "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(40, 70, 150)
            pdf.cell(0, 7, line_clean[3:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
        # Header 3
        elif line_clean.startswith("### "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.cell(0, 6, line_clean[4:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
        # Bullet point
        elif line_clean.startswith("- ") or line_clean.startswith("* "):
            pdf.set_font("Helvetica", size=9.5)
            bullet_text = line_clean[2:]
            bullet_text = re.sub(r"\*\*(.*?)\*\*", r"\1", bullet_text)
            pdf.cell(5, 5, "-")
            pdf.multi_cell(0, 5, bullet_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
        # Normal text
        else:
            if line_clean.strip() == "":
                pdf.ln(2)
            else:
                pdf.set_font("Helvetica", size=9.5)
                clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", line_clean)
                clean_text = re.sub(r"`(.*?)`", r"\1", clean_text)
                pdf.multi_cell(0, 5, clean_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pdf.output(output_path)
    print(f"Successfully generated PDF: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Markdown to PDF")
    parser.add_argument("--input", required=True, help="Input Markdown file path")
    parser.add_argument("--output", required=True, help="Output PDF file path")
    parser.add_argument("--title", default="Document", help="Document title for header")
    args = parser.parse_args()

    convert_md_to_pdf(args.input, args.output, args.title)
