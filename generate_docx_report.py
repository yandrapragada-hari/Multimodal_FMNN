import re
import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tcPr.append(shd)

def set_table_borders(table, color="D1D5DB", sz="4"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def parse_inline_runs(paragraph, text, default_size=Pt(10), default_color=RGBColor(51, 65, 85)):
    # Simple tokenizer for bold-italic, bold, italic, inline code
    pattern = re.compile(
        r'(\*\*\*.*?\*\*\*'
        r'|\*\*.*?\*\*'
        r'|\*.*?\*'
        r'|`.*?`'
        r'|\[.*?\]\(.*?\)'
        r'|[^`\*\[]+)'
    )
    tokens = pattern.findall(text)
    if not tokens:
        tokens = [text]

    for tok in tokens:
        if tok.startswith('***') and tok.endswith('***') and len(tok) >= 6:
            content = tok[3:-3]
            run = paragraph.add_run(content)
            run.font.name = "Calibri"
            run.font.size = default_size
            run.font.bold = True
            run.font.italic = True
            run.font.color.rgb = default_color
        elif tok.startswith('**') and tok.endswith('**') and len(tok) >= 4:
            content = tok[2:-2]
            run = paragraph.add_run(content)
            run.font.name = "Calibri"
            run.font.size = default_size
            run.font.bold = True
            run.font.color.rgb = RGBColor(15, 23, 42)
        elif tok.startswith('*') and tok.endswith('*') and len(tok) >= 2:
            content = tok[1:-1]
            run = paragraph.add_run(content)
            run.font.name = "Calibri"
            run.font.size = default_size
            run.font.italic = True
            run.font.color.rgb = default_color
        elif tok.startswith('`') and tok.endswith('`') and len(tok) >= 2:
            content = tok[1:-1]
            run = paragraph.add_run(content)
            run.font.name = "Consolas"
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(190, 24, 93)  # Rose/Pinkish for code
        elif tok.startswith('[') and '](' in tok and tok.endswith(')'):
            m = re.match(r'\[(.*?)\]\((.*?)\)', tok)
            if m:
                label = m.group(1).replace('`', '')
                run = paragraph.add_run(label)
                run.font.name = "Calibri"
                run.font.size = default_size
                run.font.bold = True
                run.font.color.rgb = RGBColor(2, 132, 199)  # Blue link
            else:
                run = paragraph.add_run(tok)
                run.font.name = "Calibri"
                run.font.size = default_size
                run.font.color.rgb = default_color
        else:
            # Clean up latex math tokens like $X$, $T_k$, \mu
            clean_tok = tok.replace('$', '')
            run = paragraph.add_run(clean_tok)
            run.font.name = "Calibri"
            run.font.size = default_size
            run.font.color.rgb = default_color

def convert_report_to_docx(md_path, docx_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    doc = Document()

    # Page setup - 0.8 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    i = 0
    n = len(lines)

    in_code_block = False
    code_lines = []

    in_table = False
    table_lines = []

    def flush_table():
        nonlocal in_table, table_lines
        if not table_lines:
            in_table = False
            return
        
        # Parse table rows
        rows_data = []
        for tl in table_lines:
            if re.match(r'^\s*\|?\s*:?-+:?\s*\|', tl):
                continue  # separator row
            cells = [c.strip() for c in tl.strip().strip('|').split('|')]
            if cells:
                rows_data.append(cells)
        
        table_lines = []
        in_table = False
        
        if not rows_data:
            return
        
        num_rows = len(rows_data)
        num_cols = max(len(r) for r in rows_data)
        
        tbl = doc.add_table(rows=num_rows, cols=num_cols)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        set_table_borders(tbl, color="CBD5E1")
        
        for r_idx, row in enumerate(rows_data):
            tr = tbl.rows[r_idx]
            is_header = (r_idx == 0)
            
            for c_idx in range(num_cols):
                cell = tr.cells[c_idx]
                val = row[c_idx] if c_idx < len(row) else ""
                set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
                
                if is_header:
                    set_cell_shading(cell, "1E293B")  # Dark slate
                    p = cell.paragraphs[0]
                    p.paragraph_format.space_before = Pt(3)
                    p.paragraph_format.space_after = Pt(3)
                    run = p.add_run(val.replace('**', '').replace('$', ''))
                    run.font.name = "Calibri"
                    run.font.bold = True
                    run.font.size = Pt(9.5)
                    run.font.color.rgb = RGBColor(255, 255, 255)
                else:
                    if r_idx % 2 == 1:
                        set_cell_shading(cell, "F8FAFC")  # Light zebra
                    p = cell.paragraphs[0]
                    p.paragraph_format.space_before = Pt(2)
                    p.paragraph_format.space_after = Pt(2)
                    parse_inline_runs(p, val, default_size=Pt(9), default_color=RGBColor(30, 41, 59))
        
        # Add spacing after table
        sp = doc.add_paragraph()
        sp.paragraph_format.space_before = Pt(0)
        sp.paragraph_format.space_after = Pt(4)

    def flush_code():
        nonlocal in_code_block, code_lines
        if not code_lines:
            in_code_block = False
            return
        
        code_text = "\n".join(code_lines)
        code_lines = []
        in_code_block = False

        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblPr = tbl._tbl.tblPr
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
            f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
            f'<w:left w:val="single" w:sz="18" w:space="0" w:color="0284C7"/>'
            f'<w:right w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
            f'<w:insideH w:val="none"/>'
            f'<w:insideV w:val="none"/>'
            f'</w:tblBorders>'
        )
        tblPr.append(borders)
        
        cell = tbl.cell(0, 0)
        set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
        set_cell_shading(cell, "F1F5F9")
        
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.05
        run = p.add_run(code_text)
        run.font.name = "Consolas"
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(30, 41, 59)

        sp = doc.add_paragraph()
        sp.paragraph_format.space_before = Pt(0)
        sp.paragraph_format.space_after = Pt(4)

    while i < n:
        raw_line = lines[i]
        line = raw_line.rstrip("\r\n")

        # Code block delimiter
        if line.startswith("```"):
            if in_table:
                flush_table()
            if in_code_block:
                flush_code()
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Markdown Table line
        if line.strip().startswith("|") and line.strip().endswith("|"):
            in_table = True
            table_lines.append(line.strip())
            i += 1
            continue
        elif in_table:
            flush_table()

        # Headings
        if line.startswith("# "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(line[2:].strip())
            run.font.name = "Calibri"
            run.font.size = Pt(22)
            run.font.bold = True
            run.font.color.rgb = RGBColor(15, 23, 42)
        elif line.startswith("## "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(3)
            run = p.add_run(line[3:].strip())
            run.font.name = "Calibri"
            run.font.size = Pt(15)
            run.font.bold = True
            run.font.color.rgb = RGBColor(30, 58, 138)  # Deep Blue
        elif line.startswith("### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(9)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(line[4:].strip())
            run.font.name = "Calibri"
            run.font.size = Pt(12.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(51, 65, 85)
        elif line.startswith("#### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(line[5:].strip())
            run.font.name = "Calibri"
            run.font.size = Pt(11)
            run.font.bold = True
            run.font.color.rgb = RGBColor(71, 85, 105)
        elif line.startswith("---"):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run("―" * 50)
            run.font.color.rgb = RGBColor(203, 213, 225)
        elif line.startswith("> "):
            # Callout
            quote_text = line[2:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            run = p.add_run("┃  ")
            run.font.color.rgb = RGBColor(2, 132, 199)
            parse_inline_runs(p, quote_text, default_size=Pt(9.5), default_color=RGBColor(71, 85, 105))
        elif line.startswith("- ") or line.startswith("* "):
            item_text = line[2:].strip()
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            parse_inline_runs(p, item_text, default_size=Pt(10), default_color=RGBColor(51, 65, 85))
        elif re.match(r'^\d+\.\s+', line):
            m = re.match(r'^(\d+)\.\s+(.*)', line)
            num = m.group(1)
            item_text = m.group(2).strip()
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            parse_inline_runs(p, item_text, default_size=Pt(10), default_color=RGBColor(51, 65, 85))
        elif line.strip():
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            parse_inline_runs(p, line, default_size=Pt(10), default_color=RGBColor(51, 65, 85))

        i += 1

    if in_table:
        flush_table()
    if in_code_block:
        flush_code()

    doc.save(docx_path)
    file_size = os.path.getsize(docx_path)
    print(f"Successfully generated DOCX: {docx_path} ({file_size} bytes)")

if __name__ == "__main__":
    convert_report_to_docx(
        "/home/whysooraj/Projects/Multimodal_FMNN/PROJECT_STATUS_AND_ARCHITECTURE_REPORT.md",
        "/home/whysooraj/Projects/Multimodal_FMNN/PROJECT_STATUS_AND_ARCHITECTURE_REPORT.docx"
    )
