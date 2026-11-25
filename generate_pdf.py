import re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "ReviewPulse — Automatic Opinion Summarization: Technical Documentation")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)

        # Footer
        text = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(letter[0] - 54, 32, text)
        self.drawString(54, 32, "ReviewPulse Project Documentation · Confidential & Academic Evaluation")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 44, letter[0] - 54, 44)
        self.restoreState()


def clean_markdown_text(text):
    # Convert bold **text** to <b>text</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Convert italic *text* to <i>text</i>
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    # Convert code `code` to font
    text = re.sub(r"`(.+?)`", r'<font face="Courier" color="#0F535E">\1</font>', text)
    # Convert dollar math $...$
    text = re.sub(r"\$(.+?)\$", r"<i>\1</i>", text)
    return text


def build_pdf(md_path, pdf_path):
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0F535E"),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#167D8D"),
        spaceAfter=14,
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0F535E"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#167D8D"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4,
    )

    code_style = ParagraphStyle(
        "DocCode",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B"),
        backColor=colors.HexColor("#F8FAFC"),
        borderColor=colors.HexColor("#E2E8F0"),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=8,
    )

    lines = Path(md_path).read_text(encoding="utf-8").splitlines()
    story = []

    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []

    for line in lines:
        stripped = line.strip()

        # Handle code blocks
        if stripped.startswith("```"):
            if in_code_block:
                code_text = "<br/>".join([clean_markdown_text(cl.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")) for cl in code_lines])
                story.append(Paragraph(code_text, code_style))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        # Handle Markdown tables
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            # Check if this is a separator line | --- | --- |
            if all(re.match(r"^:?-+:?$", c) for c in cells):
                continue
            table_rows.append(cells)
            in_table = True
            continue
        elif in_table:
            # End of table
            if table_rows:
                formatted_data = []
                for row_idx, r in enumerate(table_rows):
                    row_cells = []
                    for c in r:
                        c_text = clean_markdown_text(c)
                        c_style = ParagraphStyle(
                            f"Cell_{row_idx}",
                            parent=body_style,
                            fontSize=8.5,
                            leading=11,
                            fontName="Helvetica-Bold" if row_idx == 0 else "Helvetica",
                            textColor=colors.HexColor("#0F535E") if row_idx == 0 else colors.HexColor("#1E293B"),
                        )
                        row_cells.append(Paragraph(c_text, c_style))
                    formatted_data.append(row_cells)

                t = Table(formatted_data, colWidths=None)
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F5F7")),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(Spacer(1, 4))
                story.append(t)
                story.append(Spacer(1, 8))
            table_rows = []
            in_table = False

        if not stripped:
            story.append(Spacer(1, 4))
            continue

        # Headings
        if stripped.startswith("# "):
            title_text = clean_markdown_text(stripped[2:])
            story.append(Paragraph(title_text, title_style))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#167D8D"), spaceAfter=8))
        elif stripped.startswith("## "):
            h1_text = clean_markdown_text(stripped[3:])
            story.append(Spacer(1, 6))
            story.append(Paragraph(h1_text, h1_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=6))
        elif stripped.startswith("### "):
            h2_text = clean_markdown_text(stripped[4:])
            story.append(Paragraph(h2_text, h2_style))
        elif stripped.startswith("---"):
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0"), spaceAfter=6))
        elif stripped.startswith(("* ", "- ")):
            bullet_text = clean_markdown_text(stripped[2:])
            story.append(Paragraph(f"• {bullet_text}", bullet_style))
        elif re.match(r"^\d+\.\s", stripped):
            num_match = re.match(r"^(\d+\.)\s*(.*)$", stripped)
            prefix = num_match.group(1)
            item_text = clean_markdown_text(num_match.group(2))
            story.append(Paragraph(f"<b>{prefix}</b> {item_text}", bullet_style))
        elif stripped.startswith("> "):
            quote_text = clean_markdown_text(stripped[2:])
            q_style = ParagraphStyle(
                "Blockquote",
                parent=body_style,
                leftIndent=14,
                textColor=colors.HexColor("#334155"),
                backColor=colors.HexColor("#F0F5F7"),
                borderColor=colors.HexColor("#167D8D"),
                borderWidth=1,
                borderPadding=6,
            )
            story.append(Paragraph(quote_text, q_style))
        else:
            p_text = clean_markdown_text(stripped)
            story.append(Paragraph(p_text, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF built successfully at {pdf_path}")


if __name__ == "__main__":
    src = Path("DOCUMENTATION.md")
    dest = Path("ReviewPulse_Project_Documentation.pdf")
    build_pdf(src, dest)
