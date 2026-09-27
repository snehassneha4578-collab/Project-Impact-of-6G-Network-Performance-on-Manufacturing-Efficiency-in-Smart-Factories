from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import re
import os

INPUT_FILE = "6G_Smart_Factory_Research_Paper.md"
OUTPUT_FILE = "6G_Smart_Factory_Research_Paper.pdf"

# Register a Unicode-capable font available on Windows.
font_paths = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
]

font_path = next((p for p in font_paths if os.path.exists(p)), None)

if font_path:
    pdfmetrics.registerFont(TTFont("CustomFont", font_path))
    base_font = "CustomFont"
else:
    base_font = "Helvetica"

doc = SimpleDocTemplate(
    OUTPUT_FILE,
    pagesize=A4,
    rightMargin=54,
    leftMargin=54,
    topMargin=54,
    bottomMargin=54,
    title="Impact of 6G Network Performance on Manufacturing Efficiency in Smart Factories",
    author="Sneha S",
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "PaperTitle",
    parent=styles["Title"],
    fontName=base_font,
    fontSize=18,
    leading=23,
    alignment=TA_CENTER,
    spaceAfter=18,
)

author_style = ParagraphStyle(
    "Author",
    parent=styles["Normal"],
    fontName=base_font,
    fontSize=11,
    leading=16,
    alignment=TA_CENTER,
    spaceAfter=16,
)

heading_style = ParagraphStyle(
    "Heading",
    parent=styles["Heading1"],
    fontName=base_font,
    fontSize=13,
    leading=17,
    spaceBefore=12,
    spaceAfter=8,
)

subheading_style = ParagraphStyle(
    "SubHeading",
    parent=styles["Heading2"],
    fontName=base_font,
    fontSize=11,
    leading=15,
    spaceBefore=8,
    spaceAfter=5,
)

body_style = ParagraphStyle(
    "Body",
    parent=styles["BodyText"],
    fontName=base_font,
    fontSize=10,
    leading=15,
    spaceAfter=8,
)

bullet_style = ParagraphStyle(
    "Bullet",
    parent=body_style,
    leftIndent=18,
    firstLineIndent=-9,
    spaceAfter=4,
)

story = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

first_title = True
i = 0

while i < len(lines):
    line = lines[i].strip()

    if not line:
        i += 1
        continue

    # Markdown title
    if line.startswith("# ") and first_title:
        story.append(Paragraph(line[2:], title_style))
        first_title = False
        i += 1
        continue

    # Author
    if line == "**Sneha S**":
        story.append(Paragraph("Sneha S", author_style))
        i += 1
        continue

    # Main headings
    if re.match(r"^## ", line):
        text = line[3:]
        story.append(Paragraph(text, heading_style))
        i += 1
        continue

    # Subheadings
    if re.match(r"^### ", line):
        text = line[4:]
        story.append(Paragraph(text, subheading_style))
        i += 1
        continue

    # Bullet points
    if line.startswith("- "):
        text = line[2:]
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        story.append(Paragraph("- " + text, bullet_style))
        i += 1
        continue

    # Numbered list
    if re.match(r"^\d+\.\s", line):
        number = re.match(r"^(\d+)\.", line).group(1)
        text = re.sub(r"^\d+\.\s*", "", line)
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        story.append(Paragraph(number + ". " + text, body_style))
        i += 1
        continue

    # Horizontal rule
    if line == "---":
        story.append(Spacer(1, 8))
        i += 1
        continue

    # Normal paragraph
    text = line
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    text = text.replace("&", "&amp;")

    story.append(Paragraph(text, body_style))
    i += 1

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont(base_font, 8)
    canvas.drawCentredString(A4[0] / 2, 25, f"Page {doc.page}")
    canvas.restoreState()

doc.build(
    story,
    onFirstPage=add_page_number,
    onLaterPages=add_page_number,
)

print(f"PDF created successfully: {OUTPUT_FILE}")
