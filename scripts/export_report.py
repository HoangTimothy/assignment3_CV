"""Export the project's concise Markdown report to a paginated PDF."""

import argparse
import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def inline(text: str) -> str:
    text = escape(text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2" color="#126c85">\1</a>', text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def export(source: Path, destination: Path) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "Body", fontName="Helvetica", fontSize=9.5, leading=14, spaceAfter=7, alignment=TA_LEFT
        )
    )
    styles.add(ParagraphStyle("Cell", fontName="Helvetica", fontSize=7.5, leading=10))
    styles["Heading1"].fontSize = 18
    styles["Heading1"].leading = 23
    styles["Heading1"].textColor = colors.HexColor("#153c49")
    styles["Heading2"].fontSize = 12
    styles["Heading2"].leading = 17
    styles["Heading2"].textColor = colors.HexColor("#126c85")
    width = A4[0] - 36 * mm
    story = []
    lines = source.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                values = [v.strip() for v in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", v) for v in values):
                    rows.append([Paragraph(inline(v), styles["Cell"]) for v in values])
                i += 1
            table = Table(
                rows, colWidths=[width / len(rows[0])] * len(rows[0]), repeatRows=1, hAlign="LEFT"
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e1edf0")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbcdd2")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend([table, Spacer(1, 10)])
            continue
        match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
        if match:
            path = (source.parent / match[2]).resolve()
            if not path.is_file():
                raise ValueError(f"Report image missing: {path}")
            image = Image(str(path))
            ratio = min(width / image.imageWidth, 185 * mm / image.imageHeight)
            image.drawWidth = image.imageWidth * ratio
            image.drawHeight = image.imageHeight * ratio
            story.append(
                KeepTogether([image, Paragraph(inline(match[1]), styles["Cell"]), Spacer(1, 8)])
            )
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["Heading1"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["Heading2"]))
        elif line.startswith("- "):
            story.append(Paragraph("&#8226; " + inline(line[2:]), styles["Body"]))
        elif line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            story.append(Paragraph("<br/>".join(escape(v) for v in code), styles["Body"]))
        else:
            paragraph = [line]
            while (
                i + 1 < len(lines)
                and lines[i + 1].strip()
                and not lines[i + 1].startswith(("#", "|", "!", "- "))
            ):
                i += 1
                paragraph.append(lines[i].strip())
            story.append(Paragraph(inline(" ".join(paragraph)), styles["Body"]))
        i += 1
    destination.parent.mkdir(parents=True, exist_ok=True)

    def page(canvas, document):
        canvas.setStrokeColor(colors.HexColor("#c4d7dc"))
        canvas.line(18 * mm, 16 * mm, A4[0] - 18 * mm, 16 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#45626c"))
        canvas.drawString(18 * mm, 11 * mm, "Document geometry and controlled OCR | CPU evaluation")
        canvas.drawRightString(A4[0] - 18 * mm, 11 * mm, str(document.page))

    SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=22 * mm,
        title="Intelligent Document Detection and Perspective Correction",
    ).build(story, onFirstPage=page, onLaterPages=page)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
