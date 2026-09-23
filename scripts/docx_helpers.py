"""Tien ich python-docx dung de sinh bao cao (tieu de, doan van, bang, hinh, code)."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Times New Roman"


class Report:
    def __init__(self):
        self.doc = Document()
        self.fig_no = 0
        self.tab_no = 0
        self._setup_styles()

    # ---- style ---------------------------------------------------------------------------
    def _setup_styles(self):
        d = self.doc
        for s in d.sections:
            s.top_margin = s.bottom_margin = Cm(2.0)
            s.left_margin = Cm(3.0)
            s.right_margin = Cm(2.0)
        st = d.styles["Normal"]
        st.font.name = FONT
        st.font.size = Pt(13)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.line_spacing = 1.15
        for lvl, size in ((1, 16), (2, 14), (3, 13)):
            h = d.styles[f"Heading {lvl}"]
            h.font.name = FONT
            h.font.size = Pt(size)
            h.font.bold = True
            h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
            h.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
            h.paragraph_format.space_before = Pt(12 if lvl == 1 else 8)
            h.paragraph_format.space_after = Pt(4)
        for name in ("List Bullet", "List Number"):
            d.styles[name].font.name = FONT
            d.styles[name].font.size = Pt(13)

    # ---- khoi noi dung ---------------------------------------------------------------------
    def h(self, level: int, text: str):
        return self.doc.add_heading(text, level=level)

    def p(self, text: str = "", bold=False, italic=False, align=None, size=None, style=None):
        para = self.doc.add_paragraph(style=style)
        if text:
            run = para.add_run(text)
            run.bold, run.italic = bold, italic
            if size:
                run.font.size = Pt(size)
        if align == "center":
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align == "justify":
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        else:
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return para

    def rich(self, parts: list[tuple[str, dict]]):
        """parts = [(text, {"bold":..,"italic":..,"code":..}), ...]"""
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for text, fmt in parts:
            run = para.add_run(text)
            run.bold = fmt.get("bold", False)
            run.italic = fmt.get("italic", False)
            if fmt.get("code"):
                run.font.name = "Consolas"
                run.font.size = Pt(11)
        return para

    def bullets(self, items: list[str], style="List Bullet"):
        for it in items:
            para = self.doc.add_paragraph(style=style)
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            if isinstance(it, tuple):
                r = para.add_run(it[0])
                r.bold = True
                para.add_run(it[1])
            else:
                para.add_run(it)

    def numbered(self, items: list[str]):
        self.bullets(items, style="List Number")

    def code(self, text: str, size=9.5):
        for line in text.rstrip("\n").split("\n"):
            para = self.doc.add_paragraph()
            para.paragraph_format.space_after = Pt(0)
            para.paragraph_format.line_spacing = 1.0
            run = para.add_run(line if line else " ")
            run.font.name = "Consolas"
            run.font.size = Pt(size)
            self._shade(para, "F2F2F2")
        self.doc.add_paragraph()

    @staticmethod
    def _shade(para, hex_color: str):
        pPr = para._p.get_or_add_pPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_color)
        pPr.append(shd)

    def table(self, headers: list[str], rows: list[list], widths: list[float] | None = None, caption: str = "",
              size=10.5, align_center_cols: set[int] | None = None):
        if caption:
            self.tab_no += 1
            cap = self.doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = cap.add_run(f"Bảng {self.tab_no}. {caption}")
            r.bold = True
            r.font.size = Pt(11.5)
        t = self.doc.add_table(rows=1, cols=len(headers))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(headers):
            cell = t.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(h))
            run.bold = True
            run.font.size = Pt(size)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            self._shade(cell.paragraphs[0], "DCE6F1")
            self._cell_shade(cell, "DCE6F1")
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(str(v))
                run.font.size = Pt(size)
                if align_center_cols and i in align_center_cols:
                    cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Cm(w)
        self.doc.add_paragraph()
        return t

    @staticmethod
    def _cell_shade(cell, hex_color):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_color)
        tcPr.append(shd)

    def figure(self, path: str | Path, caption: str, width_cm: float = 16.0):
        path = Path(path)
        if not path.exists():
            self.p(f"[thiếu hình: {path.name}]", italic=True)
            return
        self.fig_no += 1
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.add_run().add_picture(str(path), width=Cm(width_cm))
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(f"Hình {self.fig_no}. {caption}")
        r.italic = True
        r.font.size = Pt(11.5)

    def page_break(self):
        self.doc.add_page_break()

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path))
        return Path(path)
