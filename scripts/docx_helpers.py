"""Tien ich python-docx dung de sinh bao cao (tieu de, doan van, bang, hinh, code)."""
from __future__ import annotations

import glob
import os
from functools import lru_cache
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Emu, Pt, RGBColor

FONT = "Times New Roman"


class Report:
    caption_pt = 11.5

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
        """Moi danh sach danh so lai tu 1 (mac dinh Word noi tiep so giua cac danh sach cung style)."""
        numbering = self.doc.part.numbering_part.numbering_definitions._numbering
        style_num = self.doc.styles["List Number"].element.pPr.numPr.numId.val
        num = numbering.add_num(numbering.num_having_numId(style_num).abstractNumId.val)
        num.add_lvlOverride(ilvl=0).add_startOverride(1)
        self.bullets(items, style="List Number")
        for para in self.doc.paragraphs[-len(items):]:
            numpr = para._p.get_or_add_pPr().get_or_add_numPr()
            numpr.get_or_add_ilvl().val = 0
            numpr.get_or_add_numId().val = num.numId

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
            cap.paragraph_format.keep_with_next = True
            r = cap.add_run(f"Bảng {self.tab_no}. {caption}")
            r.bold = True
            r.font.size = Pt(self.caption_pt)
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
        para.paragraph_format.keep_with_next = True
        para.add_run().add_picture(str(path), width=Cm(width_cm))
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(f"Hình {self.fig_no}. {caption}")
        r.italic = True
        r.font.size = Pt(self.caption_pt)

    def page_break(self):
        self.doc.add_page_break()

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path))
        return Path(path)


# ---------------------------------------------------------------------------------------------
# Cong thuc toan: LaTeX -> MathML (latex2mathml) -> OMML (MML2OMML.XSL di kem MS Office)
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _mml2omml():
    """Bo chuyen MathML -> OMML cua Office; None neu may khong co Office / lxml / latex2mathml."""
    try:
        from latex2mathml.converter import convert  # noqa: F401
        from lxml import etree
    except ImportError:
        return None
    cands = [os.environ.get("MML2OMML_XSL", "")]
    for root in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", "")):
        if root:
            cands += glob.glob(os.path.join(root, "Microsoft Office", "root", "Office*", "MML2OMML.XSL"))
            cands += glob.glob(os.path.join(root, "Microsoft Office", "Office*", "MML2OMML.XSL"))
    for c in cands:
        if c and os.path.isfile(c):
            return etree.XSLT(etree.parse(c))
    return None


def latex_to_omml(latex: str):
    """Tra ve phan tu m:oMath (python-docx oxml) hoac None neu khong chuyen duoc."""
    xslt = _mml2omml()
    if xslt is None:
        return None
    from latex2mathml.converter import convert
    from lxml import etree
    omml = xslt(etree.fromstring(convert(latex).encode("utf-8"))).getroot()
    return parse_xml(etree.tostring(omml))


def _field_run(para, kind: str | None = None, instr: str | None = None, size: float | None = None):
    """Mot run cua truong Word: kind = begin/separate/end, hoac instr = ma truong (TOC, PAGE...)."""
    run = para.add_run()
    if size:
        run.font.size = Pt(size)
    if kind:
        fc = OxmlElement("w:fldChar")
        fc.set(qn("w:fldCharType"), kind)
        if kind == "begin":
            fc.set(qn("w:dirty"), "true")
        run._r.append(fc)
    else:
        it = OxmlElement("w:instrText")
        it.set(qn("xml:space"), "preserve")
        it.text = instr
        run._r.append(it)
    return run


class PtitReport(Report):
    """Bao cao theo mau trinh bay cua hoc phan: A4, Times New Roman 14, tieu de mau den (chuong can giua, sang trang),
    bia co khung + logo, muc luc tu dong, so trang o chan trang, cong thuc la equation that cua Word."""
    caption_pt = 13
    body_pt = 14
    indent_cm = 1.0

    def _setup_styles(self):
        super()._setup_styles()
        d = self.doc
        for s in d.sections:
            s.page_width, s.page_height = Cm(21.0), Cm(29.7)
            s.header_distance = s.footer_distance = Cm(1.0)
        d.styles["Normal"].font.size = Pt(self.body_pt)
        for lvl, size in ((1, 20), (2, 16), (3, 14)):
            h = d.styles[f"Heading {lvl}"]
            h.font.size = Pt(size)
            h.font.bold = True
            h.font.italic = False
            h.font.color.rgb = RGBColor(0, 0, 0)
            rpr = h.element.rPr
            for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                rpr.rFonts.attrib.pop(qn(attr), None)
            for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                rpr.rFonts.set(qn(attr), FONT)
            for attr in ("w:themeColor", "w:themeShade", "w:themeTint"):
                rpr.color.attrib.pop(qn(attr), None)
            h.paragraph_format.keep_with_next = True
        h1 = d.styles["Heading 1"].paragraph_format
        h1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        h1.page_break_before = True
        h1.space_before, h1.space_after = Pt(0), Pt(18)
        h2 = d.styles["Heading 2"].paragraph_format
        h2.space_before, h2.space_after = Pt(18), Pt(6)
        h3 = d.styles["Heading 3"].paragraph_format
        h3.space_before, h3.space_after = Pt(12), Pt(4)
        for name in ("List Bullet", "List Number"):
            d.styles[name].font.size = Pt(self.body_pt)

    # ---- doan van: thut dong dau ----------------------------------------------------------------
    def p(self, text: str = "", bold=False, italic=False, align=None, size=None, style=None):
        para = super().p(text, bold=bold, italic=italic, align=align, size=size, style=style)
        if align != "center" and text:
            para.paragraph_format.first_line_indent = Cm(self.indent_cm)
        return para

    def rich(self, parts):
        para = super().rich(parts)
        para.paragraph_format.first_line_indent = Cm(self.indent_cm)
        return para

    def table(self, *args, **kwargs):
        t = super().table(*args, **kwargs)
        for row in t.rows:            # bang gon: khong gian cach doan trong o
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.space_after = Pt(1)
                    para.paragraph_format.line_spacing = 1.0
        return t

    # ---- cong thuc ------------------------------------------------------------------------------
    def pm(self, text: str, lead: str = ""):
        """Doan van co cong thuc noi dong viet trong $...$ (LaTeX); lead = cum tu in dam o dau doan."""
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        para.paragraph_format.first_line_indent = Cm(self.indent_cm)
        if lead:
            para.add_run(lead).bold = True
        for k, seg in enumerate(text.split("$")):
            if not seg:
                continue
            omath = latex_to_omml(seg) if k % 2 else None
            if k % 2 and omath is not None:
                para._p.append(omath)
            else:
                run = para.add_run(seg)
                run.italic = bool(k % 2)
        return para

    def eq(self, latex: str, fallback: str = ""):
        """Cong thuc dong rieng (display), can giua; khong co Office thi ghi dang van ban."""
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        omath = latex_to_omml(latex)
        if omath is None:
            run = para.add_run(fallback or latex)
            run.italic = True
            return para
        for pr in omath.iter(qn("m:naryPr")):      # cong thuc dong rieng: can tren/duoi cua tong nam tren/duoi
            loc = pr.find(qn("m:limLoc"))
            if loc is None:                         # thu tu trong naryPr: chr, limLoc, ...
                loc = OxmlElement("m:limLoc")
                pr.insert(1 if pr.find(qn("m:chr")) is not None else 0, loc)
            loc.set(qn("m:val"), "undOvr")
        opara = OxmlElement("m:oMathPara")
        opara.append(omath)
        para._p.append(opara)
        return para

    # ---- bia, muc luc, so trang -----------------------------------------------------------------
    def background_image(self, path: str | Path, cx: int, cy: int):
        """Anh nam sau van ban, can giua vung le (khung bia)."""
        para = self.doc.add_paragraph()
        para.paragraph_format.space_after = Pt(0)
        run = para.add_run()
        run.font.size = Pt(1)
        run.add_picture(str(path), width=Emu(cx), height=Emu(cy))
        inline = run._r.xpath(".//wp:inline")[0]
        anchor = parse_xml(
            f'<wp:anchor {nsdecls("wp")} distT="0" distB="0" distL="0" distR="0" simplePos="0" '
            'relativeHeight="251658240" behindDoc="1" locked="0" layoutInCell="1" allowOverlap="1">'
            '<wp:simplePos x="0" y="0"/>'
            '<wp:positionH relativeFrom="margin"><wp:align>center</wp:align></wp:positionH>'
            '<wp:positionV relativeFrom="margin"><wp:align>center</wp:align></wp:positionV>'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/><wp:wrapNone/>'
            '</wp:anchor>')
        for tag in ("wp:docPr", "wp:cNvGraphicFramePr", "a:graphic"):
            el = inline.find(qn(tag))
            if el is not None:
                anchor.append(el)
        inline.getparent().replace(inline, anchor)
        return para

    def picture_center(self, path: str | Path, width_cm: float, space_before: float = 0, space_after: float = 0):
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before, para.paragraph_format.space_after = Pt(space_before), Pt(space_after)
        para.add_run().add_picture(str(path), width=Cm(width_cm))
        return para

    def line(self, text: str, size: float, bold=True, italic=False, before: float = 0, after: float = 4):
        """Mot dong can giua voi co chu rieng (bia, tieu de muc luc)."""
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before, para.paragraph_format.space_after = Pt(before), Pt(after)
        run = para.add_run(text)
        run.bold, run.italic = bold, italic
        run.font.size = Pt(size)
        return para

    def info_block(self, rows: list[tuple[str, str]], widths=(5.6, 8.6), size: float = 14):
        """Khoi thong tin tren bia: 2 cot khong ke khung (nhan in dam | gia tri)."""
        t = self.doc.add_table(rows=0, cols=2)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for label, value in rows:
            cells = t.add_row().cells
            for cell, text, bold in ((cells[0], label, True), (cells[1], value, False)):
                para = cell.paragraphs[0]
                para.paragraph_format.space_after = Pt(2)
                run = para.add_run(text)
                run.bold = bold
                run.font.size = Pt(size)
        for row in t.rows:
            for cell, w in zip(row.cells, widths):
                cell.width = Cm(w)
        return t

    def toc(self, title: str = "MỤC LỤC", levels: str = "1-3"):
        """Muc luc tu dong (truong TOC); Word dien so trang khi cap nhat truong (F9) hoac khi xuat bang Word."""
        self.line(title, 20, after=12)
        para = self.doc.add_paragraph()
        _field_run(para, "begin")
        _field_run(para, instr=f' TOC \\o "{levels}" \\h \\z \\u ')
        _field_run(para, "separate")
        para.add_run("Mở bằng Word và nhấn F9 (Update Field) để cập nhật mục lục.").italic = True
        _field_run(para, "end")
        from docx.enum.style import WD_STYLE_TYPE
        for lvl in (1, 2, 3):           # kieu cac dong muc luc
            name = f"toc {lvl}"
            try:
                st = self.doc.styles[name]
            except KeyError:
                st = self.doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
                st.base_style = self.doc.styles["Normal"]
            st.font.size = Pt(13)
            st.font.bold = lvl == 1
            st.paragraph_format.left_indent = Cm(0.6 * (lvl - 1))
            st.paragraph_format.space_after = Pt(3)

    def page_numbers(self):
        """So trang giua chan trang; trang bia khong danh so."""
        sec = self.doc.sections[0]
        sec.different_first_page_header_footer = True
        para = sec.footer.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _field_run(para, "begin", size=13)
        _field_run(para, instr=" PAGE ", size=13)
        _field_run(para, "end", size=13)
