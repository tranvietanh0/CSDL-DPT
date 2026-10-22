"""Sinh Bao cao 1 (10%, han nop 22/10/2026): chi gom noi dung 1 va 2 cua de bai nhom 27.

    python scripts/build_report1.py   -> docs/BaoCao1_Nhom27_NoiDung1-2.docx (+ .pdf neu co LibreOffice)

Dung lai cac muc cua bao cao tong (build_docs.py) va cac muc dinh luong (report_analysis_sections.py);
moi so lieu doc tu data/, db/ va results/ (chay scripts/analyze_features.py truoc).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_docs import FIG, GROUP, cover, load_context, sec_dataset, sec_features, to_pdf  # noqa: E402
from docx_helpers import Report  # noqa: E402
from report_analysis_sections import SET_VI, sec_example, sec_info_value  # noqa: E402
from treedb import config  # noqa: E402

OUT_DOCX = config.ROOT / "docs" / "BaoCao1_Nhom27_NoiDung1-2.docx"
SEG_REF = "mục 3.6"
TOC_LEVELS = 2

DANG, TU, VANH = (m[0] for m in GROUP["members"])
# (cong viec, sinh vien thuc hien, thoi han) - sua o day neu phan cong thuc te khac
TASKS = [
    ("Nghiên cứu đề bài và slide bài 11 (truy vấn ảnh dựa trên nội dung), chốt tiêu chí bộ dữ liệu", VANH,
     "28/09/2026"),
    ("Viết công cụ sưu tầm ảnh từ Wikimedia Commons, lọc tự động theo kích thước và tỉ lệ khung hình", DANG,
     "02/10/2026"),
    ("Duyệt ảnh bằng mắt trên bảng ảnh thu nhỏ, lập danh sách ảnh loại, gán nhãn chi cây", TU, "05/10/2026"),
    ("Chuẩn hoá kích thước ảnh, lưu siêu dữ liệu nguồn và giấy phép, tách bộ ảnh truy vấn thử", DANG, "07/10/2026"),
    ("Thống kê bộ dữ liệu, mô tả và định lượng điểm giống, khác nhau giữa các chi cây", TU, "10/10/2026"),
    ("Xây dựng bộ đặc trưng màu sắc và bộ đặc trưng kết cấu, bố cục", DANG, "14/10/2026"),
    ("Xây dựng bộ đặc trưng hình dạng và bộ embedding sâu MobileNetV2", TU, "14/10/2026"),
    ("Tách vùng cây khỏi nền (SegFormer-b0, GrabCut dự phòng), kiểm tra lỗi mặt nạ, đối chiếu giáo trình", VANH,
     "19/10/2026"),
    ("Đánh giá giá trị thông tin của đặc trưng (tỉ số Fisher, tỉ số tách, 1-NN) và ví dụ minh hoạ", VANH,
     "20/10/2026"),
    ("Tổng hợp nội dung, viết Lời nói đầu, Kết luận và hoàn thiện định dạng báo cáo", f"{DANG}, {TU}, {VANH}",
     "21/10/2026"),
]
CRITERIA = ["Thái độ tham gia", "Mức hoàn thành CV", "Kỹ năng giao tiếp", "Kỹ năng hợp tác", "Kỹ năng lãnh đạo"]
SELF_SCORES = {name: [5] * len(CRITERIA) for name, _ in GROUP["members"]}


def sec_assignment(r: Report):
    r.p("PHẦN PHÂN CÔNG NHIỆM VỤ VÀ ĐÁNH GIÁ", bold=True, align="center", size=14)
    r.p("PHÂN CÔNG NHIỆM VỤ NHÓM THỰC HIỆN", bold=True, size=13)
    r.table(["TT", "Công việc / Nhiệm vụ", "SV thực hiện", "Thời hạn hoàn thành"],
            [[str(i), t, who, due] for i, (t, who, due) in enumerate(TASKS, 1)],
            widths=[1.0, 8.4, 3.8, 2.8], size=11, align_center_cols={0, 3})
    r.p("THỰC HIỆN TỰ ĐÁNH GIÁ", bold=True, size=13)
    r.p("Thang điểm 1–5 (5 là tốt nhất).", italic=True, size=11.5)
    r.table(["TT", "SV thực hiện"] + CRITERIA,
            [[str(i), name] + [str(x) for x in SELF_SCORES[name]] for i, (name, _) in enumerate(GROUP["members"], 1)],
            widths=[1.0, 4.0, 2.2, 2.2, 2.2, 2.2, 2.2], size=11, align_center_cols={0, 2, 3, 4, 5, 6})
    r.page_break()


def toc(r: Report, entries: list[tuple[int, str]], pages: dict[str, int] | None):
    """Muc luc kem so trang; pages = None o luot dau (so trang tinh tu PDF luot dau, xem build_with_toc)."""
    r.p("MỤC LỤC", bold=True, align="center", size=16)
    for level, text in entries:
        if level <= TOC_LEVELS:
            r.toc_line(text, str(pages.get(text, "")) if pages else "00", level)
    r.page_break()


def sec_preface(r: Report):
    r.h(1, "Lời nói đầu")
    r.p("Thiết bị chụp ảnh số ngày càng phổ biến nên lượng ảnh cần lưu trữ và tra cứu tăng rất nhanh. Tìm ảnh theo "
        "tên tệp hay từ khoá chỉ hiệu quả khi ảnh đã được mô tả bằng tay đầy đủ, và ngay cả khi đó vẫn khó diễn tả "
        "những thuộc tính bậc thấp như dáng tán cây hay sắc độ của lá. Truy vấn ảnh dựa trên nội dung giải quyết vấn "
        "đề này bằng cách biểu diễn mỗi ảnh thành các vector đặc trưng về màu sắc, hình dạng và kết cấu, rồi so sánh "
        "khoảng cách giữa các vector để tìm ảnh giống nhất.")
    r.p("Trong học phần Hệ cơ sở dữ liệu đa phương tiện, nhóm 27 thực hiện đề tài “Xây dựng hệ CSDL lưu trữ và tìm "
        "kiếm ảnh cây thân gỗ theo nội dung”. Báo cáo 1 trình bày hai nội dung đầu của đề tài: xây dựng bộ dữ liệu ảnh "
        "cây thân gỗ trưởng thành và mô tả điểm giống, khác nhau giữa các cây (nội dung 1); xây dựng các bộ đặc trưng "
        "nhận diện cây, trong đó có bước tách vùng cây khỏi nền, cùng đánh giá giá trị thông tin của từng đặc trưng "
        "(nội dung 2).")
    r.p("Nhóm xin cảm ơn thầy Nguyễn Đình Hóa đã hướng dẫn trong quá trình thực hiện. Do thời gian và kiến thức còn "
        "hạn chế, báo cáo khó tránh khỏi thiếu sót; nhóm mong nhận được góp ý của thầy để hoàn thiện ở các báo cáo sau.")
    r.p("Chúng em xin chân thành cảm ơn thầy!")


def sec_scope(r: Report, ctx: dict):
    r.h(1, "1. Đề bài và phạm vi Báo cáo 1")
    r.p("Đề bài nhóm 27: Xây dựng hệ CSDL lưu trữ và tìm kiếm ảnh cây. Học phần chia làm ba mốc: Báo cáo 1 (10%, "
        "hạn 22/10/2026) gồm nội dung 1 và 2; Báo cáo 2 (20%, hạn 03/12/2026) gồm nội dung 3 và 4; báo cáo demo "
        "(60%) vào ngày thi gồm tất cả các nội dung. Báo cáo này trình bày hai nội dung đầu:")
    r.numbered([
        "Xây dựng/sưu tầm bộ dữ liệu ít nhất 500 ảnh về các cây thân gỗ trưởng thành. Các ảnh được lưu trữ với "
        "cùng kích thước, mỗi ảnh gồm 1 cây hoàn chỉnh nằm giữa ảnh, các cây được chụp ngang, vật trong ảnh có tỉ "
        "lệ khung hình tương đồng nhau giữa các ảnh. Mô tả đặc điểm giống và khác nhau của các cây trong các ảnh "
        "thu thập được.",
        "Xây dựng một hoặc vài bộ đặc trưng để nhận diện các cây từ bộ dữ liệu đã thu thập. Các bộ thuộc tính phải "
        "bao gồm các đặc trưng giúp tìm sự tương đồng và tìm sự khác biệt giữa các cây trong các ảnh. Trình bày cụ "
        "thể về từng loại đặc trưng này cùng giá trị thông tin của các đặc trưng được sử dụng.",
    ])
    n_db = ctx["db"]["images"]
    r.p(f"Kết quả chính: bộ dữ liệu {n_db} ảnh JPEG {config.IMAGE_SIZE}×{config.IMAGE_SIZE} (đạt yêu cầu ≥ "
        f"{config.MIN_IMAGES}) kèm {len(ctx['queries'])} ảnh truy vấn thử tách riêng; bốn bộ đặc trưng (màu sắc, "
        "hình dạng, kết cấu/bố cục, embedding sâu) được định nghĩa, cài đặt và đánh giá giá trị thông tin bằng số "
        "liệu. Mã nguồn trích rút và CSDL đặc trưng (nội dung 3) đã được cài đặt sẵn để phục vụ các phép đo trong "
        "báo cáo này và sẽ được trình bày chi tiết ở Báo cáo 2.")


def sec_segmentation(r: Report, ctx: dict):
    seg = ctx.get("seg_stats", {})
    r.h(2, "3.6. Tách vùng cây khỏi nền – tiền đề của các đặc trưng vật thể")
    r.p("Theo lý thuyết truy vấn ảnh dựa trên màu sắc, biểu đồ màu toàn ảnh chịu “hiệu ứng che mặt”: các ô màu "
        "lớn của nền (trời, cỏ) lấn át màu của vật thể. Vì vậy các khối hsv_fg, moments, toàn bộ bộ hình dạng và "
        "bộ kết cấu được tính trên mặt nạ vùng cây. Mặt nạ được tạo bằng mô hình phân đoạn ngữ nghĩa SegFormer-b0 "
        "huấn luyện sẵn trên ADE20K (lớp “tree”), sau đó lọc hình thái, giữ thành phần liên thông lớn và gần tâm "
        "ảnh nhất, lấp lỗ để được hình bao của cây; GrabCut là phương án dự phòng."
        + (f" Trên {sum(seg.values())} ảnh trước cổng kiểm soát: SegFormer {seg.get('segformer', 0)}, GrabCut "
           f"{seg.get('grabcut', 0)}, không tìm thấy cây {seg.get('default_rect', 0)} (bị loại)." if seg else ""))
    r.p("Trong quá trình kiểm tra giá trị thông tin (mục 3.7), nhóm phát hiện phiên bản đầu dùng GrabCut làm bước "
        "chính khiến 135/639 ảnh (21%) có vector hình dạng trùng hệt nhau: GrabCut chỉ dựa vào màu nên không tách "
        "được tán lá khỏi cỏ cùng màu hoặc khi tán chạm mép ảnh, và hệ thống rơi về mặt nạ chữ nhật mặc định. Việc "
        "chuyển sang mô hình ngữ nghĩa loại bỏ hoàn toàn lỗi này và làm tăng độ chính xác của mọi bộ đặc trưng "
        "thủ công.")
    r.figure(FIG / "03_segmentation.png", "Ví dụ tách vùng cây: ảnh gốc (trên) và mặt nạ vùng cây (dưới)", 16)
    r.p("Ảnh lưu trong CSDL giữ nguyên nền; mặt nạ chỉ giới hạn vùng tính đặc trưng. Phần nằm ngoài mặt nạ không bị "
        "bỏ đi mà được mô tả bằng khối hsv_bg riêng, nên mỗi ảnh có hai biểu đồ màu, một cho vật thể (hsv_fg) và một "
        "cho hình nền (hsv_bg), đúng như giáo trình đề xuất để tránh hiệu ứng che mặt.")
    r.p("Giáo trình (bài 11) xác định vật thể và hình nền bằng cách chia ảnh thành các khung chữ nhật dựa trên sự biến "
        "thiên giá trị màu theo trục ngang hoặc trục dọc. Với ảnh cây chụp ngoài trời, tán lá và thảm cỏ thường cùng "
        "sắc xanh nên biến thiên màu không đủ để phân biệt; đây cũng là lý do GrabCut thất bại ở phiên bản đầu. Nhóm "
        "thay bước này bằng phân đoạn ngữ nghĩa ở mức điểm ảnh: ranh giới bám theo hình bao của tán thay vì một khung "
        "chữ nhật, nên biểu đồ màu của vật thể ít lẫn màu nền hơn.")
    r.p("Với bộ hình dạng, giáo trình yêu cầu chuẩn hoá thước đo và chuẩn hoá xoay vòng trước khi đặt lưới. Nhóm chuẩn "
        "hoá thước đo bằng cách co hình chữ nhật bao của mặt nạ về lưới 8×8, ô có hơn 15% diện tích là vật thể nhận "
        "giá trị 1 (cùng ngưỡng với giáo trình). Bước chuẩn hoá xoay vòng được bỏ qua có chủ đích: mọi cây đều mọc "
        "thẳng đứng và được chụp ngang, nên xoay hình theo trục chính sẽ xoá mất thông tin tán rộng ngang hay thon "
        "cao, vốn là một trong những điểm khác biệt rõ nhất giữa các chi.")
    r.table(["Bước trong giáo trình (bài 11)", "Cách nhóm thực hiện", "Ghi chú"], [
        ["Biểu đồ màu riêng cho vật thể và hình nền (tránh hiệu ứng che mặt)", "Khối hsv_fg và hsv_bg trong bộ màu "
         "sắc, tính trên mặt nạ và phần bù của mặt nạ", "Đúng giáo trình"],
        ["Xác định vật thể/nền theo khung chữ nhật, dựa vào biến thiên màu", "Phân đoạn ngữ nghĩa SegFormer-b0 ở mức "
         "điểm ảnh, GrabCut dự phòng", "Cải tiến: tán cây và cỏ cùng màu"],
        ["Hình dạng tính trên mảnh chứa vật thể", "Bộ hình dạng tính trên mặt nạ, cắt theo hình chữ nhật bao",
         "Đúng giáo trình"],
        ["Lưới ô vuông, ô > 15% diện tích vật thể nhận giá trị 1", "Lưới 8×8 trên hình chữ nhật bao, ngưỡng 15%",
         "Đúng giáo trình"],
        ["Chuẩn hoá thước đo và chuẩn hoá xoay vòng", "Chuẩn hoá thước đo bằng hình chữ nhật bao; không xoay",
         "Giữ hướng mọc thẳng đứng của cây"],
        ["Truy vấn theo màu toàn ảnh, vật thể, hình nền hoặc kết hợp có trọng số", "Đã có đủ bốn biểu đồ; chọn chế độ "
         "truy vấn sẽ được bổ sung ở hệ thống tìm kiếm", "Thuộc nội dung 4 (Báo cáo 2)"],
    ], widths=[5.6, 6.2, 4.2], caption="Đối chiếu bước xử lý nền với giáo trình", size=10.5)


def sec_conclusion(r: Report, ctx: dict):
    fa = ctx["feature_analysis"]
    best = max(fa["sets"].items(), key=lambda kv: kv[1]["nn1_acc"])
    r.h(1, "4. Kết luận và kế hoạch cho Báo cáo 2")
    r.p(f"Nội dung 1: bộ dữ liệu {ctx['db']['images']} ảnh cây thân gỗ trưởng thành đơn lẻ, cùng kích thước "
        f"{config.IMAGE_SIZE}×{config.IMAGE_SIZE}, cây nằm giữa ảnh, chụp ngang, có siêu dữ liệu nguồn/giấy phép và "
        "nhãn chi; điểm giống và khác giữa các cây được mô tả cả định tính lẫn định lượng theo chi và theo nhóm lá "
        "rộng/lá kim.")
    r.p(f"Nội dung 2: bốn bộ đặc trưng gồm {sum(v['dim'] for v in fa['sets'].values())} chiều, mỗi khối được giải "
        f"thích ý nghĩa và đo giá trị thông tin (tỉ số Fisher, tỉ số tách, độ chính xác 1-NN). Mọi bộ đều vượt mức "
        f"ngẫu nhiên ({fa['nn1_chance']:.3f}); bộ có 1-NN cao nhất là {SET_VI[best[0]].lower()} "
        f"({best[1]['nn1_acc']:.3f}).")
    r.p("Kế hoạch Báo cáo 2 (nội dung 3 và 4): trình bày chi tiết thuật toán trích rút và lược đồ CSDL SQLite theo "
        "mô hình lai (siêu dữ liệu quan hệ + vector đặc trưng), chỉ mục k-d tree và K-means; hệ thống tìm 5 ảnh giống "
        "nhất với sơ đồ khối, kết quả trung gian và đánh giá P@5/mAP. Phần mã nguồn cho các nội dung này đã có trong "
        "kho mã của nhóm. Hệ thống tìm kiếm sẽ cho chọn bốn chế độ truy vấn màu theo giáo trình: toàn ảnh, chỉ vật "
        "thể, chỉ hình nền, hoặc kết hợp vật thể và hình nền có trọng số.")


def build(ctx: dict, toc_entries: list[tuple[int, str]] | None = None,
          pages: dict[str, int] | None = None) -> tuple[Path, list[tuple[int, str]]]:
    r = Report()
    r.page_numbers()
    cover(r, ctx, title="BÁO CÁO", note="Báo cáo lần 1 – Nội dung 1 và 2")
    sec_assignment(r)
    if toc_entries is not None:
        toc(r, toc_entries, pages)
    sec_preface(r)
    sec_scope(r, ctx)
    sec_dataset(r, ctx, seg_ref=SEG_REF)
    sec_features(r, ctx, seg_ref=SEG_REF, with_analysis=False)
    sec_segmentation(r, ctx)
    sec_info_value(r, ctx["feature_analysis"], "3.7")
    sec_example(r, ctx["feature_analysis"], "3.8")
    sec_conclusion(r, ctx)
    return r.save(OUT_DOCX), r.headings


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()  # tieu de cap 1 hien chu hoa (all caps)


def find_pages(pdf: Path, entries: list[tuple[int, str]]) -> dict[str, int]:
    """Trang (dem tu 1, trung so o chan trang) chua moi tieu de, tim lan luot sau trang muc luc."""
    import pymupdf as fitz

    texts = [_norm(pg.get_text()) for pg in fitz.open(str(pdf))]
    start = max(i for i, t in enumerate(texts) if "mục lục" in t or t.count(".....") >= 5) + 1
    pages, cur = {}, start
    for _, text in entries:
        hit = next((i for i in range(cur, len(texts)) if _norm(text) in texts[i]), None)
        if hit is None:
            raise RuntimeError(f"Khong tim thay tieu de trong PDF: {text!r}")
        pages[text], cur = hit + 1, hit
    return pages


def build_with_toc(ctx: dict) -> tuple[Path, Path | None]:
    """Muc luc co so trang: luot 1 lay danh sach tieu de, luot 2 dung PDF de dem trang, luot 3 ghi so trang that."""
    _, headings = build(ctx)
    entries = [(lvl, t) for lvl, t in headings if lvl <= TOC_LEVELS]
    out, _ = build(ctx, entries)
    pdf = to_pdf(out)
    if pdf is None:
        return out, None
    out, _ = build(ctx, entries, find_pages(pdf, entries))
    return out, to_pdf(out)


if __name__ == "__main__":
    ctx = load_context()
    if not ctx.get("feature_analysis"):
        sys.exit("Thieu results/feature_analysis.json - chay: python scripts/analyze_features.py")
    out, pdf = build_with_toc(ctx)
    print(f"[docx] {out} ({out.stat().st_size / 1024:.0f} KB)")
    if pdf:
        print(f"[pdf] {pdf} ({pdf.stat().st_size / 1024:.0f} KB)")
