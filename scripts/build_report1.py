"""Sinh Bao cao 1 (10%, han nop 22/10/2026): chi gom noi dung 1 va 2 cua de bai nhom 27.

    python scripts/build_report1.py   -> docs/BaoCao1_Nhom27_NoiDung1-2.docx (+ .pdf neu co LibreOffice)

Dung lai cac muc cua bao cao tong (build_docs.py) va cac muc dinh luong (report_analysis_sections.py);
moi so lieu doc tu data/, db/ va results/ (chay scripts/analyze_features.py truoc).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_docs import FIG, cover, load_context, sec_dataset, sec_features, to_pdf  # noqa: E402
from docx_helpers import Report  # noqa: E402
from report_analysis_sections import SET_VI, sec_example, sec_info_value  # noqa: E402
from treedb import config  # noqa: E402

OUT_DOCX = config.ROOT / "docs" / "BaoCao1_Nhom27_NoiDung1-2.docx"
SEG_REF = "mục 3.6"


def toc(r: Report):
    r.h(1, "MỤC LỤC")
    for it in [
        "1. Đề bài và phạm vi Báo cáo 1",
        "2. Xây dựng bộ dữ liệu ảnh cây thân gỗ (Nội dung 1)",
        "    2.1. Tiêu chí của bộ dữ liệu",
        "    2.2. Nguồn và quy trình sưu tầm",
        "    2.3. Thống kê bộ dữ liệu",
        "    2.4. Đặc điểm giống nhau và khác nhau của các cây trong ảnh",
        "    2.5. Định lượng điểm giống và khác nhau theo chi cây",
        "3. Các bộ đặc trưng nhận diện cây (Nội dung 2)",
        "    3.1. Định hướng thiết kế",
        "    3.2 – 3.5. Bốn bộ đặc trưng: màu sắc, hình dạng, kết cấu/bố cục, embedding sâu",
        "    3.6. Tách vùng cây khỏi nền – tiền đề của các đặc trưng vật thể",
        "    3.7. Đánh giá định lượng giá trị thông tin của các đặc trưng",
        "    3.8. Ví dụ minh hoạ: đặc trưng tìm sự tương đồng và sự khác biệt",
        "4. Kết luận và kế hoạch cho Báo cáo 2",
    ]:
        r.p(it)
    r.page_break()


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
        "kho mã của nhóm.")


def build(ctx: dict) -> Path:
    r = Report()
    cover(r, ctx, title="BÁO CÁO 1 – NỘI DUNG 1 VÀ 2", note="Hạn nộp: 22/10/2026 (10% điểm học phần)")
    toc(r)
    sec_scope(r, ctx)
    sec_dataset(r, ctx, seg_ref=SEG_REF)
    sec_features(r, ctx, seg_ref=SEG_REF, with_analysis=False)
    sec_segmentation(r, ctx)
    sec_info_value(r, ctx["feature_analysis"], "3.7")
    sec_example(r, ctx["feature_analysis"], "3.8")
    sec_conclusion(r, ctx)
    return r.save(OUT_DOCX)


if __name__ == "__main__":
    ctx = load_context()
    if not ctx.get("feature_analysis"):
        sys.exit("Thieu results/feature_analysis.json - chay: python scripts/analyze_features.py")
    out = build(ctx)
    print(f"[docx] {out} ({out.stat().st_size / 1024:.0f} KB)")
    pdf = to_pdf(out)
    if pdf:
        print(f"[pdf] {pdf} ({pdf.stat().st_size / 1024:.0f} KB)")
