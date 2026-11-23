"""Sinh Bao cao 2 (20%, han nop 03/12/2026): noi dung 3 va 4 cua de bai nhom 27.

    python scripts/build_report2.py   -> docs/BaoCao2_Nhom27_NoiDung3-4.docx (+ .pdf neu co LibreOffice)

Cung khuon voi Bao cao 1 (build_report1.py): bia PTIT, phan cong, muc luc co so trang.
Moi so lieu doc tu data/, db/ va results/; vi du tinh tay o muc 4.2 duoc tinh lai tu CSDL va doi chieu.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_report1 as r1  # noqa: E402
from build_docs import GROUP, cover, f3, load_context  # noqa: E402
from docx_helpers import Report  # noqa: E402
from report2_content3 import sec_content3  # noqa: E402
from report2_content4 import sec_content4  # noqa: E402
from report2_theory import REFERENCES, sec_theory  # noqa: E402
from treedb import config  # noqa: E402

OUT_DOCX = config.ROOT / "docs" / "BaoCao2_Nhom27_NoiDung3-4.docx"

DANG, TU, VANH = (m[0] for m in GROUP["members"])
# (cong viec, sinh vien thuc hien, thoi han) - sua o day neu phan cong thuc te khac
TASKS = [
    ("Đọc lại slide bài 4–8 và 11, đối chiếu kiến trúc, mô hình dữ liệu, siêu dữ liệu với thiết kế hệ thống", VANH,
     "27/10/2026"),
    ("Cài đặt các thuật toán trích rút bộ đặc trưng màu sắc và kết cấu, song song hoá trích rút", DANG,
     "03/11/2026"),
    ("Cài đặt trích rút bộ đặc trưng hình dạng và embedding MobileNetV2 (ONNX)", TU, "03/11/2026"),
    ("Thiết kế lược đồ CSDL SQLite theo mô hình lai, cài đặt các thao tác ghi/đọc vector và siêu dữ liệu", DANG,
     "08/11/2026"),
    ("Chuẩn hoá, kết hợp các bộ đặc trưng; dựng cây k-d, cây R, cây tứ phân và phân cụm K-means", VANH,
     "13/11/2026"),
    ("Xây dựng module truy vấn top-5, lọc siêu dữ liệu, phản hồi liên quan và giao diện demo", TU, "18/11/2026"),
    ("Vẽ sơ đồ khối, sinh ví dụ truy vấn và tính tay độ tương đồng cho phần kết quả trung gian", VANH,
     "22/11/2026"),
    ("Đánh giá leave-one-out (P@5, mAP@5), so sánh bộ đặc trưng và chỉ mục, viết nhận xét", DANG, "26/11/2026"),
    ("Tổng hợp nội dung, viết Lời nói đầu, Kết luận và hoàn thiện định dạng báo cáo", f"{DANG}, {TU}, {VANH}",
     "01/12/2026"),
]


def sec_preface(r: Report):
    r.h(1, "Lời nói đầu")
    r.p("Ở Báo cáo 1, nhóm đã xây dựng bộ dữ liệu ảnh cây thân gỗ trưởng thành và thiết kế các bộ đặc trưng nhận "
        "diện cây. Một bộ đặc trưng chỉ có giá trị khi nó được trích rút tự động, được lưu trữ và quản trị trong một "
        "hệ CSDL, và được dùng để trả lời truy vấn theo nội dung. Báo cáo 2 trình bày đúng phần việc đó.")
    r.p("Báo cáo gồm hai nội dung của đề tài: cài đặt các thuật toán trích rút đặc trưng và xây dựng hệ CSDL quản "
        "trị đặc trưng của toàn bộ ảnh (nội dung 3); xây dựng hệ thống tìm kiếm trả về 5 ảnh giống nhất với một ảnh "
        "cây mới, kèm sơ đồ khối, kết quả trung gian và đánh giá (nội dung 4). Trước hai nội dung này, nhóm tóm tắt "
        "các kiến thức của học phần được áp dụng, để thấy rõ mỗi lựa chọn thiết kế xuất phát từ đâu.")
    r.p("Nhóm xin cảm ơn thầy Nguyễn Đình Hóa đã hướng dẫn trong quá trình thực hiện. Do thời gian và kiến thức còn "
        "hạn chế, báo cáo khó tránh khỏi thiếu sót; nhóm mong nhận được góp ý của thầy để hoàn thiện phần demo.")
    r.p("Chúng em xin chân thành cảm ơn thầy!")


def sec_scope(r: Report, ctx: dict):
    fs = ctx["feature_sets"]
    r.h(1, "1. Đề bài và phạm vi Báo cáo 2")
    r.p("Đề bài nhóm 27: Xây dựng hệ CSDL lưu trữ và tìm kiếm ảnh cây. Báo cáo 1 (10%) đã trình bày nội dung 1 và "
        "2; Báo cáo 2 (20%, hạn 03/12/2026) trình bày nội dung 3 và 4; báo cáo demo (60%) vào ngày thi gồm tất cả "
        "các nội dung. Yêu cầu của hai nội dung trong báo cáo này:")
    r.p("3. Triển khai các thuật toán/công cụ trích rút các đặc trưng đã được xác định trong bước 2. Hãy xây dựng "
        "hệ CSDL để quản trị các đặc trưng của tất cả ảnh đã có.", style=None)
    r.p("4. Triển khai hệ thống tìm kiếm ảnh tương đồng với đầu vào là một ảnh mới về một cây nào đó, đầu ra là 5 "
        "ảnh giống nhất, xếp thứ tự giảm dần về độ tương đồng nội dung với ảnh đầu vào.")
    r.bullets([
        "a. Trình bày sơ đồ khối của hệ thống, chức năng của từng khối/module của hệ thống cùng các dữ liệu vào/ra "
        "cho từng khối. Mô tả quy trình thực hiện yêu cầu của đề bài.",
        "b. Hãy minh họa các kết quả trung gian (các giá trị đặc trưng, tính toán độ tương đồng) của quá trình truy "
        "vấn ảnh.",
        "c. Đánh giá các kết quả đã đạt được.",
    ])
    r.p(f"Kết quả của Báo cáo 1 được dùng làm đầu vào: {ctx['db']['images']} ảnh {config.IMAGE_SIZE}×"
        f"{config.IMAGE_SIZE} đã tách nền (cây trên nền trắng) và {len(ctx['queries'])} ảnh truy vấn thử không nằm "
        f"trong CSDL; bốn bộ đặc trưng " + ", ".join(f"{f['set_name']} ({f['dim']} chiều, {f['metric']})" for f in fs)
        + ". Bảng dưới đây chỉ ra mỗi yêu cầu được trình bày ở mục nào.")
    r.table(["Yêu cầu", "Mục", "Kết quả chính"], [
        ["3 – trích rút đặc trưng", "3.1, 3.2", "Thuật toán và mã nguồn của 4 bộ đặc trưng"],
        ["3 – hệ CSDL quản trị đặc trưng", "3.3 – 3.5", "Lược đồ SQLite 7 bảng, chuẩn hoá, 4 cấu trúc chỉ mục"],
        ["4a – sơ đồ khối, quy trình", "4.1", "Sơ đồ khối 2 giai đoạn, bảng vào/ra của 11 khối"],
        ["4b – kết quả trung gian", "4.2", "Giá trị đặc trưng, tính độ tương đồng từng bước, 4 ví dụ"],
        ["4c – đánh giá", "4.3 – 4.5", "P@5, mAP@5 trên " + str(ctx["eval"]["n_queries"]) + " truy vấn, so sánh chỉ mục"],
    ], widths=[4.6, 2.4, 9.0], caption="Ánh xạ yêu cầu đề bài với các mục của báo cáo", size=10.5)


def sec_conclusion(r: Report, ctx: dict):
    ev = ctx["eval"]
    comb, be = ev["feature_sets"]["combined(default)"], ev["backends"]
    r.h(1, "5. Kết luận và kế hoạch demo")
    r.p(f"Nội dung 3: bốn bộ đặc trưng ({sum(f['dim'] for f in ctx['feature_sets'])} chiều) được trích rút tự động "
        f"cho toàn bộ {ctx['db']['images']} ảnh và quản trị trong CSDL SQLite theo mô hình lai, gồm siêu dữ liệu quan "
        "hệ, vector đặc trưng, thống kê chuẩn hoá, bốn cấu trúc chỉ mục đa chiều và nhật ký truy vấn.")
    r.p(f"Nội dung 4: hệ thống nhận một ảnh cây mới và trả về {ev['k']} ảnh giống nhất trong khoảng 1 giây. Mọi giá "
        "trị trung gian đều được lưu và đã được kiểm tra lại bằng tính tay. Trên "
        f"{ev['n_queries']} truy vấn leave-one-out, cấu hình kết hợp đạt P@{ev['k']} = {f3(comb['p_at_k'])} và "
        f"mAP@{ev['k']} = {f3(comb['ap_at_k'])}, gấp {comb['p_at_k'] / ev['random_baseline']['p_at_k']:.1f} lần "
        f"ngẫu nhiên. Cây k-d giữ {100 * be['kdtree']['recall_vs_linear']:.0f}% kết quả của quét tuyến tính với "
        f"thời gian xếp hạng giảm khoảng {be['linear']['ms_per_query'] / be['kdtree']['ms_per_query']:.0f} lần.")
    r.p("Kế hoạch cho buổi demo: chạy giao diện web Streamlit (tải ảnh lên, chọn chỉ mục, trọng số, bộ lọc, phản "
        "hồi liên quan) và dòng lệnh scripts/query.py; chuẩn bị thêm ảnh cây chụp thực tế để thử với ảnh ngoài "
        "nguồn Wikimedia Commons.")


def sec_references(r: Report):
    r.h(1, "Tài liệu tham khảo")
    r.numbered(REFERENCES)


def build(ctx: dict, toc_entries=None, pages=None):
    r = Report()
    r.page_numbers()
    cover(r, ctx, title="BÁO CÁO", note="Báo cáo lần 2 – Nội dung 3 và 4")
    r1.sec_assignment(r, TASKS)
    if toc_entries is not None:
        r1.toc(r, toc_entries, pages)
    sec_preface(r)
    sec_scope(r, ctx)
    sec_theory(r)
    sec_content3(r, ctx)
    sec_content4(r, ctx)
    sec_conclusion(r, ctx)
    sec_references(r)
    return r.save(OUT_DOCX), r.headings


if __name__ == "__main__":
    out, pdf = r1.build_with_toc(load_context(), build)  # muc luc co so trang, cung quy trinh Bao cao 1
    print(f"[docx] {out} ({out.stat().st_size / 1024:.0f} KB)")
    if pdf:
        print(f"[pdf] {pdf} ({pdf.stat().st_size / 1024:.0f} KB)")
