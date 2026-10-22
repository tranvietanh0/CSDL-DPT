"""Sinh Bao cao 1 (10%, han nop 22/10/2026): noi dung 1 va 2 cua de bai nhom 27, theo mau trinh bay cua hoc phan
(bia co khung + logo, muc luc tu dong, loi noi dau, Phan I - III, ket luan; cong thuc la equation that cua Word).

    python scripts/analyze_features.py   # results/feature_analysis.json (+ hinh 15, 16)
    python scripts/analyze_dataset.py    # results/dataset_quality.json (+ hinh 17, 18, 19), can onnxruntime
    python scripts/build_report1.py      # docs/BaoCao1_Nhom27_NoiDung1-2.docx (+ .pdf neu co MS Word / LibreOffice)

Moi so lieu doc tu data/, db/ va results/. Co MS Word thi muc luc va so trang duoc cap nhat san khi xuat.
"""
from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_docs import FIG, GROUP, load_context, to_pdf  # noqa: E402
from docx_helpers import PtitReport  # noqa: E402
from report_analysis_sections import SET_VI, _f, sec_example, sec_info_value, sec_quant_dataset  # noqa: E402
from treedb import config  # noqa: E402

OUT_DOCX = config.ROOT / "docs" / "BaoCao1_Nhom27_NoiDung1-2.docx"
ASSETS = config.ROOT / "docs" / "assets"
FRAME_EMU = (6361430, 8686800)      # khung bia ~ 17,7 x 24,1 cm, can giua vung le

GENUS_VI = {"Populus": "dương", "Quercus": "sồi", "Pinus": "thông", "Juglans": "óc chó", "Betula": "bạch dương",
            "Alnus": "tống quán sủi", "Tilia": "đoan", "Vachellia": "keo", "Acacia": "keo", "Salix": "liễu",
            "Adansonia": "bao báp", "Fagus": "dẻ gai", "Prunus": "mận, anh đào", "Ficus": "sung, đa",
            "Cedrus": "tuyết tùng", "Picea": "vân sam", "Olea": "ô liu", "Fraxinus": "tần bì", "Castanea": "dẻ",
            "Aesculus": "dẻ ngựa", "Acer": "phong", "Acacia/Vachellia": "keo"}
CONIFER = {"Pinus", "Picea", "Abies", "Larix", "Cedrus", "Juniperus", "Taxus", "Araucaria", "Cupressus", "Thuja"}

# Bang doi sanh hinh thai (quan sat bang mat + dac diem thuc vat hoc); so do lay tu results/dataset_quality.json
MORPHOLOGY = [
    ("Populus", "lá rộng", "Tán hình trứng đến bầu dục rộng, thường có vài thân hoặc cành lớn vươn chéo lên; thân "
                           "thẳng, vỏ xám trắng (dương trắng)",
     "Lá đơn hình tam giác – tim, mặt dưới trắng bạc (P. alba); vàng mùa thu, trụi lá mùa đông",
     "Tán ánh xanh bạc, phân cành thấp; hay gặp ở bãi ven sông, đồng ruộng"),
    ("Quercus", "lá rộng", "Tán tròn rộng, xoè ngang; cành lớn khúc khuỷu; thân to, ngắn, vỏ nứt dọc sâu",
     "Lá xẻ thuỳ tròn, xanh đậm; mùa thu nâu vàng, lá khô có thể còn trên cây",
     "Tán rộng hơn cao và đặc; cành chính vươn ngang"),
    ("Pinus", "lá kim", "Tán không đối xứng, ở cây già dẹt hoặc thành từng tầng; thân lộ dài, phần trên vỏ màu nâu cam",
     "Lá kim xanh sẫm quanh năm", "Tán sẫm màu, ít đổi theo mùa; thân dài trần, tán lệch"),
    ("Juglans", "lá rộng", "Tán tròn rộng, hơi thưa; thân ngắn, vỏ xám bạc",
     "Lá kép lông chim, xanh vàng nhạt; trụi lá mùa đông", "Tán tròn rộng, cành to vươn ngang"),
    ("Betula", "lá rộng", "Tán hẹp và thưa; cành nhánh mảnh, rủ xuống; thân mảnh, vỏ trắng có vết đen",
     "Lá nhỏ hình tam giác; mùa thu vàng tươi", "Thân trắng nổi bật, tán thưa nhiều khe hở"),
    ("Alnus", "lá rộng", "Tán hình trứng đến hình nón khi non, tròn khi già; thường nhiều thân; mọc ven nước",
     "Lá tròn xanh đậm, rụng khi còn xanh; mùa đông còn quả nón nhỏ màu đen", "Tán thưa, nhiều thân, bối cảnh ven nước"),
    ("Tilia", "lá rộng", "Tán dày đặc hình trứng hoặc vòm, đường bao đều; thân thẳng",
     "Lá hình tim, xanh tươi; mùa thu vàng", "Tán đặc, đường bao tròn đều, ít khe hở"),
    ("Acacia/Vachellia", "lá rộng", "Tán dẹt hình ô (dù), rộng hơn cao; thân phân nhánh thấp",
     "Lá kép nhỏ, có gai; xanh xám", "Mặt trên tán phẳng; nền savan khô vàng"),
]


def pc(x: float, n: int = 1) -> str:
    """Phan tram dang Viet Nam (dau phay thap phan)."""
    return f"{100 * x:.{n}f}".replace(".", ",") + "%"


# =============================================================================================
# Bia, muc luc, loi noi dau
# =============================================================================================
def cover(r: PtitReport):
    r.background_image(ASSETS / "khung_bia.jpg", *FRAME_EMU)
    r.line("HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG", 15, before=34, after=2)
    r.line("KHOA CÔNG NGHỆ THÔNG TIN", 14, after=0)
    r.picture_center(ASSETS / "logo_ptit.png", 3.4, space_before=22, space_after=22)
    r.line("BÁO CÁO", 26, after=2)
    r.line("HỆ CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN", 20, after=20)
    r.line("Đề tài: Xây dựng hệ CSDL lưu trữ và", 18, after=0)
    r.line("tìm kiếm ảnh cây thân gỗ dựa trên nội dung", 18, after=16)
    r.line("Báo cáo lần 1 – Nội dung 1 và 2", 14, after=26)
    members = GROUP["members"]
    rows = [("Giảng viên hướng dẫn", ": TS. Nguyễn Đình Hóa"), ("Lớp", f": {GROUP['class']}"),
            ("Nhóm", f": {GROUP['name'].replace('Nhóm ', '')}"),
            ("Họ tên và mã sinh viên", f": {members[0][0]} – {members[0][1]}")]
    rows += [("", f": {n} – {m}") for n, m in members[1:]]
    r.info_block(rows)
    r.line("Hà Nội, tháng 10/2026", 14, bold=False, italic=True, before=40)
    r.page_break()


def preface(r: PtitReport, ctx: dict):
    n_db = ctx["db"]["images"]
    r.h(1, "Lời nói đầu")
    r.p("Cùng với sự phát triển của thiết bị ghi hình và các kho dữ liệu mở, khối lượng ảnh số tăng rất nhanh và trở "
        "thành loại dữ liệu đa phương tiện phổ biến. Khác với hệ quản trị cơ sở dữ liệu truyền thống vốn tìm kiếm bằng "
        "phép so khớp chính xác trên dữ liệu có cấu trúc, cơ sở dữ liệu đa phương tiện phải biểu diễn nội dung của ảnh "
        "bằng các vector đặc trưng và tìm kiếm theo độ tương đồng. Truy vấn theo từ khoá phụ thuộc vào việc gán nhãn "
        "thủ công, vừa tốn công vừa chủ quan; vì vậy truy vấn ảnh dựa trên nội dung (Content-Based Image Retrieval – "
        "CBIR), trong đó người dùng đưa vào một ảnh mẫu (query by example) và hệ thống trả về các ảnh giống nhất, ngày "
        "càng được ứng dụng rộng rãi.")
    r.p("Cây thân gỗ là đối tượng quen thuộc nhưng không dễ nhận diện qua ảnh. Dáng tán, tỉ lệ giữa thân và tán, màu "
        "và kết cấu của tán lá là những dấu hiệu giúp phân biệt các loài; tuy nhiên ảnh cây được chụp ngoài trời nên "
        "nền ảnh phức tạp (bầu trời, mặt đất, công trình, cây khác), ánh sáng và mùa thay đổi mạnh, và cùng một loài có "
        "thể có nhiều dáng khác nhau tuỳ tuổi cây và môi trường sống. Do đó cần tách được cây khỏi nền và kết hợp nhiều "
        "loại đặc trưng bổ sung cho nhau.")
    r.p("Xuất phát từ yêu cầu đó, nhóm thực hiện đề tài “Xây dựng hệ CSDL lưu trữ và tìm kiếm ảnh cây” trong khuôn khổ "
        "học phần Hệ cơ sở dữ liệu đa phương tiện. Báo cáo lần 1 trình bày hai nội dung đầu của đề tài:")
    r.numbered([
        ("Xây dựng bộ dữ liệu ảnh cây thân gỗ: ", f"sưu tầm, sàng lọc và chuẩn hoá {n_db} ảnh (yêu cầu tối thiểu "
         f"{config.MIN_IMAGES}), mỗi ảnh một cây trưởng thành nằm giữa khung hình, cùng kích thước "
         f"{config.IMAGE_SIZE}×{config.IMAGE_SIZE}; thống nhất quy ước định danh, cấu trúc lưu trữ và siêu dữ liệu; "
         "thống kê, đánh giá tính cân bằng và các hạn chế của bộ dữ liệu; phân tích điểm giống và khác nhau giữa "
         "các cây."),
        ("Xây dựng các bộ đặc trưng nhận diện cây: ", "bốn bộ đặc trưng màu sắc, hình dạng, kết cấu/bố cục và "
         "embedding sâu; trình bày cơ sở toán học, ý nghĩa của từng đặc trưng và đo giá trị thông tin của chúng bằng "
         "số liệu trên chính bộ dữ liệu."),
    ])
    r.p("Báo cáo gồm ba phần chính: Phần I trình bày quá trình xây dựng bộ dữ liệu; Phần II phân tích đặc điểm giống và "
        "khác nhau giữa các cây; Phần III trình bày các bộ đặc trưng và đánh giá giá trị thông tin của chúng; cuối cùng "
        "là kết luận và kế hoạch cho Báo cáo 2 (nội dung 3 và 4).")
    r.p("Dù đã cố gắng, báo cáo khó tránh khỏi những thiếu sót. Nhóm rất mong nhận được góp ý của thầy để tiếp tục "
        "hoàn thiện hệ thống ở các giai đoạn sau.")
    r.p("Chúng em xin chân thành cảm ơn thầy!")


# =============================================================================================
# PHAN I. Bo du lieu
# =============================================================================================
def part1(r: PtitReport, ctx: dict):
    r.h(1, "PHẦN I. XÂY DỰNG BỘ DỮ LIỆU ẢNH CÂY THÂN GỖ")
    sec_overview(r, ctx)
    sec_naming(r, ctx)
    sec_digitization(r, ctx)
    sec_statistics(r, ctx)
    sec_balance_limits(r, ctx)


def sec_overview(r: PtitReport, ctx: dict):
    n_db, q = ctx["db"]["images"], ctx["quality"]
    r.h(2, "1.1. Tổng quan và nguồn dữ liệu")
    r.h(3, "1.1.1. Mục tiêu, quy mô và tiêu chí của bộ dữ liệu")
    r.pm(f"Mục tiêu của nội dung 1 là xây dựng một bộ ảnh đủ lớn và đồng nhất để làm cơ sở dữ liệu cho hệ thống tìm "
         f"kiếm ảnh cây theo nội dung. Bộ dữ liệu có {n_db} ảnh trong CSDL và {len(ctx['queries'])} ảnh truy vấn thử "
         f"tách riêng, vượt chỉ tiêu tối thiểu {config.MIN_IMAGES} ảnh của đề bài. Gọi $K$ là số chi (genus) xác định "
         "được, $n_k$ là số ảnh của chi thứ $k$ và $n_0$ là số ảnh chưa xác định được chi, tổng số ảnh của CSDL là:")
    r.eq(rf"N=n_0+\sum_{{k=1}}^{{K}}{{n_k}}={n_db - q['labeled']}+{q['labeled']}={n_db}\geq {config.MIN_IMAGES},"
         rf"\qquad K={q['n_genera']}")
    r.p("Các tiêu chí đặt ra cho từng ảnh, cụ thể hoá từ đề bài:")
    r.bullets([
        ("Đối tượng: ", "cây thân gỗ trưởng thành (cây gỗ lớn lá rộng hoặc lá kim); loại cây cọ/dừa (thân không hoá gỗ "
                        "thực sự), cây bụi, cây non, tranh vẽ và ảnh nghệ thuật."),
        ("Bố cục: ", "mỗi ảnh có đúng một cây là chủ thể, thấy cả thân và tán, cây nằm gần giữa ảnh và đủ lớn (cao từ "
                     "khoảng 1/4 chiều cao ảnh trở lên)."),
        ("Góc chụp: ", "chụp ngang tầm mắt từ phía bên; không chụp từ trên cao, không ngước từ gốc, không cận cảnh "
                       "vỏ/lá."),
        ("Lưu trữ: ", f"cùng kích thước {config.IMAGE_SIZE}×{config.IMAGE_SIZE}, định dạng JPEG, kèm siêu dữ liệu nguồn "
                      "gốc (mục 1.2)."),
    ])
    r.p("Mức độ đáp ứng các tiêu chí bố cục được đo lại bằng số liệu ở mục 1.3.2 và mục 1.5.2.")

    r.h(3, "1.1.2. Nguồn dữ liệu và quy trình thu thập, sàng lọc")
    r.p("Ảnh được sưu tầm tự động từ kho ảnh mở Wikimedia Commons thông qua MediaWiki API (module "
        "src/treedb/collect.py). Điểm xuất phát là các category về “cây mọc đơn lẻ” (Solitary trees) vì đây là những "
        "bức ảnh có đúng bố cục một cây hoàn chỉnh giữa khung hình: các category theo loài (Solitary Quercus robur, "
        "Solitary Betula pendula, Solitary Pinus sylvestris, Solitary Juglans regia, Solitary Alnus glutinosa, Solitary "
        "Populus alba…), category ảnh chất lượng (Quality images of solitary trees), category tổng (Solitary trees) và "
        "các category theo quốc gia (Solitary trees by country, duyệt sâu 2 mức). Với mỗi file, hệ thống lấy kích thước "
        "gốc, kiểu MIME, tác giả, giấy phép và danh sách category; chi (genus) của cây được nhận diện bằng biểu thức "
        "chính quy trên tên category/tiêu đề dựa trên danh mục hơn 80 chi cây thân gỗ phổ biến.")
    rej = ctx.get("rejects") or {}
    rej_vi = {"aspect": "tỉ lệ khung", "badcat": "category không phù hợp", "small": "ảnh nhỏ",
              "badword": "từ khoá loại trừ", "mime": "định dạng", "palm": "cây cọ"}
    r.numbered([
        f"Duyệt category → {ctx['n_candidates'] or '…'} file ứng viên.",
        "Lọc theo quy tắc: MIME là JPEG/PNG; cạnh ngắn ≥ 640 px; tỉ lệ rộng/cao trong [0,65; 1,6]; loại các tiêu "
        "đề/category chứa từ khoá không phù hợp (map, diagram, leaf, bark, trunk, detail, flower, aerial, panorama, "
        "forest, avenue, bonsai, painting, palm…); loại cây cọ."
        + (f" Kết quả: chấp nhận {ctx['n_accept']}/{ctx['n_total']} file; số file bị loại theo lý do: "
           + "; ".join(f"{rej_vi.get(k, k)} {v}" for k, v in rej.items()) + "." if rej else ""),
        "Sắp xếp ưu tiên nguồn “thuần” (category cây đơn lẻ theo loài → ảnh chất lượng → cây đơn lẻ chung/theo quốc "
        f"gia) và tải tối đa 1 000 ảnh ở độ rộng 1 024 px → {ctx['n_raw'] or '…'} ảnh thô.",
        f"Chuẩn hoá {config.IMAGE_SIZE}×{config.IMAGE_SIZE} (mục 1.3); khử trùng lặp bằng perceptual hash (DCT 8×8, "
        "khoảng cách Hamming ≤ 6).",
        f"Kiểm duyệt thủ công bằng mắt toàn bộ ảnh trên các bảng ghép 30 ảnh (contact sheet): loại "
        f"{len(ctx['manual_excluded'])} ảnh không đạt tiêu chí bố cục (tán bị cắt, cây quá nhỏ/xa, nhiều cây, cận "
        "cảnh, chủ thể là công trình/người, rừng ngập mặn…). Danh sách loại được lưu theo tiêu đề nguồn trong "
        "data/manual_exclude.csv để quy trình có thể tái lập.",
        f"Tách {len(ctx['queries'])} ảnh có nhãn chi làm ảnh truy vấn thử (data/queries, không đưa vào CSDL).",
        "Cổng kiểm soát chất lượng tự động sau phân đoạn (mục 3.2): chỉ nhận ảnh phân đoạn tìm thấy cây, có tỉ lệ vùng "
        "cây trong [4%; 90%] và tâm vùng cây lệch tâm ảnh không quá 30% mỗi trục → loại thêm "
        f"{len(ctx['gate_excluded'])} ảnh.",
        f"Kết quả: {n_db} ảnh trong CSDL (đạt yêu cầu ≥ {config.MIN_IMAGES} ảnh).",
    ])
    r.figure(FIG / "01_dataset_samples.png", "Một số ảnh trong bộ dữ liệu sau chuẩn hoá (nhãn chi nếu có)", 16)


def sec_naming(r: PtitReport, ctx: dict):
    imgs = ctx["images"]
    titles = [i["source_title"] or "" for i in imgs]
    nonascii = [t for t in titles if any(ord(c) > 127 for c in t)]
    spaced = [t for t in titles if " " in t]
    lens = [len(t) for t in titles]
    longest = max(titles, key=len)
    r.h(2, "1.2. Quy ước đặt tên, cấu trúc lưu trữ và siêu dữ liệu")
    r.h(3, "1.2.1. Phân tích tên file gốc và lý do chuẩn hoá")
    r.p("Tên file trên Wikimedia Commons do người đóng góp tự đặt nên rất không đồng nhất, ví dụ "
        f"“{nonascii[0] if nonascii else titles[0]}” hay “{longest}”. Giữ nguyên các tên này làm định danh trong "
        "CSDL gây ra nhiều vấn đề:")
    r.bullets([
        ("Độ dài không cố định ", f"(từ {min(lens)} đến {max(lens)} ký tự): tốn bộ nhớ lưu chuỗi, làm chậm việc đánh "
                                  "chỉ mục và so khớp chuỗi."),
        ("Chứa dấu cách, ký tự Unicode và ký tự đặc biệt ", f"({len(spaced)} tên có dấu cách, {len(nonascii)} tên có "
                                                           "ký tự ngoài bảng mã ASCII như chữ có dấu, chữ Kirin): dễ gây "
                                                           "lỗi đường dẫn khi chuyển dữ liệu giữa các hệ điều hành và lỗi "
                                                           "cú pháp khi ghép vào câu lệnh SQL (dấu nháy, dấu phẩy)."),
        ("Không mang thông tin về vị trí của ảnh ", "trong bộ dữ liệu, không thể dùng làm khoá sắp xếp."),
    ])
    r.p("Do đó mỗi ảnh được gán một mã định danh có độ dài cố định, không chứa ký tự đặc biệt; tên gốc vẫn được giữ "
        "trong siêu dữ liệu để truy xuất nguồn.")

    r.h(3, "1.2.2. Quy ước mã định danh và quy tắc đặt tên chuẩn hoá")
    n_files = len(ctx["manifest"])
    r.p("Mã định danh image_id là số nguyên tăng dần, đóng vai trò khoá chính (primary key) trong CSDL; tên file chuẩn "
        "hoá được sinh từ mã này theo các quy tắc ở bảng dưới.")
    r.table(["Loại tệp", "Quy tắc tên", "Ví dụ", "Ghi chú"], [
        ["Ảnh trong CSDL", "tree_NNNN.jpg, NNNN = image_id (4 chữ số)", f"tree_0001.jpg → tree_{n_files:04d}.jpg",
         f"data/images/; {len(ctx['gate_excluded'])} mã của ảnh bị cổng chất lượng loại được bỏ trống"],
        ["Ảnh truy vấn thử", "query_NN.jpg (2 chữ số)", f"query_01.jpg → query_{len(ctx['queries']):02d}.jpg",
         "data/queries/, không nằm trong CSDL"],
        ["Ảnh thô", "NNNNN.jpg, NNNNN = raw_id (5 chữ số)", f"00001.jpg → {ctx['n_raw'] or 0:05d}.jpg",
         "data/raw/, liên kết với ảnh chuẩn hoá qua cột raw_id"],
        ["Tên gốc", "giữ nguyên trong cột source_title", imgs[0]["source_title"],
         "truy xuất nguồn; khoá của danh sách kiểm duyệt thủ công"],
    ], widths=[2.8, 4.4, 4.4, 4.4], caption="Quy ước mã định danh và tên file", size=10)
    r.p("Ngoài ra mỗi ảnh lưu mã băm SHA-1 của nội dung file để kiểm tra tính toàn vẹn và phát hiện ảnh trùng lặp "
        "hoàn toàn.")

    r.h(3, "1.2.3. Cấu trúc thư mục lưu trữ vật lý")
    r.p("Ảnh thô, ảnh đã chuẩn hoá, siêu dữ liệu và CSDL đặc trưng được tổ chức trong kho mã của nhóm như sau:")
    r.code(
        "CSDL-DPT/\n"
        "├── data/\n"
        f"│   ├── raw/                 ảnh gốc tải về ({ctx['n_raw']} ảnh, không đưa lên kho mã)\n"
        f"│   ├── images/              {n_files} ảnh chuẩn hoá 512×512 (tree_0001 → tree_{n_files:04d})\n"
        f"│   ├── queries/             {len(ctx['queries'])} ảnh truy vấn thử (query_01 → query_"
        f"{len(ctx['queries']):02d})\n"
        "│   ├── raw_manifest.csv     siêu dữ liệu ảnh thô\n"
        "│   ├── manifest.csv         siêu dữ liệu ảnh đã chuẩn hoá\n"
        "│   ├── queries.csv          siêu dữ liệu ảnh truy vấn\n"
        f"│   └── manual_exclude.csv   {len(ctx['manual_excluded'])} ảnh loại khi kiểm duyệt thủ công\n"
        "├── db/\n"
        f"│   └── tree_features.sqlite CSDL: siêu dữ liệu + vector đặc trưng ({ctx['db']['images']} ảnh)\n"
        "├── results/\n"
        f"│   └── excluded_images.csv  {len(ctx['gate_excluded'])} ảnh bị cổng chất lượng loại\n"
        "└── src/treedb/              mã nguồn thu thập, chuẩn hoá, phân đoạn, trích rút đặc trưng", size=9.5)
    r.p(f"Thư mục data/images giữ cả {n_files} ảnh sau chuẩn hoá; {ctx['db']['images']} ảnh qua được cổng kiểm soát "
        "chất lượng được nạp vào CSDL, còn danh sách và lý do loại các ảnh còn lại được ghi trong "
        "results/excluded_images.csv.")

    r.h(3, "1.2.4. Quản lý siêu dữ liệu và mô hình hoá đối tượng ảnh")
    r.p("Theo mô hình dữ liệu đa phương tiện, mỗi ảnh được quản lý như một đối tượng gồm ba thành phần liên kết với nhau "
        "qua khoá image_id:")
    r.numbered([
        ("Thành phần dữ liệu thô: ", "đường dẫn file, định dạng, kích thước, dung lượng và mã băm."),
        ("Thành phần siêu dữ liệu mô tả: ", "nguồn, tác giả, giấy phép, kích thước gốc, chi và nhóm hình thái. Chi và "
                                            "nhóm hình thái là nhãn dùng để đánh giá ở các bước sau, không dùng khi "
                                            "tìm kiếm."),
        ("Thành phần vector đặc trưng nội dung: ", "bốn vector màu sắc, hình dạng, kết cấu và embedding, lưu trong bảng "
                                                   "features (mỗi dòng là một cặp ảnh – bộ đặc trưng, vector dạng BLOB "
                                                   "float32)."),
    ])
    im = imgs[0]
    ex = lambda k: str(im.get(k, ""))  # noqa: E731
    r.p("Siêu dữ liệu được lưu đồng thời trong tệp data/manifest.csv và bảng images của CSDL SQLite với lược đồ ở bảng "
        "dưới (ví dụ là ảnh đầu tiên của CSDL); các bảng còn lại của CSDL sẽ được trình bày ở Báo cáo 2.")
    r.table(["Trường", "Kiểu", "Ý nghĩa", f"Ví dụ ({im['filename']})"], [
        ["image_id", "INTEGER (PK)", "Mã định danh, khoá chính", ex("image_id")],
        ["filename", "TEXT (UNIQUE)", "Tên file chuẩn hoá", ex("filename")],
        ["path", "TEXT", "Đường dẫn tới file ảnh", f"data/images/{im['filename']}"],
        ["width, height", "INTEGER", "Kích thước sau chuẩn hoá", f"{im['width']}, {im['height']}"],
        ["format", "TEXT", "Định dạng lưu trữ", ex("format")],
        ["file_size", "INTEGER", "Dung lượng file (byte)", f"{int(im['file_size']):,}".replace(",", ".")],
        ["sha1", "TEXT", "Mã băm SHA-1 của nội dung", ex("sha1")[:12] + "…"],
        ["source_title", "TEXT", "Tên file gốc trên Commons", ex("source_title")],
        ["source_url", "TEXT", "Trang nguồn", "commons.wikimedia.org/wiki/…"],
        ["author", "TEXT", "Tác giả", ex("author")],
        ["license", "TEXT", "Giấy phép", ex("license")],
        ["genus", "TEXT", "Chi cây (nếu xác định được)", ex("genus")],
        ["group_label", "TEXT", "Nhóm hình thái (lá rộng / lá kim)", ex("group_label")],
        ["orig_width, orig_height", "INTEGER", "Kích thước ảnh gốc", f"{im['orig_width']}, {im['orig_height']}"],
        ["fg_coverage", "REAL", "Tỉ lệ vùng cây sau phân đoạn", _f(float(im["fg_coverage"] or 0), 3)],
        ["ingested_at", "TEXT", "Thời điểm nạp vào CSDL", ex("ingested_at")[:10]],
    ], widths=[3.4, 2.6, 5.0, 5.0], caption="Lược đồ siêu dữ liệu của đối tượng ảnh (bảng images)", size=9.5)


def sec_digitization(r: PtitReport, ctx: dict):
    q = ctx["quality"]
    rows = ctx["manifest"]
    avg_kb = sum(int(x["file_size"]) for x in rows) / len(rows) / 1024
    hp, ap = q["height_pct"], q["aspect_pct"]
    n, port = q["n"], q["portrait"]
    land = n - port
    r.h(2, "1.3. Thông số kỹ thuật số hoá và tiêu chuẩn chuẩn hoá ảnh")
    r.h(3, "1.3.1. Thông số kỹ thuật của ảnh")
    r.bullets([
        ("Định dạng: ", f"JPEG, chất lượng nén {config.JPEG_QUALITY}. Ảnh chụp ngoài trời giàu chi tiết tự nhiên nên "
                        f"JPEG cho dung lượng nhỏ (trung bình {avg_kb:.0f} KB/ảnh) mà không làm mất thông tin cần cho "
                        "đặc trưng; định dạng nén không mất dữ liệu như PNG cho file lớn hơn nhiều lần."),
        ("Không gian màu: ", "sRGB 3 kênh, 24 bit/điểm ảnh; ảnh được xoay theo thẻ EXIF trước khi xử lý. Khi trích rút "
                             "đặc trưng, ảnh được chuyển sang HSV (đặc trưng màu), ảnh xám (kết cấu), thu về 224×224 "
                             "(embedding) và 64×64 (hệ số DCT)."),
        ("Kích thước: ", f"{config.IMAGE_SIZE}×{config.IMAGE_SIZE} điểm ảnh cho mọi ảnh. Độ phân giải này đủ giữ chi "
                         "tiết cành lá cho các đặc trưng kết cấu, trùng với kích thước đầu vào của mô hình phân đoạn "
                         "SegFormer và giúp các đặc trưng chia lưới (lưới 3×3 của màu, 8×8 của hình dạng) có cùng thước "
                         "đo trên mọi ảnh."),
    ])
    r.h(3, "1.3.2. Căn cứ chọn góc chụp ngang và tỉ lệ khung hình tương đồng")
    r.rich([("Góc chụp ngang. ", {"bold": True}),
            ("Nhóm chọn ảnh chụp ngang tầm mắt từ phía bên vì ba lý do:", {})])
    r.numbered([
        "Hình chiếu đứng cho thấy đủ các dấu hiệu nhận diện cây: dáng tán, chiều dài thân lộ, tỉ lệ giữa thân và tán. "
        "Ảnh chụp từ trên cao chỉ thấy mặt tán, còn ảnh ngước từ gốc làm tán bị biến dạng phối cảnh.",
        "Trục thân cây luôn gần thẳng đứng nên các đặc trưng hình dạng như lưới 8×8 hay profile bề rộng từ đỉnh xuống "
        "gốc so sánh được trực tiếp giữa các ảnh mà không cần chuẩn hoá hướng.",
        "Ảnh có bố cục ba tầng trời – tán – đất, giúp tách vùng cây khỏi nền dễ hơn.",
    ])
    r.p("Điều kiện này được bảo đảm ngay từ khâu sưu tầm: chỉ duyệt các category “cây mọc đơn lẻ”, loại các tiêu "
        "đề/category có từ khoá aerial, drone, close, detail, bark…, và kiểm duyệt thủ công bằng mắt.")
    r.pm(f"Mọi ảnh được đưa về cùng khung vuông 1:1 cạnh $S={config.IMAGE_SIZE}$. Với ảnh gốc rộng $w$ và cao $h$, "
         "ảnh được co theo một hệ số $s$ chung cho hai chiều (không làm méo) sao cho cạnh ngắn bằng $S$, rồi cắt lấy "
         "phần giữa bắt đầu từ điểm $(x_0,y_0)$:", lead="Tỉ lệ khung hình. ")
    r.eq(r"s=\frac{S}{\mathrm{min}(w,h)},\qquad w'=\mathrm{round}(s\,w),\qquad h'=\mathrm{round}(s\,h)")
    r.eq(r"x_0=\left\lfloor\frac{w'-S}{2}\right\rfloor,\qquad y_0=\left\lfloor\frac{h'-S}{2}\right\rfloor")
    r.p("Để vật thể có tỉ lệ tương đồng giữa các ảnh, cổng kiểm soát chất lượng sau phân đoạn chỉ nhận ảnh có vùng cây "
        "chiếm 4–90% diện tích và tâm vùng cây lệch tâm ảnh không quá 30% mỗi trục. Đo lại trên "
        f"{n} ảnh của CSDL: chiều cao cây chiếm trung vị {pc(hp['50'], 0)} chiều cao ảnh (một nửa số ảnh nằm trong "
        f"khoảng {pc(hp['25'], 0)}–{pc(hp['75'], 0)}); tỉ lệ cao/rộng của khung bao cây có trung vị {_f(ap['50'], 2)} và "
        f"{pc(q['aspect_in_06_16'], 0)} số ảnh nằm trong khoảng 0,6–1,6.")
    r.rich([("Đánh giá. ", {"bold": True}),
            ("Phép cắt giữa giữ cho ảnh luôn kín khung, không có dải đệm, nhưng với ảnh gốc dạng dọc (cao hơn rộng, "
             f"{port} ảnh) phép cắt bỏ đi phần trên và dưới nên ngọn cây dễ bị cắt: {q['crown_cut_top_portrait']}/{port} "
             f"ảnh gốc dọc ({pc(q['crown_cut_top_portrait'] / port, 0)}) có tán chạm mép trên, so với "
             f"{q['crown_cut_top_landscape']}/{land} ảnh gốc ngang ({pc(q['crown_cut_top_landscape'] / land, 0)}). "
             "Đây là hạn chế chính của bộ dữ liệu hiện tại (mục 1.5.2). Hướng khắc phục là cắt khung vuông quanh khung "
             "bao của cây (có chừa lề) thay cho cắt giữa, hoặc co ảnh theo cạnh dài với hệ số dưới đây rồi đệm nền cho "
             "đủ khung vuông, để giữ trọn cây và làm tỉ lệ vật trong khung đồng đều hơn:", {})])
    r.eq(r"s'=\frac{S}{\mathrm{max}(w,h)}")


def sec_statistics(r: PtitReport, ctx: dict):
    imgs, rows, q = ctx["images"], ctx["manifest"], ctx["quality"]
    n_db = ctx["db"]["images"]
    lic = Counter(i.get("license") or "" for i in imgs)
    ow = [int(x["orig_width"]) for x in imgs if x.get("orig_width")]
    oh = [int(x["orig_height"]) for x in imgs if x.get("orig_height")]
    gr = q["groups"]
    r.h(2, "1.4. Thống kê bộ dữ liệu")
    r.h(3, "1.4.1. Bảng tổng hợp số liệu")
    r.table(["Chỉ tiêu", "Giá trị"], [
        ["Số ảnh trong CSDL", str(n_db)],
        ["Số ảnh truy vấn thử (ngoài CSDL)", str(len(ctx["queries"]))],
        ["Kích thước lưu trữ", f"{config.IMAGE_SIZE}×{config.IMAGE_SIZE}, JPEG, RGB"],
        ["Dung lượng trung bình / ảnh", f"{sum(int(x['file_size']) for x in rows) / len(rows) / 1024:.0f} KB"],
        ["Kích thước ảnh gốc (rộng × cao)", f"{min(ow)}–{max(ow)} × {min(oh)}–{max(oh)} px"],
        ["Ảnh gốc dạng dọc / ngang", f"{q['portrait']} / {n_db - q['portrait']}"],
        ["Nhóm hình thái", f"lá rộng {gr.get('broadleaf', 0)}, lá kim {gr.get('conifer', 0)}, chưa có nhãn "
                           f"{gr.get('unknown', 0)}"],
        ["Số ảnh có nhãn chi (genus)", f"{q['labeled']} ({q['n_genera']} chi)"],
        ["Số tác giả khác nhau", str(len({i.get('author') for i in imgs}))],
        ["Giấy phép", ", ".join(f"{k or '(không rõ)'}: {v}" for k, v in lic.most_common(6))],
    ], widths=[6, 10], caption="Thống kê tổng quan bộ dữ liệu", size=10.5)
    gc = q["genus_counts"]
    top = list(gc.items())[:12]
    r.table(["STT", "Chi (genus)", "Tên thường gọi", "Nhóm", "Số ảnh", "Tỉ lệ trong CSDL"],
            [[str(k), g, GENUS_VI.get(g, "—"), "lá kim" if g in CONIFER else "lá rộng", str(c), pc(c / n_db)]
             for k, (g, c) in enumerate(top, 1)]
            + [["", "Các chi khác", "", "", str(sum(gc.values()) - sum(c for _, c in top)), ""],
               ["", "Chưa xác định chi", "", "", str(n_db - q["labeled"]), pc((n_db - q["labeled"]) / n_db)]],
            widths=[1.2, 3.2, 3.4, 2.2, 2.0, 2.6], caption="Số ảnh theo chi (12 chi nhiều ảnh nhất)", size=10,
            align_center_cols={0, 3, 4, 5})
    r.h(3, "1.4.2. Các biểu đồ phân bố của bộ dữ liệu")
    r.p("Hình dưới cho biết số ảnh của từng chi và của từng nhóm hình thái, qua đó đánh giá mức độ cân bằng giữa các lớp "
        "(mục 1.5.1).")
    r.figure(FIG / "19_genus_distribution.png", "Phân bố số ảnh theo chi (trái) và theo nhóm hình thái (phải)", 16)
    r.p("Phân bố các thuộc tính bậc thấp trên toàn bộ dữ liệu được thể hiện ở hình tiếp theo: sắc độ, độ bão hoà, độ "
        "sáng trung bình của vùng cây (thể hiện sự đa dạng về mùa và ánh sáng), tỉ lệ diện tích vùng cây và tỉ lệ "
        "cao/rộng của khung bao cây (thể hiện kích thước và dáng cây trong khung hình).")
    r.figure(FIG / "02_dataset_stats.png", "Phân bố một số thuộc tính bậc thấp trên toàn bộ dữ liệu: sắc độ, độ bão "
             "hoà, độ sáng trung bình của vùng cây; tỉ lệ vùng cây; tỉ lệ cao/rộng; số ảnh theo nhóm", 16)


def sec_balance_limits(r: PtitReport, ctx: dict):
    q = ctx["quality"]
    n, b, gr = q["n"], q["balance"], q["groups"]
    hp, cp = q["height_pct"], q["coverage_pct"]
    ex = q["examples"]
    r.h(2, "1.5. Đánh giá tính cân bằng và các hạn chế của bộ dữ liệu")
    r.h(3, "1.5.1. Đánh giá tính cân bằng của bộ dữ liệu")
    r.p("Mức độ cân bằng giữa các lớp được đo bằng xác suất tiên nghiệm của từng lớp và hệ số mất cân bằng (Imbalance "
        "Ratio); IR càng gần 1 thì bộ dữ liệu càng cân bằng:")
    r.eq(r"p_k=\frac{n_k}{N},\qquad \mathrm{IR}=\frac{\mathrm{max}_{k}\,n_k}{\mathrm{min}_{k}\,n_k}")
    r.p("Bộ dữ liệu hiện tại không cân bằng ở ba mức:")
    r.bullets([
        f"Chỉ {q['labeled']}/{n} ảnh ({pc(q['labeled'] / n)}) xác định được chi từ category của Wikimedia Commons; "
        f"{n - q['labeled']} ảnh còn lại chưa có nhãn.",
        f"Với {b['n_genera']} chi có từ {b['min_images']} ảnh trở lên ({b['n_images']} ảnh), chi nhiều nhất là "
        f"{b['max_genus']} ({b['max']} ảnh, p = {_f(b['max'] / n, 3)}) và ít nhất là {b['min_genus']} ({b['min']} ảnh, "
        f"p = {_f(b['min'] / n, 3)}), nên IR = {_f(b['ir'], 1)}.",
        f"Nhóm lá rộng ({gr.get('broadleaf', 0)} ảnh) nhiều gấp {_f(gr.get('broadleaf', 0) / max(gr.get('conifer', 1), 1), 1)} "
        f"lần nhóm lá kim ({gr.get('conifer', 0)} ảnh).",
    ])
    r.p("Nguyên nhân là số ảnh “cây mọc đơn lẻ” có giấy phép mở trên Commons rất khác nhau giữa các loài: các loài phổ "
        "biến ở châu Âu (dương, sồi, thông, óc chó) có nhiều ảnh được phân loại kỹ, còn phần lớn ảnh trong category "
        "chung chỉ được ghi là “Solitary trees”. Đề bài không yêu cầu bộ dữ liệu có nhãn; nhãn chi chỉ dùng để đánh giá "
        "đặc trưng và kết quả tìm kiếm. Ảnh hưởng của mất cân bằng đến đánh giá được kiểm soát bằng cách chỉ đánh giá "
        "trên các chi có đủ ảnh (từ 9 ảnh trở lên ở mục 3.10), so sánh với mức ngẫu nhiên tính bằng hoán vị nhãn (đã "
        "phản ánh tỉ lệ giữa các chi) và báo cáo thêm chỉ số theo nhóm hình thái. Hướng cải thiện là bổ sung ảnh có "
        "nhãn cho các chi ít ảnh và gán nhãn lại các ảnh chưa xác định được chi.")

    r.h(3, "1.5.2. Các hạn chế tồn tại trong bộ dữ liệu thực tế")
    r.p("Các hạn chế dưới đây được đo tự động trên mặt nạ vùng cây của từng ảnh (script scripts/analyze_dataset.py, "
        "kết quả trong results/dataset_quality.json); ví dụ minh hoạ ở hình cuối mục.")
    r.rich([("a. Tán cây chạm hoặc bị cắt ở mép khung hình. ", {"bold": True, "italic": True}),
            (f"{q['crown_cut_top']}/{n} ảnh ({pc(q['crown_cut_top'] / n)}) có vùng cây chạm mép trên trên "
             f"{pc(q['thresholds']['top_cut'], 0)} chiều rộng ảnh, phần lớn do ảnh gốc dọc bị cắt giữa (mục 1.3.2). Một "
             "số trường hợp bị đếm thừa vì mặt nạ dính vào rừng ở nền phía sau, nên đây là ước lượng. Tán bị cắt làm "
             "sai lệch các đặc trưng hình dạng (khung bao, lưới 8×8, profile bề rộng, đường bao).", {})])
    r.rich([("b. Kích thước cây trong khung không đồng đều. ", {"bold": True, "italic": True}),
            (f"Chiều cao cây dao động từ {pc(hp['0'], 0)} đến {pc(hp['100'], 0)} chiều cao ảnh; {q['small_tree']} ảnh có "
             f"cây thấp hơn 1/4 chiều cao ảnh (chưa đạt tiêu chí ở mục 1.1.1) và {q['below_third']} ảnh thấp hơn 1/3. "
             f"Diện tích vùng cây dao động từ {pc(cp['0'])} đến {pc(cp['100'])}. Các đặc trưng hình dạng được chuẩn hoá "
             "theo khung bao của cây nên ít bị ảnh hưởng, nhưng các đặc trưng màu toàn ảnh và bố cục chịu ảnh hưởng "
             "rõ.", {})])
    s = ex["series"]
    r.rich([("c. Nhiều ảnh của cùng một cây. ", {"bold": True, "italic": True}),
            ("Một số tác giả đóng góp cả loạt ảnh chụp cùng một cây ở các thời điểm hoặc góc khác nhau (ví dụ "
             f"{', '.join(sorted(x.replace('.jpg', '') for x in s))}). Khử trùng lặp bằng perceptual hash chỉ loại được "
             f"ảnh gần như giống hệt nên các ảnh này vẫn nằm trong CSDL: {pc(q['same_author_nn'])} số ảnh có ảnh giống "
             "nhất (theo embedding sâu) là ảnh cùng tác giả. Điều này làm các chỉ số đánh giá dựa trên ảnh gần nhất cao "
             "hơn thực tế, nên mục 3.10 báo cáo thêm chỉ số đã loại ảnh cùng tác giả.", {})])
    r.rich([("d. Nhãn thiếu và mất cân bằng. ", {"bold": True, "italic": True}),
            (f"{pc((n - q['labeled']) / n, 0)} số ảnh chưa có nhãn chi và số ảnh giữa các chi chênh lệch lớn "
             "(mục 1.5.1).", {})])
    r.rich([("e. Biến thiên điều kiện chụp. ", {"bold": True, "italic": True}),
            ("Cùng một loài xuất hiện với tán xanh, vàng, trụi lá hoặc phủ tuyết; nhiều ảnh chụp ngược sáng, trong "
             f"sương mù hoặc hoàng hôn; {q['grayscale']} ảnh là ảnh đen trắng hoặc gần đơn sắc (độ bão hoà trung bình "
             f"dưới {pc(q['thresholds']['gray_saturation'], 0)}) nên đặc trưng màu gần như không có tác dụng; một số ảnh "
             "đã được chỉnh màu nghệ thuật; nền có công trình, xe cộ, động vật. Các biến thiên này được giảm bớt nhờ "
             "dùng không gian HSV, tách vật thể khỏi nền, đặc trưng LBP bất biến với độ sáng và kết hợp nhiều bộ đặc "
             "trưng.", {})])
    r.figure(FIG / "18_dataset_limitations.png", "Ví dụ các hạn chế của bộ dữ liệu: tán bị cắt ở ảnh gốc dọc (hàng "
             "trên), cây nhỏ trong khung (giữa), loạt ảnh chụp cùng một cây (dưới)", 15.5)


# =============================================================================================
# PHAN II. Giong / khac
# =============================================================================================
def part2(r: PtitReport, ctx: dict):
    q, fa = ctx["quality"], ctx["feature_analysis"]
    gs = q["genus_stats"]
    cp = q["coverage_pct"]
    r.h(1, "PHẦN II. PHÂN TÍCH ĐẶC ĐIỂM GIỐNG VÀ KHÁC NHAU GIỮA CÁC CÂY")
    r.h(2, "2.1. Đặc điểm giống nhau giữa các cây")
    r.h(3, "2.1.1. Cấu trúc sinh học và hình học tổng thể")
    r.p("Mọi cây thân gỗ trưởng thành trong ảnh đều gồm hai phần nhìn thấy được: thân gỗ chính (một hoặc vài thân) và "
        "tán gồm cành, lá. Trục chính của cây gần như thẳng đứng; thân ở dưới hẹp hơn tán ở trên nên profile bề rộng "
        "theo chiều cao có dạng “hẹp ở gốc, rộng ở giữa hoặc ở trên”. Hình bao của tán gần đối xứng trái – phải qua "
        "trục thân.")
    r.h(3, "2.1.2. Bố cục ảnh và màu sắc")
    r.p("Các ảnh có bố cục ba tầng thống nhất: phần trên là bầu trời (xanh lam, trắng xám hoặc ánh hoàng hôn), phần giữa "
        "là tán cây, phần dưới là mặt đất (cỏ, ruộng, cát, tuyết). Vùng cây chiếm trung vị khoảng "
        f"{pc(cp['50'])} diện tích ảnh. Màu chủ đạo của vật thể thuộc dải xanh lục – nâu – xám; màu nền thuộc dải xanh "
        "lam – trắng (trời) và xanh lục – vàng – nâu (đất). Vì vậy biểu đồ màu toàn ảnh của các ảnh khá giống nhau, "
        "phải tách riêng biểu đồ của vật thể và của nền mới phân biệt tốt.")
    r.h(3, "2.1.3. Kết cấu")
    r.p("Tán cây có kết cấu không đều, nhiều cạnh nhỏ theo mọi hướng, khác hẳn vùng trời phẳng và vùng đất tương đối "
        "mịn. Đây là cơ sở để các đặc trưng kết cấu (GLCM, LBP, hướng cạnh) mô tả “chất liệu” của tán.")
    r.h(3, "2.1.4. Tính liên thông của vật thể")
    r.p("Cây là thành phần liên thông lớn nhất và nằm gần tâm ảnh (lệch tâm không quá 30%, mục 1.3.2), nên có thể tách "
        "thành một mặt nạ nhị phân duy nhất và tính các đặc trưng hình dạng (diện tích, bao lồi, moment, đường bao) "
        "trên cùng một cơ sở toán học cho mọi ảnh.")

    r.h(2, "2.2. Đặc điểm khác nhau giữa các cây")
    r.h(3, "2.2.1. Dáng tán và tỉ lệ thân – tán")
    asp = {g: v["aspect"] for g, v in gs.items()}
    sol = {g: v["solidity"] for g, v in gs.items()}
    flat, tall = min(asp, key=asp.get), sorted(asp, key=asp.get)[-2:]
    dense = sorted(sol, key=sol.get)[-2:]
    sparse = sorted(sol, key=sol.get)[:3]
    nm = lambda g: f"{GENUS_VI[g]} ({g})"  # noqa: E731
    r.p("Đây là khác biệt rõ nhất giữa các chi. Theo số đo trên mặt nạ vùng cây (hình và bảng dưới), "
        f"{nm(flat)} có tán dẹt nhất (tỉ lệ cao/rộng trung bình {_f(asp[flat], 2)}), sồi (Quercus) và óc chó "
        f"(Juglans) có tán tròn rộng ({_f(asp['Quercus'], 2)}; {_f(asp['Juglans'], 2)}), trong khi "
        f"{' và '.join(nm(g) for g in tall)} có tán cao hơn ({'; '.join(_f(asp[g], 2) for g in tall)}). "
        f"{' và '.join(nm(g) for g in dense)} có tán đặc nhất (solidity {'; '.join(_f(sol[g], 2) for g in dense)}), "
        f"còn {', '.join(nm(g) for g in sparse)} có tán thưa, nhiều khoảng hở (solidity "
        f"{'; '.join(_f(sol[g], 2) for g in sparse)}). Các khác biệt này được khai thác bởi tỉ lệ cao/rộng, solidity, "
        "tâm sai, moment Hu, lưới 8×8 và profile bề rộng của bộ đặc trưng hình dạng.")
    r.figure(FIG / "17_genus_representatives.png", "Ảnh đại diện của các chi (ảnh có dáng gần trung bình của chi nhất): "
             "ảnh và đường bao vùng cây (trên), hình bóng của cây (dưới) kèm số đo trung bình của chi", 16.5)
    r.h(3, "2.2.2. Mùa và trạng thái lá")
    r.p("Lá xanh mùa hè, vàng/cam mùa thu, trụi lá mùa đông, phủ tuyết hoặc sương giá. Sắc độ (H) và độ bão hoà (S) "
        "của vùng cây thay đổi mạnh theo mùa; cây trụi lá lộ hệ cành nên có mật độ cạnh và biểu đồ LBP khác hẳn cây có "
        "lá. Do đó cùng một cây chụp ở hai mùa có thể khác nhau về màu nhiều hơn hai cây khác loài chụp cùng mùa.")
    r.h(3, "2.2.3. Nhóm lá kim và lá rộng")
    g2 = fa["groups"]
    v_b, v_c = g2["broadleaf"]["Độ sáng V trung bình"][0], g2["conifer"]["Độ sáng V trung bình"][0]
    e_b, e_c = g2["broadleaf"]["Mật độ cạnh"][0], g2["conifer"]["Mật độ cạnh"][0]
    r.p(f"Tán lá kim có màu xanh sẫm (độ sáng V trung bình {_f(v_c, 2)} so với {_f(v_b, 2)} của lá rộng), ít đổi màu "
        f"theo mùa, kết cấu mịn và đều hơn (mật độ cạnh {_f(e_c, 2)} so với {_f(e_b, 2)}); dáng tán hình nón hoặc lệch, "
        "không đối xứng. Tán lá rộng sáng hơn, kết cấu thô hơn và đổi màu rõ theo mùa.")
    r.h(3, "2.2.4. Bối cảnh, ánh sáng và khoảng cách chụp")
    r.p("Nền ảnh rất đa dạng: đồng cỏ, ruộng hoa cải vàng, hoang mạc, tuyết, bờ biển, mặt nước; ánh sáng ngược "
        "(silhouette), sương mù, hoàng hôn, ảnh đen trắng. Các yếu tố này ảnh hưởng đến biểu đồ màu của nền, độ sáng và "
        f"độ tương phản. Khoảng cách chụp khác nhau làm tỉ lệ vùng cây dao động từ {pc(cp['0'])} đến {pc(cp['100'])}; "
        "các đặc trưng hình dạng được chuẩn hoá theo khung bao của cây để giảm ảnh hưởng này.")

    r.h(2, "2.3. Bảng đối sánh hình thái và đặc trưng thị giác theo chi")
    r.p("Bảng dưới tổng hợp đặc điểm hình thái của các chi có nhiều ảnh nhất trong CSDL theo các tiêu chí quan sát được "
        "trên ảnh: dáng tán và thân, lá và màu theo mùa, dấu hiệu nhận diện trong ảnh; hai cột cuối là số đo trung bình "
        "trên mặt nạ vùng cây. Ảnh đại diện của từng chi ở hình trong mục 2.2.1.")
    rows = []
    for g, grp, shape, leaf, sign in MORPHOLOGY:
        st = gs[g]
        rows.append([f"{g} ({GENUS_VI[g]}), {st['n']} ảnh", grp, shape, leaf, sign, _f(st["aspect"], 2),
                     _f(st["solidity"], 2)])
    r.table(["Chi", "Nhóm", "Dáng tán, thân", "Lá, màu theo mùa", "Dấu hiệu nhận diện", "Cao/ rộng", "Độ đặc tán"],
            rows, widths=[2.4, 1.3, 3.6, 3.1, 3.0, 1.2, 1.2],
            caption="Đối sánh hình thái và đặc trưng thị giác của các chi trong CSDL", size=9,
            align_center_cols={1, 5, 6})
    sec_quant_dataset(r, fa, "2.4")


# =============================================================================================
# PHAN III. Bo dac trung
# =============================================================================================
def part3(r: PtitReport, ctx: dict):
    fa = ctx["feature_analysis"]
    r.h(1, "PHẦN III. XÂY DỰNG CÁC BỘ ĐẶC TRƯNG NHẬN DIỆN CÂY")
    r.p("Mỗi ảnh cây được biểu diễn bằng các vector đặc trưng số để CSDL có thể đo khoảng cách tương đồng giữa các ảnh. "
        "Phần này trình bày định hướng thiết kế, bước tách vùng cây khỏi nền, bốn bộ đặc trưng cùng cơ sở toán học, "
        "cách chuẩn hoá và kết hợp các bộ, so sánh ưu nhược điểm, và đánh giá định lượng giá trị thông tin của từng "
        "đặc trưng.")
    sec_design(r, ctx)
    sec_segmentation(r, ctx)
    sec_color(r)
    sec_shape(r)
    sec_texture(r)
    sec_deep(r)
    sec_normalize(r, ctx)
    sec_illustration(r)
    sec_pros_cons(r)
    sec_info_value(r, fa, "3.10")
    sec_example(r, fa, "3.11")


def sec_design(r: PtitReport, ctx: dict):
    fs = {f["set_name"]: f for f in ctx["feature_sets"]}
    r.h(2, "3.1. Định hướng thiết kế")
    r.p("Theo lý thuyết truy vấn ảnh dựa trên nội dung, ảnh được mô tả bằng các thuộc tính bậc thấp: màu sắc (của vật "
        "thể và của nền), hình dạng/kích thước vật thể, và bố cục/kết cấu (độ thô, độ tương phản, hướng sắp xếp của các "
        "nét). Mỗi loại thuộc tính bắt được một khía cạnh khác nhau của sự giống và khác nên hệ thống xây dựng ba bộ "
        "đặc trưng thủ công tương ứng và bổ sung một bộ đặc trưng ngữ nghĩa bậc cao (embedding từ mạng nơ-ron tích "
        "chập) để thu hẹp “khoảng cách ngữ nghĩa”. Các bộ được lưu trữ riêng, có độ đo khoảng cách riêng và được kết "
        "hợp theo trọng số khi truy vấn. Mọi đặc trưng đều được tính trên ảnh đã chuẩn hoá và (trừ embedding) dựa trên "
        "mặt nạ vùng cây để loại ảnh hưởng của nền.")
    r.table(["Bộ đặc trưng", "Số chiều", "Độ đo", "Trả lời câu hỏi", "Tìm tương đồng", "Tìm khác biệt"], [
        ["1. Màu sắc", str(fs["color"]["dim"]), "L1", "Cây/nền có màu gì, phân bố ở đâu?",
         "Cùng mùa, cùng loại tán, cùng bối cảnh", "Lá xanh ↔ vàng ↔ trụi lá; trời ↔ tuyết ↔ cát"],
        ["2. Hình dạng", str(fs["shape"]["dim"]), "L2 (z-score)", "Tán cây hình gì, thân dài hay ngắn?",
         "Cùng chi (tán tròn / cao hẹp / hình ô)", "Tỉ lệ cao/rộng, tâm sai, độ lõm tán"],
        ["3. Kết cấu / bố cục", str(fs["texture"]["dim"]), "L2 (z-score)", "Tán mịn hay thô, nét theo hướng nào?",
         "Lá kim ↔ lá kim; trụi lá ↔ trụi lá", "Độ thô, mật độ cạnh, năng lượng DCT"],
        ["4. Embedding sâu", str(fs.get("deep", {}).get("dim", "—")), "cosine",
         "Ảnh “trông giống” ảnh nào về ngữ nghĩa?", "Khái quát hoá loài, dáng cây", "Phân biệt cảnh nền, vật thể"],
    ], widths=[2.8, 1.5, 2.0, 3.4, 3.2, 3.1], caption="Tổng quan các bộ đặc trưng", size=9.5)
    r.figure(FIG / "05_feature_sets.png", "Cấu trúc các khối trong từng bộ đặc trưng lưu trong CSDL", 15)


def sec_segmentation(r: PtitReport, ctx: dict):
    seg = ctx.get("seg_stats", {})
    r.h(2, "3.2. Tách vùng cây khỏi nền – tiền đề của các đặc trưng vật thể")
    r.p("Theo lý thuyết truy vấn ảnh dựa trên màu sắc, biểu đồ màu toàn ảnh chịu “hiệu ứng che mặt”: các ô màu lớn của "
        "nền (trời, cỏ) lấn át màu của vật thể. Vì vậy các khối hsv_fg, moments, toàn bộ bộ hình dạng và bộ kết cấu được "
        "tính trên mặt nạ vùng cây M. Mặt nạ được tạo bằng mô hình phân đoạn ngữ nghĩa SegFormer-b0 huấn luyện sẵn trên "
        "ADE20K (lớp “tree”), sau đó lọc hình thái (đóng, mở), giữ thành phần liên thông lớn và gần tâm ảnh nhất và "
        "lấp lỗ để được hình bao của cây; GrabCut là phương án dự phòng khi không có mô hình."
        + (f" Trên {sum(seg.values())} ảnh trước cổng kiểm soát: SegFormer {seg.get('segformer', 0)}, GrabCut "
           f"{seg.get('grabcut', 0)}, không tìm thấy cây {seg.get('default_rect', 0)} (bị loại)." if seg else ""))
    r.p("Trong quá trình kiểm tra giá trị thông tin (mục 3.10), nhóm phát hiện phiên bản đầu dùng GrabCut làm bước chính "
        "khiến 135/639 ảnh (21%) có vector hình dạng trùng hệt nhau: GrabCut chỉ dựa vào màu nên không tách được tán lá "
        "khỏi cỏ cùng màu hoặc khi tán chạm mép ảnh, và hệ thống rơi về mặt nạ chữ nhật mặc định. Việc chuyển sang mô "
        "hình ngữ nghĩa loại bỏ lỗi này và làm tăng độ chính xác của mọi bộ đặc trưng thủ công.")
    r.figure(FIG / "03_segmentation.png", "Ví dụ tách vùng cây: ảnh gốc (trên) và mặt nạ vùng cây (dưới)", 16)


def sec_color(r: PtitReport):
    r.h(2, "3.3. Bộ đặc trưng 1 – Màu sắc (369 chiều, độ đo L1)")
    r.pm("Không gian màu HSV được chọn thay cho RGB vì sắc độ $H$ tách “màu gì” khỏi độ sáng $V$, nên ít phụ thuộc "
         "thiết bị và điều kiện chiếu sáng. Mỗi kênh được lượng tử hoá thành một số mức; $H$ được chia mịn nhất vì mang "
         "nhiều thông tin nhất. Với mỗi điểm ảnh $p$, gọi $q(p)$ là chỉ số ô màu chứa $(H_p,S_p,V_p)$; biểu đồ màu của "
         "vùng $R$ được chuẩn hoá để tổng bằng 1, không phụ thuộc số điểm ảnh:")
    r.eq(r"h_j=\frac{1}{|R|}\sum_{p\in R}{\left[q(p)=j\right]},\quad j=1,\dots,n,\qquad \sum_{j=1}^{n}{h_j}=1")
    r.pm(r"trong đó $R$ là toàn ảnh (khối hsv_global, $n=8\times 3\times 3=72$ ô), từng ô của lưới 3×3 (khối hsv_grid, "
         r"mỗi ô $n=4\times 2\times 2=16$), vùng cây $M$ (khối hsv_fg) hoặc phần nền (khối hsv_bg). Bộ màu gồm 5 khối:")
    r.table(["Khối", "Chiều", "Cách tính", "Giá trị thông tin"], [
        ["hsv_global", "72", "Biểu đồ HSV 8×3×3 trên toàn ảnh",
         "Ấn tượng màu tổng thể (cây + nền), tìm ảnh cùng “tông” màu; nhạy với nền nên dễ bị “che mặt”"],
        ["hsv_grid", "144", "Chia ảnh lưới 3×3, mỗi vùng một biểu đồ 4×2×2 = 16 ô",
         "Quan hệ không gian của màu: trời ở trên, tán ở giữa, đất ở dưới; phân biệt ảnh cùng tổng màu nhưng bố trí khác"],
        ["hsv_fg", "72", "Biểu đồ HSV chỉ trên mặt nạ vùng cây",
         "Màu của chính cây (lá xanh/vàng/nâu, trụi lá), khắc phục hiệu ứng che mặt của nền"],
        ["hsv_bg", "72", "Biểu đồ HSV trên phần nền (ngoài mặt nạ)",
         "Bối cảnh (trời xanh, tuyết, cỏ, cát); hữu ích khi muốn tìm ảnh cùng cảnh"],
        ["moments", "9", "Trung bình, độ lệch chuẩn, độ lệch (skewness) của H, S, V trên vùng cây",
         "Mô tả gọn phân bố màu; bền với nhiễu lượng tử hoá (hai màu gần nhau rơi vào hai ô khác nhau)"],
    ], widths=[2.3, 1.3, 5.2, 7.2], caption="Các khối của bộ đặc trưng màu sắc", size=9.5)
    r.pm(r"Moment màu được tính trên $N$ điểm ảnh của vùng cây, với $P_{c,i}$ là giá trị kênh $c\in\{H,S,V\}$ (đã đưa "
         r"về $[0,1]$) của điểm ảnh thứ $i$:")
    r.eq(r"\mu_c=\frac{1}{N}\sum_{i=1}^{N}{P_{c,i}},\qquad \sigma_c=\sqrt{\frac{1}{N}\sum_{i=1}^{N}{\left(P_{c,i}-"
         r"\mu_c\right)^2}}")
    r.eq(r"s_c=\tanh\left(\frac{1}{N\sigma_c^3}\sum_{i=1}^{N}{\left(P_{c,i}-\mu_c\right)^3}\right)")
    r.p("Hàm tanh đưa độ lệch về khoảng (−1, 1) để cùng thang đo với hai moment còn lại. Hai vector màu được so sánh "
        "bằng khoảng cách L1; với các biểu đồ đã chuẩn hoá, L1 liên hệ trực tiếp với độ giao biểu đồ (histogram "
        "intersection):")
    r.eq(r"d_{L1}(X,Y)=\sum_{j=1}^{n}{\left|x_j-y_j\right|}=2\left(1-\sum_{j=1}^{n}{\mathrm{min}\left(x_j,y_j\right)}\right)")
    r.p("Khoảng cách bằng 0 khi hai phân bố màu trùng nhau và đạt tối đa 2 cho mỗi biểu đồ khi hai ảnh không có màu "
        "chung. Bộ màu giúp tìm ảnh cùng mùa, cùng trạng thái lá và cùng bối cảnh; trong đó biểu đồ màu vùng cây "
        "(hsv_fg) là khối màu mang nhiều thông tin nhất về chi cây (mục 3.10).")


def sec_shape(r: PtitReport):
    r.h(2, "3.4. Bộ đặc trưng 2 – Hình dạng tán/thân cây (113 chiều, độ đo L2)")
    r.p("Đặc trưng hình dạng được tính trên mặt nạ nhị phân M của cây (mục 3.2) và được thiết kế để ít phụ thuộc vào vị "
        "trí, kích thước và (một phần) góc quay của cây trong ảnh:")
    r.table(["Khối", "Chiều", "Cách tính", "Giá trị thông tin"], [
        ["region", "10", "Diện tích tương đối; tỉ lệ cao/rộng của hình chữ nhật cơ bản; extent; solidity; tâm sai; độ "
                         "dài trục chính, trục phụ; hướng trục chính; độ lệch tâm (dx, dy)",
         "Thước đo hình dạng cơ bản: tán cao hẹp ↔ tán tròn rộng; tán đặc ↔ tán thưa, lõm (cây trụi lá có solidity "
         "thấp)"],
        ["hu", "7", "7 moment bất biến Hu, lấy −sign·log₁₀|φ|",
         "Bất biến với tịnh tiến, tỉ lệ và xoay; mô tả phân bố khối lượng của hình"],
        ["grid", "64", "Lưới 8×8 trên hình chữ nhật cơ bản, ô = 1 nếu vật thể chiếm > 15% diện tích ô",
         "Chuỗi nhị phân mô tả hình thể; so sánh trực tiếp dáng cây: tán tròn, tán hình nón, thân dài ở dưới…"],
        ["fourier", "16", "Biến đổi Fourier của đường bao ngoài (128 điểm), 16 hệ số biên độ đã chuẩn hoá",
         "Mô tả độ gồ ghề và độ đối xứng của đường viền tán ở nhiều tần số"],
        ["profile", "16", "16 dải ngang từ đỉnh xuống gốc, mỗi dải = tỉ lệ bề rộng bị vật thể chiếm",
         "“Chữ ký” dáng cây theo chiều cao: nơi tán rộng nhất, độ dài thân lộ, tán hình ô hay hình nón"],
    ], widths=[2.0, 1.2, 6.3, 6.5], caption="Các khối của bộ đặc trưng hình dạng", size=9.5)
    r.pm(r"Gọi $|M|$ là số điểm ảnh của cây, $W\times H$ là kích thước ảnh, $w_b\times h_b$ là kích thước hình chữ nhật "
         r"cơ bản (khung bao) và $\mathrm{conv}(M)$ là bao lồi của cây. Các thước đo vùng (khối region) gồm diện tích "
         r"tương đối $A$, tỉ lệ cao/rộng $r$, độ lấp đầy khung bao $E$, độ đặc $S$ và tâm sai $e$:")
    r.eq(r"A=\frac{|M|}{WH},\qquad r=\frac{h_b}{w_b},\qquad E=\frac{|M|}{w_b h_b},\qquad S=\frac{|M|}"
         r"{\left|\mathrm{conv}(M)\right|},\qquad e=\sqrt{1-\frac{b^2}{a^2}}")
    r.pm("với $a$, $b$ là bán trục lớn và bán trục nhỏ của elip có cùng moment bậc hai với vùng cây. Moment bất biến "
         "Hu (khối hu) được xây dựng từ moment trung tâm $\\mu_{pq}$ và moment trung tâm chuẩn hoá $\\eta_{pq}$ của "
         "mặt nạ:")
    r.eq(r"\mu_{pq}=\sum_{x}{\sum_{y}{(x-\bar{x})^p(y-\bar{y})^q M(x,y)}},\qquad \eta_{pq}=\frac{\mu_{pq}}"
         r"{\mu_{00}^{1+(p+q)/2}}")
    r.eq(r"\phi_1=\eta_{20}+\eta_{02},\qquad \phi_2=\left(\eta_{20}-\eta_{02}\right)^2+4\eta_{11}^2,\ \dots,\qquad "
         r"H_i=-\mathrm{sign}(\phi_i)\log_{10}\left|\phi_i\right|")
    r.pm(r"Bảy moment $\phi_1,\dots,\phi_7$ bất biến với tịnh tiến, co giãn và xoay; chúng được lấy logarit vì giá trị "
         r"chênh nhau nhiều bậc. Lưới 8×8 (khối grid) chia khung bao thành 64 ô $C_{uv}$; mỗi ô nhận giá trị 1 khi vật "
         r"thể chiếm quá 15% diện tích ô:")
    r.eq(r"g_{uv}=\begin{cases}1, & \frac{|M\cap C_{uv}|}{|C_{uv}|}>0{,}15\\ 0, & \text{ngược lại}\end{cases}")
    r.pm(r"Hệ số Fourier (khối fourier): đường bao ngoài được lấy mẫu đều $K=128$ điểm, biểu diễn dạng số phức "
         r"$z_k=x_k+i\,y_k$; bỏ $F_0$ để loại ảnh hưởng của vị trí, chia cho $|F_1|$ để loại ảnh hưởng của kích thước "
         r"và lấy biên độ để loại ảnh hưởng của điểm bắt đầu và phép xoay:")
    r.eq(r"F_m=\sum_{k=0}^{K-1}{z_k\,e^{-2\pi i mk/K}},\qquad f_m=\frac{\left|F_{m+1}\right|}{\left|F_1\right|},\quad "
         r"m=1,\dots,16")
    r.pm(r"Profile bề rộng (khối profile) chia khung bao thành 16 dải ngang $B_t$ từ đỉnh xuống gốc; $w_t$ là tỉ lệ số "
         r"cột trong dải có điểm ảnh thuộc cây:")
    r.eq(r"w_t=\frac{1}{w_b}\left|\left\{x:\exists y\in B_t,\ M(x,y)=1\right\}\right|,\quad t=1,\dots,16")
    r.p("Các chiều có thang đo khác nhau nên trước khi tính khoảng cách Euclid (L2) mỗi chiều được chuẩn hoá z-score "
        "(mục 3.7). Bộ hình dạng giúp tìm cây cùng dáng tán và phân biệt các chi có dáng khác nhau, độc lập với màu "
        "(không phụ thuộc mùa và ánh sáng).")


def sec_texture(r: PtitReport):
    r.h(2, "3.5. Bộ đặc trưng 3 – Kết cấu và bố cục (85 chiều, độ đo L2)")
    r.table(["Khối", "Chiều", "Cách tính", "Giá trị thông tin"], [
        ["glcm", "10", "Ma trận đồng xuất hiện mức xám (32 mức, khoảng cách 1 và 3, trung bình 4 hướng) trên khung bao "
                       "của cây: contrast, dissimilarity, homogeneity, energy, correlation",
         "Độ thô/mịn và độ tương phản của tán: lá kim mịn đều ↔ lá rộng thô; tán đặc ↔ tán thưa lộ trời"],
        ["lbp", "28", "Biểu đồ Local Binary Pattern đồng nhất (P=8, R=1: 10 ô; P=16, R=2: 18 ô) trên vùng cây",
         "Mẫu vi kết cấu cục bộ, bất biến với thay đổi độ sáng đơn điệu; phân biệt lá, cành trụi, tuyết phủ"],
        ["edge", "9", "Biểu đồ 8 hướng gradient Sobel (trọng số theo độ lớn) + mật độ cạnh",
         "Hướng sắp xếp các nét (cành thẳng đứng của cây trụi lá ↔ tán lá đẳng hướng) và “độ sắc của các nét”"],
        ["tamura", "3", "Độ thô (coarseness), độ tương phản (contrast), tính định hướng (directionality)",
         "Ba thuộc tính kết cấu gần với cảm nhận của người"],
        ["dct", "35", "Ảnh xám thu về 64×64, biến đổi cosine rời rạc, lấy khối 6×6 tần số thấp bỏ hệ số DC",
         "Bố cục tổng thể (phân bố sáng tối theo vùng lớn), tương ứng với đánh chỉ số trên miền nén (JPEG)"],
    ], widths=[2.0, 1.2, 6.3, 6.5], caption="Các khối của bộ đặc trưng kết cấu / bố cục", size=9.5)
    r.pm(r"Ảnh xám trong khung bao của cây được lượng tử hoá $L=32$ mức; $P(i,j)$ là xác suất hai điểm ảnh cách nhau "
         r"$d$ ($d=1,3$) theo hướng $\theta$ có mức xám $i$ và $j$. Năm thuộc tính sau được lấy trung bình trên 4 hướng "
         r"0°, 45°, 90°, 135° ($\mu$, $\sigma$ là trung bình và độ lệch chuẩn của phân bố biên theo hàng và cột):",
         lead="GLCM. ")
    r.eq(r"\mathrm{Contrast}=\sum_{i,j}{(i-j)^2P(i,j)},\qquad \mathrm{Dissimilarity}=\sum_{i,j}{|i-j|\,P(i,j)}")
    r.eq(r"\mathrm{Homogeneity}=\sum_{i,j}{\frac{P(i,j)}{1+(i-j)^2}},\qquad \mathrm{Energy}=\sqrt{\sum_{i,j}{P(i,j)^2}}")
    r.eq(r"\mathrm{Correlation}=\sum_{i,j}{\frac{(i-\mu_i)(j-\mu_j)P(i,j)}{\sigma_i\sigma_j}}")
    r.pm(r"Mỗi điểm ảnh trung tâm $g_c$ được so sánh với $P$ điểm lân cận $g_p$ trên đường tròn bán kính $R$:",
         lead="LBP. ")
    r.eq(r"\mathrm{LBP}_{P,R}=\sum_{p=0}^{P-1}{s\left(g_p-g_c\right)2^p},\qquad s(u)=\begin{cases}1, & u\geq 0\\ 0, & "
         r"u<0\end{cases}")
    r.pm(r"Chỉ giữ các mẫu đồng nhất (uniform, tối đa hai lần chuyển 0↔1) nên biểu đồ có $P+2$ ô: 10 ô với "
         r"$(P,R)=(8,1)$ và 18 ô với $(P,R)=(16,2)$.")
    r.pm(r"Gradient Sobel $G_x$, $G_y$ cho độ lớn và hướng cạnh; biểu đồ 8 hướng $e_k$ được tích luỹ theo độ lớn trên "
         r"vùng cây, kèm mật độ cạnh $\rho$:", lead="Hướng cạnh. ")
    r.eq(r"G=\sqrt{G_x^2+G_y^2},\qquad \theta=\arctan\frac{G_y}{G_x}\ (\mathrm{mod}\ \pi)")
    r.eq(r"e_k=\frac{\sum_{\theta_p\in k}{G_p}}{\sum_{p}{G_p}},\qquad \rho=\frac{\left|\left\{p\in M:G_p>60\right\}"
         r"\right|}{|M|}")
    r.pm("Độ tương phản và tính định hướng được tính theo:", lead="Tamura. ")
    r.eq(r"F_{con}=\frac{\sigma}{\alpha_4^{1/4}},\quad \alpha_4=\frac{\mu_4}{\sigma^4},\qquad F_{dir}=1+\frac{\sum_{k=1}"
         r"^{16}{p_k\ln p_k}}{\ln 16}")
    r.pm(r"trong đó $\mu_4$ là moment trung tâm bậc 4 của mức xám và $p_k$ là tỉ lệ cạnh mạnh rơi vào hướng thứ $k$; "
         r"độ thô $F_{crs}$ là trung bình $\log_2$ của kích thước cửa sổ ($2^1,\dots,2^5$) cho chênh lệch trung bình "
         r"lớn nhất tại mỗi điểm.")
    r.pm(r"Ảnh xám $f$ được thu về $N=64$ rồi biến đổi cosine rời rạc hai chiều:", lead="DCT. ")
    r.eq(r"C(u,v)=\alpha(u)\alpha(v)\sum_{x=0}^{N-1}{\sum_{y=0}^{N-1}{f(x,y)\cos\frac{(2x+1)u\pi}{2N}\cos\frac{(2y+1)"
         r"v\pi}{2N}}}")
    r.pm(r"Giữ 6×6 hệ số tần số thấp nhất ($0\leq u,v<6$) trừ hệ số DC $C(0,0)$ là độ sáng trung bình, còn 35 hệ số "
         r"mô tả bố cục sáng – tối tổng thể; đây cũng là các hệ số được lưu trong ảnh JPEG nên phù hợp với đánh chỉ số "
         r"trên miền nén.")


def sec_deep(r: PtitReport):
    r.h(2, "3.6. Bộ đặc trưng 4 (mở rộng) – Embedding sâu MobileNetV2 (1280 chiều, độ đo cosine)")
    r.p("Các thuộc tính bậc thấp không nắm bắt hết ngữ nghĩa của ảnh (“khoảng cách ngữ nghĩa”). Để bổ sung, hệ thống "
        "dùng mạng MobileNetV2 huấn luyện sẵn trên ImageNet (định dạng ONNX, chạy bằng ONNX Runtime trên CPU): ảnh được "
        "thu về 224×224, chuẩn hoá theo trung bình và độ lệch chuẩn của ImageNet, rồi lấy đầu ra của lớp Global Average "
        "Pooling (trước lớp phân loại).")
    r.pm(r"Với $F_{u,v}(I)$ là vector 1280 kênh tại vị trí $(u,v)$ của bản đồ đặc trưng cuối cùng kích thước "
         r"$H\times W=7\times 7$, embedding và khoảng cách cosine là:")
    r.eq(r"\mathbf{f}(I)=\frac{1}{HW}\sum_{u=1}^{H}{\sum_{v=1}^{W}{F_{u,v}(I)}}\in\mathbb{R}^{1280},\qquad "
         r"\hat{\mathbf{f}}=\frac{\mathbf{f}}{\|\mathbf{f}\|_2}")
    r.eq(r"d_{\cos}(I_1,I_2)=1-\hat{\mathbf{f}}(I_1)\cdot\hat{\mathbf{f}}(I_2)")
    r.p("Độ tương đồng cosine giữa hai embedding phản ánh sự giống nhau về hình thái tổng thể mà mạng đã học từ hàng "
        "triệu ảnh tự nhiên. Đây là đặc trưng tuỳ chọn: nếu không có mô hình, hệ thống vẫn hoạt động với ba bộ thủ công.")


def sec_normalize(r: PtitReport, ctx: dict):
    w = config.DEFAULT_WEIGHTS
    im = ctx["index_meta"]
    km, kd = im["kmeans"], im["kdtree"]
    r.h(2, "3.7. Chuẩn hoá và kết hợp các bộ đặc trưng")
    r.pm(r"Các bộ đặc trưng có số chiều và thang đo khác nhau nên cần được đồng bộ trước khi so sánh và kết hợp. Với "
         r"các bộ dùng khoảng cách L2 (hình dạng, kết cấu), mỗi chiều $j$ được chuẩn hoá z-score theo trung bình "
         r"$\mu_j$ và độ lệch chuẩn $\sigma_j$ của chiều đó trên toàn CSDL (lưu trong bảng feature_stats):")
    r.eq(r"z_j=\frac{x_j-\mu_j}{\sigma_j}")
    r.pm(r"Hệ số thang đo $s_k$ của bộ $k$ là khoảng cách trung bình giữa hai ảnh ngẫu nhiên trong CSDL, nên "
         r"$d_k/s_k\approx 1$ với hai ảnh “bình thường” và nhỏ hơn 1 khi hai ảnh giống nhau hơn mức ngẫu nhiên. Khoảng "
         r"cách tổng hợp giữa ảnh truy vấn $Q$ và ảnh $I$ là tổng có trọng số $w_k$ của các khoảng cách đã chuẩn hoá, và "
         r"độ tương đồng được suy ra từ khoảng cách:")
    r.eq(r"D(Q,I)=\sum_{k}{w_k\frac{d_k(Q,I)}{s_k}},\qquad \sum_{k}{w_k}=1,\qquad \mathrm{sim}(Q,I)=\frac{1}{1+D(Q,I)}")
    r.p(f"Trọng số mặc định: màu {_f(w['color'], 2)}; hình dạng {_f(w['shape'], 2)}; kết cấu {_f(w['texture'], 2)}; "
        f"embedding {_f(w['deep'], 2)}. Cách chọn trọng số và quy trình tìm kiếm sẽ được trình bày ở Báo cáo 2.")
    r.pm(r"Để quan sát phân bố ảnh trong không gian đặc trưng, vector kết hợp được chiếu bằng phân tích thành phần "
         r"chính (PCA). Với $\lambda_j$ là các trị riêng giảm dần của ma trận hiệp phương sai và $d$ là số chiều gốc, "
         r"$m$ thành phần chính đầu tiên giữ lại tỉ lệ phương sai:", lead="Giảm chiều và quan sát cấu trúc cụm. ")
    r.eq(r"\frac{\sum_{j=1}^{m}{\lambda_j}}{\sum_{j=1}^{d}{\lambda_j}}")
    r.pm(f"Với $m={kd['pca_dim']}$ thành phần, tỉ lệ này là {pc(kd['explained_variance'])}. Thuật toán K-means chia "
         f"{km['n']} ảnh thành $K={km['k']}$ cụm $C_k$ bằng cách cực tiểu hoá tổng bình phương khoảng cách từ mỗi điểm "
         r"$\mathbf{x}$ tới tâm cụm $\boldsymbol{\mu}_k$:")
    r.eq(r"J=\sum_{k=1}^{K}{\sum_{\mathbf{x}\in C_k}{\left\|\mathbf{x}-\boldsymbol{\mu}_k\right\|^2}}")
    r.p(f"Kích thước các cụm là {', '.join(str(s) for s in km['sizes'])} ảnh. Hình dưới chiếu các ảnh lên hai thành phần "
        "chính đầu tiên: các cụm K-means chiếm những vùng tương đối tách biệt của mặt phẳng, còn hai nhóm lá rộng và lá "
        "kim trải đều và chồng lấn nhau, phù hợp với nhận xét ở mục 2.4 rằng không một thuộc tính đơn lẻ nào đủ để phân "
        "biệt loài. Chỉ mục kd-tree và K-means dùng để tăng tốc tìm kiếm sẽ được trình bày ở Báo cáo 2.")
    r.figure(FIG / "06_clusters.png", "Phân bố ảnh trên hai thành phần chính đầu tiên: tô màu theo cụm K-means (trái) "
             "và theo nhóm hình thái (phải)", 16)


def sec_illustration(r: PtitReport):
    r.h(2, "3.8. Minh hoạ giá trị đặc trưng của một ảnh")
    r.p("Hình dưới hiển thị các giá trị trung gian khi trích rút đặc trưng cho ảnh truy vấn thử query_01.jpg (một cây "
        "bạch dương): mặt nạ vùng cây, biểu đồ HSV toàn ảnh, biểu đồ vật thể/nền, lưới 8×8, profile bề rộng, các thước "
        "đo vùng, biểu đồ LBP, hướng cạnh, hệ số DCT và các thuộc tính GLCM/Tamura. Mỗi khối trong các bảng ở mục "
        "3.3–3.5 tương ứng một ô trong hình.")
    r.figure(FIG / "11_query01_linear_features.png", "Giá trị trung gian của các khối đặc trưng cho ảnh query_01.jpg", 16)
    r.p("Có thể thấy biểu đồ màu của vật thể tập trung ở các ô xanh lục còn biểu đồ nền dồn vào một ô (cỏ), minh hoạ vì "
        "sao cần tách riêng hai biểu đồ; profile bề rộng hẹp dần về phía gốc đúng với dáng tán cao của cây bạch dương; "
        "tỉ lệ cao/rộng của khung bao lớn hơn 1 và biểu đồ LBP có ô cuối (mẫu không đồng nhất) cao do tán lá dày nhiều "
        "chi tiết nhỏ.")


def sec_pros_cons(r: PtitReport):
    r.h(2, "3.9. So sánh ưu, nhược điểm của các phương pháp")
    r.table(["Bộ", "Khối / phương pháp", "Ưu điểm", "Nhược điểm"], [
        ["Màu sắc", "Biểu đồ HSV toàn ảnh, lưới 3×3",
         "Đơn giản, tính nhanh; bất biến với tịnh tiến; lưới 3×3 giữ được bố cục màu",
         "Nhạy với ánh sáng và mùa; bị nền chi phối (che mặt)"],
        ["", "Biểu đồ vật thể / nền", "Tách màu của cây khỏi bối cảnh, khắc phục hiệu ứng che mặt",
         "Phụ thuộc chất lượng mặt nạ phân đoạn"],
        ["", "Moment màu", "Gọn (9 chiều), bền với nhiễu lượng tử hoá", "Mất thông tin về hình dạng của phân bố màu"],
        ["Hình dạng", "Thước đo vùng, moment Hu",
         "Ít chiều, dễ giải thích; Hu bất biến tịnh tiến, co giãn, xoay",
         "Nhạy với lỗi mặt nạ và tán bị cắt; chỉ là hình chiếu 2D nên phụ thuộc góc chụp"],
        ["", "Lưới 8×8, profile", "So sánh dáng tán trực tiếp, trực quan",
         "Không bất biến với xoay (cây luôn đứng nên chấp nhận được); nhạy khi tán bị cắt"],
        ["", "Hệ số Fourier", "Mô tả đường viền ở nhiều tần số; bất biến xoay, tỉ lệ",
         "Nhạy với nhiễu ở biên tán (cành lá lởm chởm)"],
        ["Kết cấu", "GLCM, Tamura", "Giải thích rõ ràng, gần với cảm nhận độ thô/mịn của người",
         "Phụ thuộc tham số d, θ, số mức xám; Tamura tính chậm"],
        ["", "LBP", "Bất biến với độ sáng đơn điệu; tỉ số Fisher cao nhất trong mọi khối",
         "Nhạy với nhiễu và độ phân giải; không chứa thông tin màu"],
        ["", "Hướng cạnh, DCT", "Ít chiều; DCT tương thích với miền nén JPEG",
         "DCT phản ánh bối cảnh nhiều hơn bản thân cây"],
        ["Embedding", "MobileNetV2 GAP", "Đặc trưng ngữ nghĩa bậc cao, 1-NN cao nhất; không cần thiết kế thủ công",
         "1280 chiều, khó giải thích; cần mô hình ONNX; học từ ImageNet chứ không riêng ảnh cây"],
    ], widths=[2.0, 3.4, 5.3, 5.3], caption="So sánh ưu và nhược điểm giữa các phương pháp trích rút", size=9.5)


# =============================================================================================
# Ket luan
# =============================================================================================
def conclusion(r: PtitReport, ctx: dict):
    fa, q = ctx["feature_analysis"], ctx["quality"]
    sets = fa["sets"]
    best = max(sets, key=lambda s: sets[s]["nn1_acc"])
    best_xa = max(sets, key=lambda s: sets[s]["nn1_acc_xauthor"])
    r.h(1, "PHẦN KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN")
    r.h(2, "Tóm tắt kết quả đạt được")
    r.p(f"Ở nội dung 1, nhóm xây dựng được bộ dữ liệu {ctx['db']['images']} ảnh cây thân gỗ trưởng thành mọc đơn lẻ, "
        f"cùng kích thước {config.IMAGE_SIZE}×{config.IMAGE_SIZE}, cây nằm gần giữa ảnh và được chụp ngang; có quy ước "
        "định danh, cấu trúc lưu trữ và lược đồ siêu dữ liệu thống nhất (nguồn, tác giả, giấy phép, chi, nhóm hình "
        "thái); được thống kê, đánh giá tính cân bằng và các hạn chế bằng số liệu. Điểm giống và khác giữa các cây được "
        "mô tả cả định tính (bảng đối sánh hình thái theo chi) lẫn định lượng (số đo theo chi và theo nhóm lá rộng/lá "
        "kim).")
    r.p(f"Ở nội dung 2, nhóm xây dựng bốn bộ đặc trưng gồm {sum(v['dim'] for v in sets.values())} chiều (màu sắc, hình "
        "dạng, kết cấu/bố cục, embedding sâu); mỗi khối được trình bày cơ sở toán học, ý nghĩa và được đo giá trị thông "
        f"tin (tỉ số Fisher, tỉ số tách, độ chính xác 1-NN). Mọi bộ đều vượt mức ngẫu nhiên ({_f(fa['nn1_chance'])}); "
        f"bộ có 1-NN cao nhất là {SET_VI[best].lower()} ({_f(sets[best]['nn1_acc'])}); khi loại ảnh cùng tác giả, bộ "
        f"cao nhất là {SET_VI[best_xa].lower()} ({_f(sets[best_xa]['nn1_acc_xauthor'])}).")
    r.h(2, "Hạn chế và hướng khắc phục")
    r.bullets([
        f"Khoảng {pc(q['crown_cut_top'] / q['n'], 0)} số ảnh có tán chạm mép trên do cắt giữa ảnh gốc dọc: chuyển sang "
        "cắt khung vuông quanh khung bao của cây hoặc co theo cạnh dài rồi đệm nền, sau đó trích rút lại đặc trưng.",
        "Kích thước cây trong khung chưa đồng đều: dùng cùng phép cắt quanh khung bao để cây chiếm tỉ lệ khung hình gần "
        "như nhau.",
        "Nhiều ảnh của cùng một cây: gom các ảnh cùng loạt chụp khi đánh giá (không tính ảnh cùng cây là kết quả đúng) "
        "hoặc chỉ giữ một vài ảnh mỗi cây.",
        f"{pc((q['n'] - q['labeled']) / q['n'], 0)} số ảnh chưa có nhãn chi và các chi mất cân bằng: bổ sung ảnh có "
        "nhãn cho các chi ít ảnh.",
    ])
    r.h(2, "Kế hoạch cho Báo cáo 2 (nội dung 3 và 4)")
    r.p("Ở giai đoạn tiếp theo, nhóm sẽ trình bày chi tiết thuật toán trích rút và lược đồ CSDL SQLite theo mô hình lai "
        "(siêu dữ liệu quan hệ + vector đặc trưng), các cấu trúc chỉ mục kd-tree và K-means; xây dựng hệ thống tìm 5 ảnh "
        "giống nhất với ảnh đầu vào theo khoảng cách tổng hợp ở mục 3.7, kèm sơ đồ khối, kết quả trung gian và đánh giá "
        "bằng các chỉ số P@5 và mAP. Phần mã nguồn cho các nội dung này đã có trong kho mã của nhóm.")


# =============================================================================================
def build(ctx: dict) -> Path:
    r = PtitReport()
    r.page_numbers()
    cover(r)
    r.toc()
    preface(r, ctx)
    part1(r, ctx)
    part2(r, ctx)
    part3(r, ctx)
    conclusion(r, ctx)
    return r.save(OUT_DOCX)


def finalize_with_word(docx: Path) -> Path | None:
    """Mo bang MS Word (COM): cap nhat muc luc + so trang, luu lai docx va xuat PDF. None neu khong co Word."""
    if sys.platform != "win32":
        return None
    pdf = docx.with_suffix(".pdf")
    esc = lambda p: str(p).replace("'", "''")  # noqa: E731
    ps = (
        "$ErrorActionPreference = 'Stop'\n"
        "$w = New-Object -ComObject Word.Application\n"
        "$w.Visible = $false\n"
        "$w.DisplayAlerts = 0\n"
        "try {\n"
        f"  $d = $w.Documents.Open('{esc(docx)}', $false, $false)\n"
        "  $d.Fields.Update() | Out-Null\n"
        "  foreach ($t in $d.TablesOfContents) { $t.Update() }\n"
        "  $d.Save()\n"
        f"  $d.ExportAsFixedFormat('{esc(pdf)}', 17)\n"
        "  $d.Close($false)\n"
        "} finally { $w.Quit() }\n"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], check=True,
                       capture_output=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[word] khong xuat duoc bang MS Word: {e}")
        return None
    return pdf if pdf.exists() else None


if __name__ == "__main__":
    ctx = load_context()
    if not ctx.get("feature_analysis"):
        sys.exit("Thieu results/feature_analysis.json - chay: python scripts/analyze_features.py")
    if not ctx.get("quality"):
        sys.exit("Thieu results/dataset_quality.json - chay: python scripts/analyze_dataset.py")
    out = build(ctx)
    print(f"[docx] {out} ({out.stat().st_size / 1024:.0f} KB)")
    pdf = finalize_with_word(out) or to_pdf(out)
    if pdf:
        print(f"[pdf] {pdf} ({pdf.stat().st_size / 1024:.0f} KB)")
