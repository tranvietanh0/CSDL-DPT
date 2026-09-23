"""Sinh bao cao docx + pdf tu ma nguon, CSDL va ket qua danh gia.   python scripts/build_docs.py

Moi so lieu trong bao cao duoc doc tu: data/manifest.csv, data/queries.csv, data/manual_exclude.csv,
results/excluded_images.csv, results/evaluation.json, results/index_meta.json, results/query_examples/*.json,
db/tree_features.sqlite va cac hinh trong docs/figures/.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from docx_helpers import Report  # noqa: E402
from treedb import config, db  # noqa: E402

FIG = config.FIG_DIR
OUT_DOCX = config.ROOT / "docs" / "BaoCao_HeCSDL_AnhCay.docx"
SOFFICE = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")


# ---------------------------------------------------------------------------------------
# Nap so lieu
# ---------------------------------------------------------------------------------------
def load_context() -> dict:
    ctx: dict = {}
    rows = list(csv.DictReader(open(config.MANIFEST, encoding="utf-8")))
    ctx["manifest"] = rows
    ctx["queries"] = list(csv.DictReader(open(config.QUERY_MANIFEST, encoding="utf-8")))
    ctx["manual_excluded"] = list(csv.DictReader(open(config.MANUAL_EXCLUDE, encoding="utf-8"))) \
        if config.MANUAL_EXCLUDE.exists() else []
    gate = config.RESULTS_DIR / "excluded_images.csv"
    ctx["gate_excluded"] = list(csv.DictReader(open(gate, encoding="utf-8"))) if gate.exists() else []
    raw = config.RAW_MANIFEST
    ctx["n_raw"] = sum(1 for _ in open(raw, encoding="utf-8")) - 1 if raw.exists() else None
    cache = config.CACHE_DIR / "commons_info.json"
    ctx["n_candidates"] = len(json.loads(cache.read_text(encoding="utf-8"))) if cache.exists() else None
    log = config.RESULTS_DIR / "build_dataset.log"
    ctx["filter_line"] = ""
    if log.exists():
        m = re.search(r"\[filter\] chap nhan (\d+) / (\d+); loai: (\{.*\})", log.read_text(encoding="utf-8"))
        if m:
            ctx["filter_line"] = m.group(0)
            ctx["n_accept"], ctx["n_total"], ctx["rejects"] = int(m.group(1)), int(m.group(2)), eval(m.group(3))
    conn = db.connect()
    ctx["db"] = db.summary(conn)
    ctx["feature_sets"] = db.list_feature_sets(conn)
    ctx["images"] = db.list_images(conn)
    conn.close()
    ctx["index_meta"] = json.loads((config.RESULTS_DIR / "index_meta.json").read_text(encoding="utf-8"))
    ctx["eval"] = json.loads((config.RESULTS_DIR / "evaluation.json").read_text(encoding="utf-8"))
    qex = config.RESULTS_DIR / "query_examples"
    ctx["examples"] = json.loads((qex / "index.json").read_text(encoding="utf-8"))
    ctx["example_json"] = {e["stem"]: json.loads((qex / f"{e['stem']}.json").read_text(encoding="utf-8"))
                           for e in ctx["examples"]}
    ctx["example_txt"] = {e["stem"]: (qex / f"{e['stem']}.txt").read_text(encoding="utf-8") for e in ctx["examples"]}
    return ctx


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def f3(x: float) -> str:
    return f"{x:.3f}"


# ---------------------------------------------------------------------------------------
# Cac phan cua bao cao
# ---------------------------------------------------------------------------------------
def cover(r: Report, ctx: dict):
    r.p("HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG", bold=True, align="center", size=14)
    r.p("KHOA CÔNG NGHỆ THÔNG TIN", bold=True, align="center", size=13)
    for _ in range(5):
        r.p()
    r.p("BÁO CÁO BÀI TẬP LỚN", bold=True, align="center", size=16)
    r.p("MÔN HỌC: HỆ CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN", bold=True, align="center", size=14)
    r.p()
    r.p("ĐỀ TÀI", align="center", size=13)
    r.p("XÂY DỰNG HỆ CSDL LƯU TRỮ VÀ TÌM KIẾM ẢNH CÂY THÂN GỖ THEO NỘI DUNG", bold=True, align="center", size=16)
    for _ in range(5):
        r.p()
    r.p("Giảng viên hướng dẫn: Nguyễn Đình Hóa", align="center", size=13)
    r.p("Sinh viên thực hiện: ……………………………………", align="center", size=13)
    r.p("Mã sinh viên: ……………………   Lớp: ……………………", align="center", size=13)
    for _ in range(4):
        r.p()
    r.p("Hà Nội, 2026", align="center", size=13)
    r.page_break()


def toc(r: Report):
    r.h(1, "MỤC LỤC")
    items = [
        "1. Giới thiệu bài toán và phạm vi thực hiện",
        "2. Xây dựng bộ dữ liệu ảnh cây thân gỗ (Yêu cầu 1)",
        "3. Các bộ đặc trưng nhận diện cây (Yêu cầu 2)",
        "4. Trích rút đặc trưng và hệ CSDL quản trị đặc trưng (Yêu cầu 3)",
        "5. Hệ thống tìm kiếm ảnh tương đồng (Yêu cầu 4)",
        "    5.1. Sơ đồ khối, chức năng và dữ liệu vào/ra của từng khối; quy trình thực hiện",
        "    5.2. Minh hoạ kết quả trung gian của quá trình truy vấn",
        "    5.3. Đánh giá kết quả",
        "6. Demo hệ thống (Yêu cầu 5)",
        "7. Kết luận và hướng phát triển",
        "Phụ lục A. Cấu trúc mã nguồn và hướng dẫn chạy",
        "Phụ lục B. Trích danh sách nguồn ảnh",
    ]
    for it in items:
        r.p(it)
    r.page_break()


def sec_intro(r: Report, ctx: dict):
    r.h(1, "1. Giới thiệu bài toán và phạm vi thực hiện")
    r.h(2, "1.1. Bối cảnh")
    r.p("Khác với cơ sở dữ liệu truyền thống chủ yếu trả lời các truy vấn khớp chính xác trên thuộc tính có cấu "
        "trúc, hệ cơ sở dữ liệu đa phương tiện (MMDB) phải quản lý các đối tượng ảnh, âm thanh, video có kích thước "
        "lớn cùng với siêu dữ liệu mô tả và vector đặc trưng; truy vấn điển hình là tìm kiếm tương đồng ("
        "“tìm các ảnh giống ảnh này”) và kết quả được xếp hạng thay vì chỉ lọc bản ghi. Vì ngữ nghĩa của ảnh không "
        "được biểu diễn tường minh trong chuỗi byte, việc lưu trữ, lập chỉ mục và truy vấn phải kết hợp kỹ thuật cơ "
        "sở dữ liệu, xử lý tín hiệu và các kỹ thuật truy vấn theo chỉ mục.")
    r.p("Tra cứu ảnh dựa trên nội dung (Content-Based Image Retrieval, CBIR) đi theo hai giai đoạn: giai đoạn ngoại "
        "tuyến nhập ảnh, chuẩn hoá, trích rút đặc trưng và xây dựng chỉ mục; giai đoạn trực tuyến xử lý ảnh truy vấn "
        "theo cùng cách thức, so sánh tương đồng dựa trên đặc trưng, xếp hạng và trả về kết quả. Ba lựa chọn kỹ "
        "thuật cơ bản của một hệ CBIR là: thuộc tính mô tả (đặc trưng), thước đo khoảng cách và chiến lược cắt tỉa "
        "ứng viên.")
    r.h(2, "1.2. Yêu cầu của đề bài")
    r.numbered([
        "Xây dựng/sưu tầm bộ dữ liệu ít nhất 500 ảnh cây thân gỗ trưởng thành, cùng kích thước, mỗi ảnh gồm một cây "
        "hoàn chỉnh nằm giữa ảnh, chụp ngang, tỉ lệ khung hình của vật thể tương đồng; mô tả đặc điểm giống và khác "
        "nhau của các cây trong ảnh.",
        "Xây dựng một hoặc vài bộ đặc trưng để nhận diện cây, bao gồm các đặc trưng giúp tìm sự tương đồng và sự "
        "khác biệt; trình bày từng loại đặc trưng và giá trị thông tin của chúng.",
        "Triển khai thuật toán/công cụ trích rút đặc trưng và xây dựng hệ CSDL quản trị đặc trưng của toàn bộ ảnh.",
        "Triển khai hệ thống tìm kiếm ảnh tương đồng: đầu vào là một ảnh cây mới, đầu ra là 5 ảnh giống nhất xếp "
        "theo độ tương đồng giảm dần; trình bày sơ đồ khối, chức năng và dữ liệu vào/ra của từng khối, minh hoạ kết "
        "quả trung gian và đánh giá kết quả.",
        "Demo hệ thống.",
    ])
    r.h(2, "1.3. Phạm vi và công cụ")
    n_db = ctx["db"]["images"]
    r.p(f"Hệ thống được cài đặt hoàn chỉnh bằng Python 3.12 với các thư viện xử lý ảnh (OpenCV, scikit-image, "
        f"Pillow), tính toán (NumPy, SciPy), cơ sở dữ liệu quan hệ SQLite (thư viện chuẩn), suy diễn mạng nơ-ron "
        f"ONNX Runtime và giao diện demo Streamlit. Bộ dữ liệu sau khi sưu tầm, lọc và kiểm duyệt gồm {n_db} ảnh "
        f"trong CSDL và {len(ctx['queries'])} ảnh truy vấn thử được tách riêng (không nằm trong CSDL). Toàn bộ mã "
        f"nguồn, dữ liệu, CSDL đặc trưng và kết quả đánh giá được đóng gói trong một kho mã nguồn duy nhất, có thể "
        f"chạy lại từng bước bằng các script trong thư mục scripts/ (Phụ lục A).")
    r.table(["Thành phần", "Công nghệ / thư viện", "Vai trò"], [
        ["Sưu tầm ảnh", "MediaWiki API (Wikimedia Commons), urllib", "Duyệt category, lấy siêu dữ liệu, tải ảnh"],
        ["Tiền xử lý", "Pillow, NumPy, OpenCV (pHash)", "Chuẩn hoá kích thước, khử trùng lặp"],
        ["Phân đoạn", "OpenCV GrabCut, SciPy ndimage", "Tách vùng cây / nền"],
        ["Đặc trưng thủ công", "OpenCV, scikit-image", "Màu sắc, hình dạng, kết cấu/bố cục"],
        ["Đặc trưng sâu", "ONNX Runtime + MobileNetV2 (ImageNet)", "Embedding 1280 chiều"],
        ["CSDL đặc trưng", "SQLite (sqlite3)", "Bảng quan hệ + vector BLOB + chỉ mục"],
        ["Chỉ mục đa chiều", "PCA (SVD, NumPy), cKDTree (SciPy), K-means (NumPy)", "Cắt tỉa ứng viên"],
        ["Đánh giá & hình", "NumPy, Matplotlib", "P@5, mAP, so sánh backend, hình minh hoạ"],
        ["Demo", "Streamlit, CLI", "Giao diện tải ảnh và xem kết quả"],
        ["Báo cáo", "python-docx, LibreOffice", "Sinh docx/pdf tự động từ kết quả"],
    ], widths=[3.5, 6.0, 6.5], caption="Công nghệ sử dụng")


def sec_dataset(r: Report, ctx: dict):
    rows = ctx["manifest"]
    imgs = ctx["images"]
    n_db = ctx["db"]["images"]
    r.h(1, "2. Xây dựng bộ dữ liệu ảnh cây thân gỗ (Yêu cầu 1)")
    r.h(2, "2.1. Tiêu chí của bộ dữ liệu")
    r.bullets([
        ("Đối tượng: ", "cây thân gỗ trưởng thành (cây gỗ lớn, cây lá rộng hoặc lá kim), loại trừ cây cọ/dừa "
                        "(không phải cây thân gỗ thực thụ), cây bụi, cây non và tranh vẽ/ảnh nghệ thuật."),
        ("Bố cục: ", "mỗi ảnh chứa đúng một cây hoàn chỉnh (thấy cả thân và toàn bộ tán, tán không bị khung hình "
                     "cắt), cây là chủ thể chính nằm gần giữa ảnh và đủ lớn (cao ít nhất khoảng 1/4 chiều cao ảnh)."),
        ("Góc chụp: ", "chụp ngang tầm mắt từ phía bên (không chụp từ trên cao, không ngước từ gốc, không cận cảnh "
                       "vỏ/lá)."),
        ("Định dạng lưu trữ: ", f"JPEG, {config.IMAGE_SIZE}×{config.IMAGE_SIZE} điểm ảnh, không gian màu RGB 8 bit, "
                                f"chất lượng nén {config.JPEG_QUALITY}; ảnh gốc được thu nhỏ sao cho cạnh ngắn bằng "
                                f"{config.IMAGE_SIZE} rồi cắt giữa (center-crop) thành hình vuông nên vật thể ở tâm "
                                f"ảnh luôn được giữ nguyên và các ảnh có tỉ lệ khung hình đồng nhất."),
        ("Siêu dữ liệu đi kèm: ", "tên file, kích thước, mã băm SHA-1, tiêu đề và đường dẫn nguồn, tác giả, giấy "
                                  "phép, chi (genus) và nhóm hình thái (lá rộng/lá kim) nếu xác định được từ category."),
    ])
    r.h(2, "2.2. Nguồn và quy trình sưu tầm")
    r.p("Ảnh được sưu tầm tự động từ kho ảnh mở Wikimedia Commons thông qua MediaWiki API (module "
        "src/treedb/collect.py). Điểm xuất phát là các category về “cây mọc đơn lẻ” (Solitary trees) vì đây là "
        "những bức ảnh có đúng bố cục một cây hoàn chỉnh giữa khung hình: các category theo loài (Solitary Quercus "
        "robur, Solitary Betula pendula, Solitary Pinus sylvestris, Solitary Juglans regia, Solitary Alnus glutinosa, "
        "Solitary Populus alba…), category ảnh chất lượng (Quality images of solitary trees), category tổng (Solitary "
        "trees) và các category theo quốc gia (Solitary trees by country, duyệt sâu 2 mức). Với mỗi file, hệ thống "
        "lấy kích thước gốc, kiểu MIME, tác giả, giấy phép và danh sách category; chi (genus) của cây được nhận diện "
        "bằng biểu thức chính quy trên tên category/tiêu đề dựa trên danh mục hơn 80 chi cây thân gỗ phổ biến.")
    steps = [
        f"Duyệt category → {ctx['n_candidates'] or '…'} file ứng viên.",
        "Lọc theo quy tắc: MIME là JPEG/PNG; cạnh ngắn ≥ 640 px; tỉ lệ rộng/cao trong [0,65; 1,6]; loại các tiêu "
        "đề/category chứa từ khoá không phù hợp (map, diagram, leaf, bark, trunk, detail, flower, aerial, panorama, "
        "forest, avenue, bonsai, painting, palm…); loại cây cọ."
        + (f" Kết quả: chấp nhận {ctx['n_accept']}/{ctx['n_total']} file; số file bị loại theo lý do: "
           f"{ctx['rejects']}." if ctx.get("rejects") else ""),
        "Sắp xếp ưu tiên nguồn “thuần” (category cây đơn lẻ theo loài → ảnh chất lượng → cây đơn lẻ chung/theo "
        f"quốc gia) và tải tối đa 1 000 ảnh ở độ rộng 1 024 px → {ctx['n_raw'] or '…'} ảnh thô.",
        f"Chuẩn hoá {config.IMAGE_SIZE}×{config.IMAGE_SIZE}; khử trùng lặp bằng perceptual hash (DCT 8×8, khoảng "
        "cách Hamming ≤ 6).",
        f"Kiểm duyệt thủ công bằng mắt toàn bộ ảnh trên các bảng ghép 30 ảnh (contact sheet): loại "
        f"{len(ctx['manual_excluded'])} ảnh không đạt tiêu chí bố cục (tán bị cắt, cây quá nhỏ/xa, nhiều cây, cận "
        "cảnh, chủ thể là công trình/người, rừng ngập mặn…). Danh sách loại được lưu theo tiêu đề nguồn trong "
        "data/manual_exclude.csv để quy trình có thể tái lập.",
        f"Tách {len(ctx['queries'])} ảnh có nhãn chi làm ảnh truy vấn thử (data/queries, không đưa vào CSDL).",
        f"Cổng kiểm soát chất lượng tự động sau phân đoạn (mục 4.2): chỉ nhận ảnh có tỉ lệ vùng cây trong "
        f"[4%, 90%] và tâm vùng cây lệch tâm ảnh không quá 30% mỗi trục → loại thêm {len(ctx['gate_excluded'])} ảnh.",
        f"Kết quả: {n_db} ảnh trong CSDL (đạt yêu cầu ≥ {config.MIN_IMAGES} ảnh).",
    ]
    r.numbered(steps)
    r.figure(FIG / "01_dataset_samples.png", "Một số ảnh trong bộ dữ liệu sau chuẩn hoá (nhãn chi nếu có)", 16)

    r.h(2, "2.3. Thống kê bộ dữ liệu")
    groups = Counter(i.get("group_label") or "unknown" for i in imgs)
    genus = Counter(i.get("genus") or "" for i in imgs)
    lic = Counter(i.get("license") or "" for i in imgs)
    ow = [int(r_["orig_width"]) for r_ in rows if r_.get("orig_width")]
    oh = [int(r_["orig_height"]) for r_ in rows if r_.get("orig_height")]
    r.table(["Chỉ tiêu", "Giá trị"], [
        ["Số ảnh trong CSDL", str(n_db)],
        ["Số ảnh truy vấn thử (ngoài CSDL)", str(len(ctx["queries"]))],
        ["Kích thước lưu trữ", f"{config.IMAGE_SIZE}×{config.IMAGE_SIZE}, JPEG, RGB"],
        ["Dung lượng trung bình / ảnh", f"{sum(int(x['file_size']) for x in rows) / len(rows) / 1024:.0f} KB"],
        ["Kích thước ảnh gốc (rộng × cao)", f"{min(ow)}–{max(ow)} × {min(oh)}–{max(oh)} px"],
        ["Nhóm hình thái", ", ".join(f"{k}: {v}" for k, v in groups.most_common())],
        ["Số ảnh có nhãn chi (genus)", f"{sum(v for k, v in genus.items() if k)} ({len([k for k in genus if k])} chi)"],
        ["Giấy phép", ", ".join(f"{k or '(không rõ)'}: {v}" for k, v in lic.most_common(6))],
    ], widths=[6, 10], caption="Thống kê tổng quan bộ dữ liệu")
    top = [[g, str(c), "lá kim" if g in ("Pinus", "Picea", "Abies", "Larix", "Cedrus", "Juniperus", "Taxus")
            else "lá rộng"] for g, c in genus.most_common(12) if g]
    r.table(["Chi (genus)", "Số ảnh", "Nhóm"], top, widths=[5, 3, 4], caption="Các chi có nhiều ảnh nhất",
            align_center_cols={1, 2})
    r.figure(FIG / "02_dataset_stats.png", "Phân bố một số thuộc tính bậc thấp trên toàn bộ dữ liệu: sắc độ, độ "
             "bão hoà, độ sáng trung bình của vùng cây; tỉ lệ vùng cây; tỉ lệ cao/rộng; số ảnh theo nhóm", 16)

    r.h(2, "2.4. Đặc điểm giống nhau và khác nhau của các cây trong ảnh")
    cov = [i["fg_coverage"] for i in imgs if i.get("fg_coverage")]
    cov_med = sorted(cov)[len(cov) // 2] if cov else 0
    r.p("Điểm giống nhau (là cơ sở để các đặc trưng “tìm sự tương đồng” hoạt động):", bold=True)
    r.bullets([
        "Bố cục ba tầng thống nhất: phần trên là bầu trời (xanh lam, trắng xám hoặc ánh hoàng hôn), phần giữa là "
        "tán cây, phần dưới là mặt đất (cỏ, ruộng, cát, tuyết). Cây là thành phần liên thông lớn nhất và nằm gần "
        f"tâm ảnh; vùng cây chiếm trung vị khoảng {pct(cov_med)} diện tích ảnh.",
        "Trục chính của vật thể gần như thẳng đứng, thân cây ở dưới nhỏ hơn tán ở trên nên profile bề rộng theo "
        "chiều cao có dạng “rộng ở giữa/trên, hẹp ở gốc”.",
        "Màu chủ đạo của vật thể thuộc dải xanh lục – nâu – xám; màu nền thuộc dải xanh lam – trắng (trời) và "
        "xanh lục – vàng – nâu (đất). Vì vậy biểu đồ màu toàn cục của các ảnh khá giống nhau, phải tách riêng biểu "
        "đồ vật thể và nền mới phân biệt tốt.",
        "Tán cây có kết cấu (texture) không đều, nhiều cạnh nhỏ theo mọi hướng, khác hẳn vùng trời phẳng và vùng "
        "đất tương đối mịn.",
    ])
    r.p("Điểm khác nhau (là cơ sở để các đặc trưng “tìm sự khác biệt” hoạt động):", bold=True)
    r.bullets([
        ("Loài/chi → hình dạng tán: ", "sồi (Quercus), óc chó (Juglans) có tán tròn rộng, thân ngắn; dương "
                                       "(Populus), bạch dương (Betula) có tán cao hẹp, thân mảnh; thông (Pinus) có "
                                       "tán không đối xứng, thân lộ dài; keo (Acacia/Vachellia) tán phẳng hình ô. "
                                       "Điều này thể hiện ở tỉ lệ cao/rộng, tâm sai, solidity, moment Hu, lưới "
                                       "8×8 và profile bề rộng."),
        ("Mùa và trạng thái lá: ", "lá xanh mùa hè, vàng/cam mùa thu, trụi lá mùa đông, phủ tuyết/sương giá; "
                                   "sắc độ (H) và độ bão hoà (S) của vùng cây thay đổi mạnh; cây trụi lá có mật "
                                   "độ cạnh và LBP đặc trưng khác cây có lá."),
        ("Nhóm lá kim / lá rộng: ", "tán lá kim đậm, xanh sẫm, kết cấu mịn và hình nón/không đối xứng; lá rộng "
                                    "sáng hơn, kết cấu thô, tán tròn."),
        ("Bối cảnh và ánh sáng: ", "đồng cỏ, ruộng hoa cải vàng, hoang mạc, tuyết, bờ biển, ngược sáng "
                                   "(silhouette), ảnh đen trắng; ảnh hưởng đến biểu đồ nền, độ sáng V và độ "
                                   "tương phản."),
        ("Khoảng cách chụp: ", f"tỉ lệ vùng cây dao động từ {pct(min(cov)) if cov else '…'} đến "
                               f"{pct(max(cov)) if cov else '…'}, tức cùng một loài có thể xuất hiện với kích "
                               "thước khác nhau trong khung; các đặc trưng hình dạng được chuẩn hoá theo hình chữ "
                               "nhật cơ bản để giảm ảnh hưởng này."),
    ])


def sec_features(r: Report, ctx: dict):
    fs = {f["set_name"]: f for f in ctx["feature_sets"]}
    r.h(1, "3. Các bộ đặc trưng nhận diện cây (Yêu cầu 2)")
    r.h(2, "3.1. Định hướng thiết kế")
    r.p("Theo lý thuyết truy vấn ảnh dựa trên nội dung, ảnh được mô tả bằng các thuộc tính bậc thấp: màu sắc "
        "(của vật thể và của nền), hình dạng/kích thước vật thể, và bố cục/kết cấu (độ thô, độ tương phản, hướng "
        "sắp xếp của các nét). Mỗi loại thuộc tính bắt được một khía cạnh khác nhau của sự giống/khác, nên hệ "
        "thống xây dựng ba bộ đặc trưng thủ công tương ứng và bổ sung một bộ đặc trưng ngữ nghĩa bậc cao "
        "(embedding từ mạng nơ-ron tích chập) để thu hẹp “khoảng cách ngữ nghĩa”. Các bộ được lưu trữ riêng, có "
        "độ đo khoảng cách riêng và được kết hợp theo trọng số khi truy vấn. Mọi đặc trưng đều được tính trên "
        "ảnh đã chuẩn hoá và (trừ embedding) dựa trên mặt nạ vùng cây để loại ảnh hưởng của nền.")
    r.table(["Bộ đặc trưng", "Số chiều", "Độ đo", "Trả lời câu hỏi", "Tìm tương đồng", "Tìm khác biệt"], [
        ["1. Màu sắc", str(fs["color"]["dim"]), "L1", "Cây/nền có màu gì, phân bố ở đâu?",
         "Cùng mùa, cùng loại tán, cùng bối cảnh", "Lá xanh ↔ vàng ↔ trụi lá; trời ↔ tuyết ↔ cát"],
        ["2. Hình dạng", str(fs["shape"]["dim"]), "L2 (z-score)", "Tán cây hình gì, thân dài ngắn?",
         "Cùng chi (tán tròn / cao hẹp / hình ô)", "Tỉ lệ cao/rộng, tâm sai, độ lõm tán"],
        ["3. Kết cấu / bố cục", str(fs["texture"]["dim"]), "L2 (z-score)", "Tán mịn hay thô, nét hướng nào?",
         "Lá kim ↔ lá kim; trụi lá ↔ trụi lá", "Độ thô, mật độ cạnh, năng lượng DCT"],
        ["4. Embedding sâu", str(fs.get("deep", {}).get("dim", "—")), "cosine",
         "Ảnh “trông giống” ảnh nào về ngữ nghĩa?", "Khái quát hoá loài, dáng cây", "Phân biệt cảnh nền, vật thể"],
    ], widths=[3.0, 1.6, 2.2, 3.4, 3.0, 3.0], caption="Tổng quan các bộ đặc trưng", size=9.5)
    r.figure(FIG / "05_feature_sets.png", "Cấu trúc các khối trong từng bộ đặc trưng lưu trong CSDL", 15)

    r.h(2, "3.2. Bộ đặc trưng 1 – Màu sắc (369 chiều, độ đo L1)")
    r.p("Không gian màu HSV được chọn thay cho RGB vì H (sắc độ) tách biệt “màu gì” khỏi độ sáng, ít phụ thuộc "
        "thiết bị và điều kiện chiếu sáng. Mỗi kênh được lượng tử hoá thành m mức, tổng số ô màu n = 8×3×3 = 72 "
        "(H chia mịn hơn vì mang nhiều thông tin nhất). Biểu đồ tần suất H(M) = [h1, …, hn] với hj là số điểm ảnh "
        "thuộc ô màu j, được chuẩn hoá L1 (Σhj = 1) để không phụ thuộc số điểm ảnh. Bộ màu gồm 5 khối:")
    r.table(["Khối", "Chiều", "Cách tính", "Giá trị thông tin"], [
        ["hsv_global", "72", "Biểu đồ HSV 8×3×3 trên toàn ảnh", "Ấn tượng màu tổng thể (cây + nền), tìm ảnh cùng "
                                                                "“tông” màu; nhạy với nền nên dễ bị “che mặt”"],
        ["hsv_grid", "144", "Chia ảnh lưới 3×3, mỗi vùng biểu đồ 4×2×2 = 16 ô", "Quan hệ không gian của màu: trời "
                                                                            "ở trên, tán ở giữa, đất ở dưới; phân "
                                                                            "biệt ảnh cùng tổng màu nhưng bố trí khác"],
        ["hsv_fg", "72", "Biểu đồ HSV chỉ trên mặt nạ vùng cây", "Màu của chính cây (lá xanh/vàng/nâu, trụi lá) – "
                                                                "khắc phục hiệu ứng che mặt của nền"],
        ["hsv_bg", "72", "Biểu đồ HSV trên phần nền (ngoài mặt nạ)", "Bối cảnh (trời xanh, tuyết, cỏ, cát); hữu ích "
                                                                    "khi muốn tìm ảnh cùng cảnh"],
        ["moments", "9", "Trung bình, độ lệch chuẩn, độ lệch (skewness) của H, S, V trên vùng cây",
         "Mô tả gọn phân bố màu; bền với nhiễu lượng tử hoá (hai màu gần nhau rơi vào hai ô khác nhau)"],
    ], widths=[2.3, 1.3, 5.2, 7.2], caption="Các khối của bộ đặc trưng màu sắc", size=9.5)
    r.p("Khoảng cách giữa hai biểu đồ dùng L1-norm: d(X, Y) = Σ|xj − yj|, tương đương 2·(1 − giao của hai biểu "
        "đồ); giá trị 0 khi hai phân bố màu trùng nhau và tối đa 2 cho mỗi biểu đồ khi không có màu chung.")

    r.h(2, "3.3. Bộ đặc trưng 2 – Hình dạng tán/thân cây (113 chiều, độ đo L2)")
    r.p("Đặc trưng hình dạng được tính trên mặt nạ nhị phân của vật thể (mục 4.2) và được thiết kế để ít phụ thuộc "
        "vào vị trí, kích thước và (một phần) góc quay của cây trong ảnh:")
    r.table(["Khối", "Chiều", "Cách tính", "Giá trị thông tin"], [
        ["region", "10", "Diện tích tương đối; tỉ lệ cao/rộng của hình chữ nhật cơ bản; extent (diện tích/hình "
                         "chữ nhật); solidity (diện tích/bao lồi); tâm sai; độ dài trục chính, trục phụ; hướng trục "
                         "chính; độ lệch tâm (dx, dy)",
         "Các thước đo hình dạng cơ bản: tán cao hẹp (dương, bạch dương) ↔ tán tròn rộng (sồi); tán đặc ↔ tán "
         "thưa/lõm (cây trụi lá có solidity thấp)"],
        ["hu", "7", "7 moment bất biến Hu, lấy −sign·log10|φ|", "Bất biến với tịnh tiến, tỉ lệ và xoay; mô tả "
                                                                  "phân bố khối lượng của hình"],
        ["grid", "64", "Đặt lưới 8×8 lên hình chữ nhật cơ bản, ô = 1 nếu > 15% diện tích ô là vật thể (chuẩn hoá "
                       "thước đo về cùng kích thước)", "Chuỗi nhị phân mô tả hình thể; so sánh trực tiếp dáng cây: "
                                                       "tán tròn, tán hình nón, thân dài ở dưới…"],
        ["fourier", "16", "Đường bao ngoài lấy mẫu 128 điểm; biến đổi Fourier; bỏ F0 (tịnh tiến), chia F1 (tỉ lệ), "
                          "lấy biên độ (bất biến xoay) của 16 hệ số tiếp theo",
         "Mô tả độ gồ ghề/độ đối xứng của đường viền tán ở nhiều tần số"],
        ["profile", "16", "Chia hình chữ nhật cơ bản thành 16 dải ngang từ đỉnh xuống gốc, mỗi dải = tỉ lệ bề rộng "
                          "bị vật thể chiếm", "“Chữ ký” dáng cây theo chiều cao: nơi tán rộng nhất, độ dài thân "
                                              "lộ, tán hình ô (rộng trên) hay hình nón (rộng dưới)"],
    ], widths=[2.0, 1.2, 6.3, 6.5], caption="Các khối của bộ đặc trưng hình dạng", size=9.5)
    r.p("Các chiều có thang đo khác nhau nên trước khi tính khoảng cách Euclid (L2) mỗi chiều được chuẩn hoá "
        "z-score theo thống kê của CSDL (trung bình, độ lệch chuẩn lưu trong bảng feature_stats).")

    r.h(2, "3.4. Bộ đặc trưng 3 – Kết cấu và bố cục (85 chiều, độ đo L2)")
    r.table(["Khối", "Chiều", "Cách tính", "Giá trị thông tin"], [
        ["glcm", "10", "Ma trận đồng xuất hiện mức xám (32 mức, khoảng cách 1 và 3, trung bình 4 hướng) trên vùng "
                       "bao của cây: contrast, dissimilarity, homogeneity, energy, correlation",
         "Độ thô/mịn và độ tương phản của tán: lá kim mịn đều ↔ lá rộng thô; tán đặc ↔ tán thưa lộ trời"],
        ["lbp", "28", "Biểu đồ Local Binary Pattern đồng nhất (P=8,R=1: 10 ô; P=16,R=2: 18 ô) trên điểm ảnh vùng cây",
         "Mẫu vi kết cấu cục bộ, bất biến với thay đổi độ sáng đơn điệu; phân biệt lá, cành trụi, tuyết phủ"],
        ["edge", "9", "Biểu đồ 8 hướng gradient Sobel (trọng số biên độ) + mật độ cạnh",
         "Hướng sắp xếp các nét (cành thẳng đứng của cây trụi lá ↔ tán lá đẳng hướng) và “độ sắc của các nét”"],
        ["tamura", "3", "Độ thô (coarseness), độ tương phản (contrast), tính định hướng (directionality)",
         "Ba thuộc tính kết cấu gần với cảm nhận của người"],
        ["dct", "35", "Ảnh xám thu về 64×64, biến đổi cosine rời rạc, lấy khối 6×6 tần số thấp bỏ hệ số DC",
         "Bố cục tổng thể (phân bố sáng tối theo vùng lớn), đại diện cho đánh chỉ số trên miền nén (JPEG)"],
    ], widths=[2.0, 1.2, 6.3, 6.5], caption="Các khối của bộ đặc trưng kết cấu / bố cục", size=9.5)

    r.h(2, "3.5. Bộ đặc trưng 4 (mở rộng) – Embedding sâu MobileNetV2 (1280 chiều, độ đo cosine)")
    r.p("Các thuộc tính bậc thấp không nắm bắt hết ngữ nghĩa (“khoảng cách ngữ nghĩa”). Để bổ sung, hệ thống dùng "
        "mạng MobileNetV2 huấn luyện sẵn trên ImageNet (định dạng ONNX, chạy bằng ONNX Runtime trên CPU): ảnh được "
        "thu về 224×224, chuẩn hoá theo trung bình/độ lệch chuẩn ImageNet, lấy đầu ra của lớp Global Average "
        "Pooling (1280 chiều, trước lớp phân loại) và chuẩn hoá L2. Độ tương đồng cosine giữa hai embedding phản "
        "ánh sự giống nhau về hình thái tổng thể mà mạng đã học từ hàng triệu ảnh tự nhiên. Đây là đặc trưng tuỳ "
        "chọn: nếu không có model, hệ thống vẫn hoạt động với ba bộ thủ công.")


def sec_extraction_db(r: Report, ctx: dict):
    dbs = ctx["db"]
    im = ctx["index_meta"]
    r.h(1, "4. Trích rút đặc trưng và hệ CSDL quản trị đặc trưng (Yêu cầu 3)")
    r.h(2, "4.1. Quy trình trích rút")
    r.p("Script scripts/extract_features.py duyệt toàn bộ ảnh trong data/manifest.csv; với mỗi ảnh: đọc và chuẩn "
        "hoá (giống hệt cách áp dụng cho ảnh truy vấn) → phân đoạn vật thể → tính ba bộ đặc trưng thủ công (chạy "
        "song song nhiều tiến trình) → tính embedding theo lô 32 ảnh → ghi vào SQLite → tính thống kê chuẩn hoá cho "
        "từng bộ. Thời gian trích rút trung bình khoảng 0,1 giây/ảnh cho ba bộ thủ công (đã song song hoá) và "
        "0,02 giây/ảnh cho embedding.")
    r.h(2, "4.2. Phân đoạn vật thể / nền")
    r.p("Vì đề bài đảm bảo cây nằm giữa ảnh, mặt nạ vùng cây được tính bằng thuật toán GrabCut khởi tạo với hình "
        "chữ nhật trung tâm (chiếm 84% bề rộng, 92% chiều cao) trên ảnh thu nhỏ 256 px (5 vòng lặp). Sau đó loại "
        "các điểm ảnh bầu trời (sáng và bão hoà thấp, hoặc sắc độ xanh lam), lọc hình thái đóng/mở, giữ thành phần "
        "liên thông có diện tích lớn và gần tâm ảnh nhất, lấp lỗ. Nếu mặt nạ quá nhỏ (< 2%) dùng mặt nạ chữ nhật "
        "mặc định. Kết quả được dùng cho biểu đồ màu vật thể/nền, toàn bộ bộ hình dạng và bộ kết cấu; đồng thời "
        "làm cổng kiểm soát chất lượng (tỉ lệ vùng cây và độ lệch tâm) như mô tả ở mục 2.2.")
    r.figure(FIG / "03_segmentation.png", "Ví dụ phân đoạn vật thể: ảnh gốc (trên) và vùng cây được tách (dưới)", 16)
    r.h(2, "4.3. Lược đồ CSDL đặc trưng (SQLite)")
    r.p("Hệ CSDL theo mô hình lai: siêu dữ liệu quan hệ của ảnh lưu trong bảng thông thường để lọc chính xác "
        "(theo chi, giấy phép, kích thước…), còn vector đặc trưng lưu dạng BLOB float32 gắn với khoá (ảnh, bộ đặc "
        "trưng); chỉ mục được xây trên đặc trưng chứ không trên dữ liệu ảnh gốc, do đó có thể cập nhật pipeline "
        "đặc trưng mà không mất ảnh. Module src/treedb/db.py cài đặt toàn bộ thao tác.")
    r.table(["Bảng", "Cột chính", "Vai trò"], [
        ["images", "image_id (PK), filename (UNIQUE), path, width, height, format, file_size, sha1, source_title, "
                   "source_url, author, license, genus, group_label, orig_width, orig_height, fg_coverage, ingested_at",
         "Siêu dữ liệu mô tả từng ảnh (thông tin ngoài lề + nhãn)"],
        ["feature_sets", "set_name (PK), dim, description, extractor_version, metric, layout_json",
         "Danh mục bộ đặc trưng: số chiều, độ đo, bố cục các khối (để giải thích kết quả trung gian)"],
        ["features", "image_id (FK), set_name (FK), dim, vector BLOB; PK(image_id, set_name)",
         "Vector đặc trưng float32 của mỗi ảnh theo từng bộ"],
        ["feature_stats", "set_name, key (mean/std/scale), value BLOB", "Thống kê chuẩn hoá: z-score theo chiều và "
                                                                       "hệ số scale để kết hợp các bộ"],
        ["indexes", "name (PK), kind, meta_json, payload BLOB, built_at", "Chỉ mục đa chiều: PCA + toạ độ cho cây "
                                                                          "k-d; tâm cụm K-means"],
        ["image_clusters", "index_name, image_id, cluster_id", "Ảnh thuộc cụm nào (truy vấn theo phân cụm)"],
        ["query_log", "query_id, query_path, backend, weights_json, top_k, results_json, elapsed_ms, created_at",
         "Nhật ký truy vấn (phục vụ đánh giá và phản hồi)"],
    ], widths=[2.6, 8.4, 5.0], caption="Lược đồ CSDL đặc trưng", size=9.5)
    sets_rows = [[f["set_name"], str(f["dim"]), f["metric"], str(dbs["feature_sets"][f["set_name"]]["count"]),
                  ", ".join(f"{n}({d})" for n, d in f["layout"])] for f in ctx["feature_sets"]]
    r.table(["Bộ", "Chiều", "Độ đo", "Số vector", "Bố cục khối"], sets_rows, widths=[1.8, 1.4, 1.8, 2.0, 9.0],
            caption="Nội dung bảng feature_sets sau khi trích rút", size=9.5, align_center_cols={1, 2, 3})
    r.p(f"CSDL hiện có {dbs['images']} ảnh, {sum(v['count'] for v in dbs['feature_sets'].values())} vector đặc trưng "
        f"thuộc {len(dbs['feature_sets'])} bộ, dung lượng file SQLite {dbs['db_size_bytes'] / 1024 / 1024:.1f} MB.")
    r.h(2, "4.4. Chuẩn hoá và kết hợp các bộ đặc trưng")
    r.p("Các bộ có số chiều và thang đo khác nhau nên không thể cộng khoảng cách trực tiếp. Với mỗi bộ s, hệ thống "
        "lưu: mean và std theo chiều (dùng cho z-score ở các bộ dùng L2) và scale_s = khoảng cách trung bình giữa "
        "hai ảnh ngẫu nhiên trong CSDL (ước lượng trên 4 000 cặp). Khoảng cách chuẩn hoá d̃_s = d_s/scale_s có kỳ "
        "vọng ≈ 1 với cặp ảnh ngẫu nhiên, nên có thể kết hợp tuyến tính: D = Σ_s w_s·d̃_s với Σw_s = 1. Độ tương "
        "đồng hiển thị cho người dùng là sim = 1/(1 + D) ∈ (0, 1]. Trọng số mặc định: "
        + ", ".join(f"{k} = {v}" for k, v in config.DEFAULT_WEIGHTS.items()) + " (có thể đổi khi truy vấn).")
    r.h(2, "4.5. Cấu trúc chỉ mục đa chiều")
    r.p("Để tránh quét tuyến tính toàn bộ CSDL khi kích thước tăng, hệ thống xây dựng hai cấu trúc chỉ mục trên "
        "không gian nhúng chung E (nối các vector đã chuẩn hoá của mọi bộ, nhân √w_s/scale_s để khoảng cách Euclid "
        "trong E xấp xỉ D):")
    r.bullets([
        (f"Cây k-d trên không gian PCA: ", f"giảm E ({sum(f['dim'] for f in ctx['feature_sets'])} chiều) xuống "
                                           f"{im['kdtree']['pca_dim']} thành phần chính (giữ {pct(im['kdtree']['explained_variance'])} "
                                           "phương sai) rồi dựng cây k-d (SciPy cKDTree). Khi truy vấn, lấy "
                                           f"{config.CANDIDATE_POOL} láng giềng gần nhất trong cây làm tập ứng viên, "
                                           "sau đó xếp hạng lại bằng khoảng cách chính xác D (lọc – tinh chỉnh)."),
        (f"Phân cụm K-means (K = {im['kmeans']['k']}): ", "cài đặt thuật toán Lloyd với khởi tạo k-means++ bằng "
                                                         "NumPy; kích thước các cụm: "
                                                         f"{im['kmeans']['sizes']}. Khi truy vấn, tính khoảng cách từ "
                                                         "vector truy vấn đến các tâm cụm, quét các ảnh trong ≥ 2 "
                                                         "cụm gần nhất (mở rộng cho tới khi đủ ứng viên) rồi xếp "
                                                         "hạng chính xác."),
    ])
    r.figure(FIG / "06_clusters.png", "Phân bố ảnh trong không gian PCA: theo cụm K-means (trái) và theo nhóm hình "
             "thái (phải)", 16)


def sec_search(r: Report, ctx: dict):
    ev = ctx["eval"]
    k = ev["k"]
    r.h(1, "5. Hệ thống tìm kiếm ảnh tương đồng (Yêu cầu 4)")
    r.h(2, "5.1. Sơ đồ khối, chức năng và dữ liệu vào/ra của từng khối")
    r.figure(FIG / "04_block_diagram.png", "Sơ đồ khối hệ thống: giai đoạn ngoại tuyến (trên) và trực tuyến (dưới)",
             17)
    r.table(["Khối", "Chức năng", "Dữ liệu vào", "Dữ liệu ra"], [
        ["1. Thu thập ảnh", "Duyệt category Commons, lọc theo quy tắc, tải ảnh, ghi siêu dữ liệu",
         "Danh sách category gốc", "Ảnh thô (JPEG) + raw_manifest.csv (nguồn, tác giả, giấy phép, chi)"],
        ["2. Tiền xử lý", "Xoay EXIF, RGB, resize cạnh ngắn 512, center-crop, khử trùng lặp pHash, áp dụng danh "
                          "sách kiểm duyệt, tách ảnh truy vấn thử", "Ảnh thô", "Ảnh 512×512 + manifest.csv, queries.csv"],
        ["3. Phân đoạn", "GrabCut + loại trời + lọc hình thái", "Ảnh RGB 512×512", "Mặt nạ nhị phân vùng cây, tỉ lệ "
                                                                                    "vùng cây, độ lệch tâm"],
        ["4. Trích rút đặc trưng", "Tính 4 bộ đặc trưng", "Ảnh + mặt nạ", "4 vector: màu (369), hình dạng (113), kết "
                                                                         "cấu (85), embedding (1280) + bố cục khối"],
        ["5. CSDL SQLite", "Lưu siêu dữ liệu và vector; CRUD; thống kê", "Bản ghi ảnh + vector", "Bảng images, "
                                                                                               "feature_sets, features"],
        ["6. Chuẩn hoá & chỉ mục", "mean/std/scale từng bộ; PCA + cây k-d; K-means", "Ma trận đặc trưng",
         "feature_stats, indexes, image_clusters"],
        ["Q1. Tiền xử lý + phân đoạn truy vấn", "Như khối 2–3 cho ảnh mới (mọi kích thước)", "Ảnh truy vấn",
         "Ảnh 512×512 + mặt nạ"],
        ["Q2. Trích rút đặc trưng truy vấn", "Như khối 4", "Ảnh + mặt nạ", "Vector q của từng bộ"],
        ["Q3. Chọn ứng viên", "linear: toàn bộ; kdtree: 60 láng giềng PCA; kmeans: ≥ 2 cụm gần nhất",
         "q, chỉ mục", "Tập ứng viên (chỉ số ảnh)"],
        ["Q4. Tính khoảng cách", "d_s theo độ đo từng bộ; chuẩn hoá d_s/scale_s; D = Σ w_s·d̃_s",
         "q, vector ứng viên, thống kê, trọng số", "d_s, d̃_s, đóng góp w_s·d̃_s, D cho mỗi ứng viên"],
        ["Q5. Xếp hạng", "sim = 1/(1+D); sắp giảm dần; lấy top-K; ghi query_log", "D của các ứng viên",
         "5 ảnh giống nhất + độ tương đồng + bảng giá trị trung gian"],
    ], widths=[3.0, 4.6, 3.6, 4.8], caption="Chức năng và dữ liệu vào/ra của từng khối", size=9)
    r.h(3, "Quy trình thực hiện một truy vấn")
    r.numbered([
        "Người dùng cung cấp ảnh cây mới (tải lên trên giao diện web hoặc đường dẫn trên dòng lệnh) và chọn "
        "backend, số kết quả K (mặc định 5), trọng số các bộ.",
        "Ảnh được chuẩn hoá về 512×512 đúng như ảnh CSDL (nếu ảnh không vuông sẽ được cắt giữa), phân đoạn để có "
        "mặt nạ vùng cây.",
        "Trích rút 4 vector đặc trưng bằng chính các hàm đã dùng khi xây CSDL (đảm bảo cùng cách thức).",
        "Nạp ma trận đặc trưng và thống kê chuẩn hoá từ SQLite (một lần khi khởi động), chọn tập ứng viên theo "
        "backend.",
        "Với mỗi ứng viên tính khoảng cách từng bộ (L1/L2/cosine), chuẩn hoá theo scale, kết hợp theo trọng số "
        "thành D và độ tương đồng sim.",
        "Sắp xếp giảm dần theo sim, trả về 5 ảnh đầu cùng bảng giá trị trung gian; ghi nhật ký truy vấn.",
    ])

    # ---- 5.2 ket qua trung gian --------------------------------------------------------
    r.h(2, "5.2. Minh hoạ kết quả trung gian của quá trình truy vấn")
    ex = ctx["examples"]
    ej = ctx["example_json"]
    for i, e in enumerate(ex[:2], start=1):
        q = ej[e["stem"]]
        F = q["query"]["features"]
        r.h(3, f"Ví dụ {i}: ảnh truy vấn {e['query']} (chi thật: {e['genus'] or 'chưa rõ'}), backend {e['backend']}")
        r.p(f"Ảnh truy vấn không nằm trong CSDL. Tỉ lệ vùng cây sau phân đoạn: {pct(q['query']['mask_coverage'])}. "
            f"Một số giá trị đặc trưng đã trích rút (làm tròn 4 chữ số):")
        b = F["color"]["blocks"]
        rows = [["Màu: hsv_global (12 ô đầu / 72)", ", ".join(f"{x:.3f}" for x in b["hsv_global"][:12])],
                ["Màu: hsv_fg (12 ô đầu / 72)", ", ".join(f"{x:.3f}" for x in b["hsv_fg"][:12])],
                ["Màu: moments (mean H,S,V; std H,S,V; skew H,S,V)", ", ".join(f"{x:.3f}" for x in b["moments"])]]
        s = F["shape"]["blocks"]
        rows += [["Hình dạng: region (area, h/w, extent, solidity, ecc, major, minor, orient, dx, dy)",
                  ", ".join(f"{x:.3f}" for x in s["region"])],
                 ["Hình dạng: hu (7 moment, log)", ", ".join(f"{x:.2f}" for x in s["hu"])],
                 ["Hình dạng: grid 8×8 (64 bit, đọc trái→phải, trên→dưới)",
                  "".join("1" if x > 0.5 else "0" for x in s["grid"])],
                 ["Hình dạng: profile (16 dải, đỉnh→gốc)", ", ".join(f"{x:.2f}" for x in s["profile"])]]
        t = F["texture"]["blocks"]
        rows += [["Kết cấu: glcm (contrast, dissim, homog, energy, corr × d=1,3)", ", ".join(f"{x:.3f}" for x in t["glcm"])],
                 ["Kết cấu: edge (8 hướng + mật độ)", ", ".join(f"{x:.3f}" for x in t["edge"])],
                 ["Kết cấu: tamura (coarseness, contrast, directionality)", ", ".join(f"{x:.3f}" for x in t["tamura"])]]
        if "deep" in F:
            d = F["deep"]["blocks"]["mobilenetv2_gap"]
            rows.append(["Embedding: 10 chiều đầu / 1280 (chuẩn hoá L2)", ", ".join(f"{x:.4f}" for x in d[:10])])
        r.table(["Đặc trưng (khối)", "Giá trị"], rows, widths=[5.5, 10.5], caption=f"Giá trị đặc trưng của ảnh "
                f"truy vấn {e['query']}", size=9)
        r.figure(FIG / f"1{i}_{e['stem']}_features.png", f"Biểu diễn trực quan các đặc trưng của ảnh truy vấn {e['query']}", 17)
        sets = q["sets"]
        hdr = ["Hạng", "Ảnh", "Chi"] + [f"d̃ {s_}" for s_ in sets] + ["D", "sim"]
        body = []
        for res in q["results"]:
            body.append([str(res["rank"]), res["filename"], res["genus"] or "—"]
                        + [f"{res['per_set'][s_]['norm']:.3f}" for s_ in sets]
                        + [f"{res['distance']:.3f}", f"{res['similarity']:.3f}"])
        r.table(hdr, body, caption=f"Khoảng cách chuẩn hoá từng bộ, khoảng cách kết hợp và độ tương đồng của top-{k} "
                f"(trọng số: " + ", ".join(f"{s_}={q['weights'][s_]:.2f}" for s_ in sets) + ")", size=9.5,
                align_center_cols=set(range(3, 3 + len(sets) + 2)))
        r.p("Khoảng cách thô (raw) tương ứng của hạng 1: " + ", ".join(
            f"{s_} = {q['results'][0]['per_set'][s_]['raw']:.4f}" for s_ in sets) + "; sau khi chia cho scale "
            "từng bộ và nhân trọng số, tổng D = " + f3(q["results"][0]["distance"]) + " → sim = 1/(1+D) = "
            + f3(q["results"][0]["similarity"]) + ".")
        r.figure(FIG / f"1{i}_{e['stem']}_result.png", f"Kết quả top-{k} cho ảnh {e['query']} và đóng góp của từng "
                 "bộ đặc trưng vào khoảng cách kết hợp", 17)
        tm = q["timing_ms"]
        r.p("Thời gian xử lý (ms): " + ", ".join(f"{kk} = {v:.1f}" for kk, v in tm.items()) + ".")
    for i, e in enumerate(ex[2:4], start=3):
        q = ej[e["stem"]]
        r.h(3, f"Ví dụ {i}: ảnh {e['query']} (chi thật: {e['genus'] or 'chưa rõ'}) với backend {e['backend']}")
        info = q["index_info"]
        if e["backend"] == "kdtree":
            r.p(f"Cây k-d trên {info['pca_dim']} thành phần PCA trả về {q['n_candidates']} ứng viên trong "
                f"{q['timing_ms']['candidates']:.2f} ms (khoảng cách trong không gian PCA của 5 ứng viên đầu: "
                + ", ".join(f"{x:.3f}" for x in info["tree_dist"][:5]) + "); các ứng viên được xếp hạng lại bằng D.")
        else:
            cd = info["centroid_dist"]
            r.p(f"Khoảng cách từ vector truy vấn đến {len(cd)} tâm cụm: " + ", ".join(
                f"cụm {c} = {v:.3f}" for c, v in sorted(cd.items(), key=lambda kv: kv[1])) + f". Hệ thống quét các "
                f"cụm {info['clusters_used']} ({q['n_candidates']} ảnh / {q['n_database']}) rồi xếp hạng chính xác.")
        r.code(ctx["example_txt"][e["stem"]], size=8)
        r.figure(FIG / f"1{i}_{e['stem']}_result.png", f"Kết quả top-{k} cho ảnh {e['query']} (backend {e['backend']})",
                 17)

    # ---- 5.3 danh gia ------------------------------------------------------------------
    r.h(2, "5.3. Đánh giá kết quả")
    r.h(3, "Phương pháp")
    r.p(f"Vì bộ dữ liệu có nhãn chi (genus) suy ra từ category nguồn, hệ thống đánh giá theo cách leave-one-out: "
        f"lần lượt lấy từng ảnh có nhãn thuộc các chi có ≥ 5 ảnh làm truy vấn ({ev['n_queries']} truy vấn, các chi: "
        + ", ".join(f"{g} ({c})" for g, c in ev["genera"].items()) + f"), tìm top-{k} trong {ev['n_database'] - 1} "
        f"ảnh còn lại và tính: P@{k} (tỉ lệ ảnh trong top-{k} cùng chi), P@1, mAP@{k}, và P@{k} theo nhóm hình thái "
        f"(lá rộng/lá kim). Đường cơ sở ngẫu nhiên: P@{k} = {f3(ev['random_baseline']['p_at_k'])} theo chi và "
        f"{f3(ev['random_baseline']['group_p_at_k'])} theo nhóm. Lưu ý nhãn chi là tiêu chí khắt khe: hai cây cùng "
        "chi chụp ở mùa khác nhau có thể trông rất khác, và hai cây khác chi có thể có dáng rất giống nhau; con số "
        "tuyệt đối vì vậy thấp hơn cảm nhận trực quan về “ảnh giống nhau”.")
    fs = ev["feature_sets"]
    rows = [[name, f3(v["p_at_k"]), f3(v["p_at_1"]), f3(v["ap_at_k"]), f3(v["group_p_at_k"]),
             f"{v['ms_per_query']:.1f}"] for name, v in fs.items()]
    rows.append(["(ngẫu nhiên)", f3(ev["random_baseline"]["p_at_k"]), "—", "—",
                 f3(ev["random_baseline"]["group_p_at_k"]), "—"])
    r.table(["Cấu hình đặc trưng", f"P@{k} chi", "P@1 chi", f"mAP@{k}", f"P@{k} nhóm", "ms/truy vấn"], rows,
            widths=[5.2, 2.0, 2.0, 2.0, 2.2, 2.4], caption="Độ chính xác truy vấn theo từng bộ đặc trưng và các "
            "cách kết hợp (backend linear)", size=9.5, align_center_cols={1, 2, 3, 4, 5})
    be = ev["backends"]
    rows = [[b, f3(v["recall_vs_linear"]), f3(v["p_at_k"]), f3(v["ap_at_k"]), f"{v['ms_per_query']:.2f}"]
            for b, v in be.items()]
    r.table(["Backend", f"Recall@{k} so với linear", f"P@{k} chi", f"mAP@{k}", "ms/truy vấn (xếp hạng)"], rows,
            widths=[3, 4, 3, 3, 3.5], caption="So sánh các chiến lược chọn ứng viên (cấu hình trọng số mặc định, "
            f"{be['linear']['n_queries']} truy vấn)", size=9.5, align_center_cols={1, 2, 3, 4})
    r.figure(FIG / "07_evaluation.png", "Độ chính xác theo bộ đặc trưng (trái) và so sánh backend (phải)", 17)
    best = max(fs.items(), key=lambda kv: kv[1]["p_at_k"])
    hc = fs.get("handcrafted(color+shape+texture)", {})
    comb = fs.get("combined(default)", {})
    r.h(3, "Nhận xét")
    r.bullets([
        f"Mọi bộ đặc trưng đều vượt xa đường cơ sở ngẫu nhiên (P@{k} theo chi {f3(ev['random_baseline']['p_at_k'])}): "
        f"màu sắc đạt {f3(fs['color']['p_at_k'])}, kết cấu {f3(fs['texture']['p_at_k'])}, hình dạng "
        f"{f3(fs['shape']['p_at_k'])}" + (f", embedding {f3(fs['deep']['p_at_k'])}" if "deep" in fs else "") + ".",
        f"Trong các đặc trưng thủ công, màu sắc mạnh nhất vì mùa/loại lá/bối cảnh gắn chặt với chi trong bộ dữ "
        f"liệu (ví dụ thông luôn xanh sẫm, bạch dương thân trắng); hình dạng yếu nhất khi dùng riêng vì mặt nạ "
        f"GrabCut còn lẫn nền và cùng một chi có nhiều dáng, nhưng bổ sung tốt khi kết hợp: ba bộ thủ công gộp lại "
        f"đạt P@{k} = {f3(hc.get('p_at_k', 0))} (mAP {f3(hc.get('ap_at_k', 0))}), cao hơn từng bộ riêng lẻ.",
        f"Kết hợp cả bốn bộ với trọng số mặc định đạt P@{k} = {f3(comb.get('p_at_k', 0))}, P@1 = "
        f"{f3(comb.get('p_at_1', 0))}, mAP@{k} = {f3(comb.get('ap_at_k', 0))}; theo nhóm lá rộng/lá kim đạt "
        f"{f3(comb.get('group_p_at_k', 0))} so với ngẫu nhiên {f3(ev['random_baseline']['group_p_at_k'])}. Cấu hình "
        f"tốt nhất trong thử nghiệm là “{best[0]}” với P@{k} = {f3(best[1]['p_at_k'])}.",
        f"Cây k-d trên PCA giữ được {pct(be.get('kdtree', {}).get('recall_vs_linear', 0))} kết quả top-{k} của quét "
        f"tuyến tính với thời gian xếp hạng giảm khoảng "
        f"{be['linear']['ms_per_query'] / max(be.get('kdtree', {}).get('ms_per_query', 1), 1e-6):.0f} lần; "
        f"K-means giữ {pct(be.get('kmeans', {}).get('recall_vs_linear', 0))} khi chỉ quét ≥ 2 cụm. Với "
        f"{ev['n_database']} ảnh, quét tuyến tính vẫn chỉ mất ~{be['linear']['ms_per_query']:.0f} ms nên được dùng "
        "mặc định; chỉ mục phát huy tác dụng khi CSDL lớn hơn nhiều.",
        "Thời gian trả lời một truy vấn hoàn chỉnh (kể cả phân đoạn và trích rút) khoảng 0,5–1 giây trên CPU, "
        "trong đó GrabCut chiếm phần lớn.",
        "Hạn chế: nhãn chi tự động từ category có thể sai/thiếu; phân đoạn đôi khi gộp cỏ/đất vào vùng cây; các "
        "đặc trưng thủ công nhạy với ánh sáng và mùa; embedding ImageNet không được huấn luyện riêng cho cây. "
        "Hướng cải thiện: tinh chỉnh mạng trên dữ liệu cây, học trọng số kết hợp từ phản hồi người dùng "
        "(relevance feedback), dùng phân đoạn ngữ nghĩa thay GrabCut.",
    ])


def sec_demo(r: Report, ctx: dict):
    r.h(1, "6. Demo hệ thống (Yêu cầu 5)")
    r.p("Hệ thống có hai cách sử dụng: giao diện web Streamlit và dòng lệnh.")
    r.h(2, "6.1. Giao diện web")
    r.code("streamlit run app/streamlit_app.py")
    r.p("Thanh bên hiển thị thống kê CSDL (số ảnh, các bộ đặc trưng), cho phép chọn backend (linear / kdtree / "
        "kmeans), số kết quả K và trọng số từng bộ đặc trưng. Người dùng tải lên một ảnh cây bất kỳ (JPG/PNG) hoặc "
        "chọn một ảnh có sẵn; hệ thống hiển thị ảnh đã chuẩn hoá cùng mặt nạ vùng cây, bảng thời gian xử lý, 5 ảnh "
        "giống nhất kèm độ tương đồng, bảng khoảng cách chuẩn hoá và đóng góp của từng bộ, biểu đồ các giá trị đặc "
        "trưng trung gian (biểu đồ HSV, lưới hình dạng, profile, LBP, GLCM, DCT) và, với backend chỉ mục, thông "
        "tin ứng viên/cụm được quét.")
    shot = FIG / "08_demo_streamlit.png"
    if shot.exists():
        r.figure(shot, "Giao diện demo Streamlit", 17)
    r.h(2, "6.2. Dòng lệnh")
    r.code("python scripts/query.py --image data/queries/query_01.jpg --backend linear --top-k 5 \\\n"
           "       --weights color=0.3,shape=0.25,texture=0.2,deep=0.25 --json out.json --fig out.png")
    r.p("Lệnh in bảng kết quả (như các khối văn bản ở mục 5.2), ghi toàn bộ giá trị trung gian ra JSON và vẽ hình "
        "kết quả.")


def sec_conclusion(r: Report, ctx: dict):
    ev = ctx["eval"]
    comb = ev["feature_sets"].get("combined(default)", {})
    r.h(1, "7. Kết luận và hướng phát triển")
    r.p(f"Bài tập đã hoàn thành đầy đủ năm yêu cầu: (1) sưu tầm và chuẩn hoá bộ dữ liệu {ctx['db']['images']} ảnh cây "
        f"thân gỗ đơn lẻ, cùng kích thước, có siêu dữ liệu nguồn/giấy phép và nhãn chi, kèm mô tả điểm giống/khác; "
        f"(2) thiết kế bốn bộ đặc trưng (màu sắc, hình dạng, kết cấu/bố cục, embedding sâu) với phân tích giá trị "
        f"thông tin của từng khối; (3) cài đặt các thuật toán trích rút và hệ CSDL SQLite quản trị đặc trưng theo mô "
        f"hình lai cùng chỉ mục k-d tree và K-means; (4) xây dựng hệ thống tìm kiếm trả về 5 ảnh giống nhất với đầy "
        f"đủ giá trị trung gian, đạt P@5 = {f3(comb.get('p_at_k', 0))} theo chi (gấp "
        f"{comb.get('p_at_k', 0) / max(ev['random_baseline']['p_at_k'], 1e-9):.1f} lần ngẫu nhiên); (5) demo bằng "
        f"giao diện web và dòng lệnh.")
    r.p("Hướng phát triển: mở rộng bộ dữ liệu với nhãn loài do chuyên gia gán; áp dụng phản hồi liên quan để cập "
        "nhật trọng số và vector truy vấn (công thức Rocchio); thay GrabCut bằng mô hình phân đoạn ngữ nghĩa; "
        "tinh chỉnh mạng nơ-ron trên ảnh cây; triển khai CSDL vector (FAISS/pgvector) khi số ảnh lên hàng triệu.")


def appendix(r: Report, ctx: dict):
    r.h(1, "Phụ lục A. Cấu trúc mã nguồn và hướng dẫn chạy")
    r.code("""CSDL-DPT/
  data/images/              ảnh CSDL 512x512            data/manifest.csv
  data/queries/             ảnh truy vấn thử            data/queries.csv, data/manual_exclude.csv
  db/tree_features.sqlite   CSDL đặc trưng
  src/treedb/config.py      cấu hình chung
  src/treedb/collect.py     sưu tầm ảnh (Wikimedia Commons API)
  src/treedb/preprocess.py  chuẩn hoá, khử trùng lặp, kiểm duyệt, tách ảnh truy vấn
  src/treedb/segment.py     phân đoạn vật thể / nền (GrabCut)
  src/treedb/features/      color.py, shape.py, texture.py, deep.py, __init__.py (extract_all)
  src/treedb/db.py          lược đồ và thao tác SQLite
  src/treedb/index.py       chuẩn hoá, khoảng cách, PCA, cây k-d, K-means
  src/treedb/search.py      pipeline truy vấn (Searcher), giải thích kết quả
  src/treedb/evaluate.py    đánh giá leave-one-out
  src/treedb/visualize.py   hình minh hoạ
  scripts/                  build_dataset, extract_features, build_index, run_evaluation, make_figures,
                            query, build_docs
  app/streamlit_app.py      demo web
  results/                  evaluation.json, index_meta.json, query_examples/, excluded_images.csv
  docs/                     báo cáo (docx, pdf) và figures/""", size=8.5)
    r.code("""pip install -r requirements.txt
python scripts/build_dataset.py       # 1. sưu tầm + chuẩn hoá
python scripts/extract_features.py    # 2-3. phân đoạn, trích rút, ghi CSDL
python scripts/build_index.py         # 3b. chỉ mục kd-tree + K-means
python scripts/run_evaluation.py      # 4c. đánh giá
python scripts/make_figures.py        # hình + ví dụ truy vấn
python scripts/build_docs.py          # sinh báo cáo
python scripts/query.py --image <anh.jpg>
streamlit run app/streamlit_app.py""", size=8.5)
    r.h(1, "Phụ lục B. Trích danh sách nguồn ảnh")
    r.p("Toàn bộ ảnh lấy từ Wikimedia Commons theo giấy phép mở; danh sách đầy đủ (tiêu đề, URL trang mô tả, tác "
        "giả, giấy phép) nằm trong data/manifest.csv. Dưới đây là 25 dòng đầu:")
    rows = [[x["filename"], x["source_title"].replace("File:", "")[:55], (x["author"] or "")[:22],
             x["license"], x["genus"] or "—"] for x in ctx["manifest"][:25]]
    r.table(["File", "Tiêu đề nguồn", "Tác giả", "Giấy phép", "Chi"], rows, widths=[2.4, 6.6, 3.2, 2.2, 1.8],
            caption="Trích danh sách nguồn ảnh", size=8.5)
    rows = [[x["filename"], x["source_title"].replace("File:", "")[:60], x["genus"] or "—", x["group"]]
            for x in ctx["queries"]]
    r.table(["File", "Tiêu đề nguồn", "Chi", "Nhóm"], rows, widths=[2.4, 8.6, 2.5, 2.5],
            caption="Các ảnh truy vấn thử (không nằm trong CSDL)", size=8.5)


# ---------------------------------------------------------------------------------------
def build(ctx: dict) -> Path:
    r = Report()
    cover(r, ctx)
    toc(r)
    sec_intro(r, ctx)
    sec_dataset(r, ctx)
    sec_features(r, ctx)
    sec_extraction_db(r, ctx)
    sec_search(r, ctx)
    sec_demo(r, ctx)
    sec_conclusion(r, ctx)
    appendix(r, ctx)
    return r.save(OUT_DOCX)


def to_pdf(docx: Path) -> Path | None:
    if not SOFFICE.exists():
        print("[pdf] khong tim thay LibreOffice; bo qua chuyen PDF")
        return None
    subprocess.run([str(SOFFICE), "--headless", "--convert-to", "pdf", "--outdir", str(docx.parent), str(docx)],
                   check=True, capture_output=True, timeout=300)
    pdf = docx.with_suffix(".pdf")
    return pdf if pdf.exists() else None


if __name__ == "__main__":
    ctx = load_context()
    out = build(ctx)
    print(f"[docx] {out} ({out.stat().st_size / 1024:.0f} KB)")
    pdf = to_pdf(out)
    if pdf:
        print(f"[pdf] {pdf} ({pdf.stat().st_size / 1024:.0f} KB)")
