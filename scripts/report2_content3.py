"""Bao cao 2 - muc 3: trich rut dac trung va he CSDL quan tri dac trung (noi dung 3 cua de bai)."""
from __future__ import annotations

from build_docs import FIG, pct
from docx_helpers import Report
from treedb import config, db


def _timing(ctx: dict) -> dict:
    """Thoi gian do tren anh truy van thu (ms), lay tu vi du truy van da luu."""
    return ctx["example_json"][ctx["examples"][0]["stem"]]["timing_ms"]


def sec_overview(r: Report, ctx: dict):
    tm = _timing(ctx)
    r.h(2, "3.1. Quy trình xử lý ngoại tuyến")
    r.p("Theo mô hình lai của bài giảng [1, bài 1, tr. 18], dữ liệu ảnh không được truy vấn trực tiếp trên điểm ảnh "
        "mà qua các vector đặc trưng được trích rút trước; quy trình dưới đây là các bước 2–4 của vòng đời đối tượng "
        "đa phương tiện [1, bài 1, tr. 15]. Giai đoạn ngoại tuyến (offline) đưa mọi ảnh vào CSDL theo "
        "chuỗi bước cố định; giai đoạn trực tuyến (mục 4) dùng lại đúng các hàm này cho ảnh truy vấn, nhờ vậy vector "
        "truy vấn và vector trong CSDL cùng một không gian:")
    r.numbered([
        "Chuẩn hoá ảnh: xoay theo EXIF, đổi sang RGB, thu cạnh ngắn về 512 điểm ảnh và cắt giữa thành "
        f"{config.IMAGE_SIZE}×{config.IMAGE_SIZE} (scripts/build_dataset.py).",
        "Tách nền: SegFormer-b0 [10] (lớp “tree” của ADE20K) tạo mặt nạ, ghép cây lên nền trắng, lưu vào "
        "data/images_nobg (src/treedb/segment.py; chi tiết ở Báo cáo 1, mục 3.6).",
        "Trích rút bốn bộ đặc trưng màu sắc, hình dạng, kết cấu và embedding sâu (src/treedb/features/).",
        "Ghi siêu dữ liệu và vector vào SQLite (src/treedb/db.py).",
        "Tính thống kê chuẩn hoá mean/std/scale cho từng bộ (src/treedb/index.py).",
        "Dựng các cấu trúc chỉ mục đa chiều: cây k-d, cây R, cây tứ phân, phân cụm K-means "
        "(scripts/build_index.py).",
    ])
    r.p(f"Thời gian đo trên một ảnh truy vấn thử (CPU, ms): tách nền {tm['remove_background']:.0f}, màu sắc "
        f"{tm['extract_color']:.0f}, hình dạng {tm['extract_shape']:.0f}, kết cấu {tm['extract_texture']:.0f}, "
        f"embedding {tm['extract_deep']:.0f}. Khi xây CSDL, ba bộ thủ công chạy song song nhiều tiến trình và "
        "embedding chạy theo lô 32 ảnh, nên toàn bộ CSDL được trích rút trong khoảng một phút.")


def sec_algorithms(r: Report, ctx: dict):
    fs = {f["set_name"]: f for f in ctx["feature_sets"]}
    r.h(2, "3.2. Thuật toán trích rút từng bộ đặc trưng")
    r.p("Ý nghĩa và giá trị thông tin của từng đặc trưng đã trình bày ở Báo cáo 1. Mục này mô tả cách cài đặt: "
        "đầu vào là ảnh RGB đã tách nền I (512×512) và mặt nạ nhị phân M của cây; đầu ra là một vector số thực "
        "float32 cùng “bố cục khối” (tên khối, số chiều) để giải thích kết quả sau này.")
    r.h(3, f"a) Màu sắc ({fs['color']['dim']} chiều) – src/treedb/features/color.py")
    r.p("Cơ sở: biểu đồ tần suất màu, biểu đồ theo vùng, biểu đồ luỹ tích, PWH và đổi không gian màu "
        "[1, bài 11, tr. 9, 15–18]. Bài giảng chia mỗi dải RGB thành m mức (n = m³ ô); nhóm dùng HSV với số mức "
        "khác nhau cho từng kênh, H chia mịn hơn vì mang nhiều thông tin màu nhất.")
    r.numbered([
        "Đổi I sang HSV (OpenCV: H ∈ [0, 180), S, V ∈ [0, 256)).",
        "hsv_fg: lượng tử hoá H, S, V thành 8, 3, 3 mức; ô màu của điểm ảnh là j = 9·h + 3·s + v (72 ô). Đếm số "
        "điểm ảnh cây (M = 1) rơi vào từng ô rồi chia tổng: H(M) = [h₁, …, h₇₂], Σhⱼ = 1.",
        "hsv_grid: lấy hình chữ nhật bao của M, chia lưới 3×3; mỗi ô lưới lập biểu đồ 4×2×2 = 16 ô màu chỉ trên "
        "điểm ảnh cây; nối 9 biểu đồ thành 144 chiều.",
        "hsv_cum: biểu đồ luỹ tích chᵢ = Σ_{j ≤ i} hⱼ, chia 72 để cùng thang đo.",
        "pwh: 72 màu đại diện là tâm các ô HSV, đổi sang CIE Lab. Lấy mẫu đều tối đa 6 000 điểm ảnh cây; mỗi điểm "
        "chia trọng số cho 4 màu đại diện gần nhất theo tỉ lệ nghịch (1 + khoảng cách Lab), tổng trọng số của một "
        "điểm bằng 1; chuẩn hoá tổng.",
        "moments: trung bình, độ lệch chuẩn và độ lệch (qua tanh để giới hạn) của H, S, V trên điểm ảnh cây.",
    ])
    r.h(3, f"b) Hình dạng ({fs['shape']['dim']} chiều) – src/treedb/features/shape.py")
    r.p("Cơ sở: trục chính, trục phụ, hình chữ nhật cơ bản, tâm sai; lưới ô vuông ngưỡng 15%; moment bất biến và "
        "hệ số Fourier [1, bài 11, tr. 20–21, 25], moment Hu [8].")
    r.numbered([
        "region: lấy thành phần liên thông lớn nhất của M (skimage regionprops), tính diện tích tương đối, tỉ lệ "
        "cao/rộng của hình chữ nhật bao, extent, solidity, tâm sai, độ dài trục chính/phụ, hướng trục chính và độ "
        "lệch tâm so với tâm ảnh.",
        "hu: 7 moment bất biến Hu của M, lấy −sign(φ)·log₁₀|φ| để đưa về cùng bậc độ lớn.",
        "grid: cắt M theo hình chữ nhật bao, co về lưới 8×8 bằng nội suy diện tích; ô có tỉ lệ vật thể > 0,15 "
        "nhận giá trị 1 (64 bit).",
        "fourier: đường bao ngoài lớn nhất lấy mẫu 128 điểm thành số phức z = x + iy; FFT; bỏ F₀, chia biên độ "
        "cho |F₁|, giữ biên độ 16 hệ số F₂…F₁₇.",
        "profile: chia hình chữ nhật bao thành 16 dải ngang từ ngọn xuống gốc; giá trị mỗi dải là tỉ lệ cột có "
        "điểm vật thể.",
    ])
    r.h(3, f"c) Kết cấu và bố cục ({fs['texture']['dim']} chiều) – src/treedb/features/texture.py")
    r.p("Cơ sở: độ thô, độ tương phản, hướng sắp xếp các nét [1, bài 11, tr. 26] và chỉ số trên hệ số DCT "
        "[1, bài 11, tr. 27]; cài đặt cụ thể theo GLCM [5], LBP [6] và đặc trưng Tamura [7].")
    r.numbered([
        "Đổi sang ảnh xám, cắt theo hình chữ nhật bao của cây.",
        "glcm: lượng tử 32 mức xám; ma trận đồng xuất hiện đối xứng ở khoảng cách 1 và 3, bốn hướng 0°, 45°, 90°, "
        "135°; tính contrast, dissimilarity, homogeneity, energy, correlation, lấy trung bình theo hướng.",
        "lbp: mẫu nhị phân cục bộ dạng đồng nhất (uniform) với (P, R) = (8, 1) và (16, 2); lập biểu đồ 10 và 18 ô "
        "trên điểm ảnh cây.",
        "edge: gradient Sobel; biểu đồ 8 hướng trong [0, π) có trọng số theo biên độ, cộng mật độ cạnh (tỉ lệ điểm "
        "có biên độ > 60).",
        "tamura: độ thô (kích thước cửa sổ 2ᵏ cho chênh lệch trung bình lớn nhất), độ tương phản σ/κ^¼, tính định "
        "hướng = 1 − entropy chuẩn hoá của biểu đồ hướng gradient mạnh.",
        "dct: ảnh xám thu về 64×64, biến đổi DCT 2 chiều, giữ khối 6×6 tần số thấp, bỏ hệ số DC (35 chiều).",
    ])
    r.h(3, f"d) Embedding sâu ({fs['deep']['dim']} chiều) – src/treedb/features/deep.py")
    r.p("Phần mở rộng của nhóm cho thuộc tính bậc cao (“đặc trưng sâu” [1, bài 2, tr. 17]). Ảnh đã tách nền thu về "
        "224×224, chuẩn hoá theo trung bình/độ lệch chuẩn ImageNet, đưa qua MobileNetV2 [9] (ONNX Runtime). Lấy đầu ra lớp Global Average Pooling 1280 chiều và chuẩn hoá L2 (‖v‖ = 1), nên độ đo "
        "cosine chỉ còn là tích vô hướng.")


def sec_database(r: Report, ctx: dict):
    dbs = ctx["db"]
    r.h(2, "3.3. Thiết kế hệ CSDL quản trị đặc trưng")
    r.h(3, "Mô hình dữ liệu")
    r.p("Hệ CSDL dùng mô hình lai [1, bài 1, tr. 18]: phần quan hệ lưu siêu dữ liệu của ảnh để truy vấn chính xác bằng SQL, phần "
        "vector lưu đặc trưng nội dung để truy vấn tương tự. Ảnh gốc là đối tượng dữ liệu đa phương tiện được lưu "
        "dạng tệp và được tham chiếu bằng đường dẫn [1, bài 2, tr. 42], kèm mã băm SHA-1 để kiểm tra toàn vẹn. Hệ quản trị chọn SQLite vì gọn (một tệp), "
        "có giao dịch ACID và đủ cho vài nghìn ảnh; vector được lưu dạng BLOB float32 để đọc nhanh vào ma trận NumPy.")
    r.h(3, "Phân loại siêu dữ liệu")
    r.p("Theo bài giảng, siêu dữ liệu ảnh gồm văn bản mô tả, các thuộc tính lưu bằng mô hình quan hệ và siêu dữ liệu "
        "dựa trên nội dung (toàn bộ đặc trưng ảnh) [1, bài 6, tr. 15]. CSDL của nhóm quản lý cả ba loại:")
    r.table(["Loại siêu dữ liệu", "Thuộc tính trong CSDL", "Nguồn tạo"], [
        ["Quan hệ: nguồn gốc, quyền sử dụng", "source_title, source_url, author, license, "
         "ingested_at", "Wikimedia Commons API, tự động"],
        ["Quan hệ: thông số kỹ thuật của tệp", "filename, path, width, height, format, file_size, sha1, orig_width, "
         "orig_height", "Tính khi chuẩn hoá ảnh"],
        ["Mô tả nội dung (mức nhận diện)", "genus (chi cây), group_label (lá rộng / lá kim), source_title",
         "Suy từ category nguồn, kiểm tra bằng mắt"],
        ["Dựa trên nội dung (mức miêu tả)", "fg_coverage + 4 vector đặc trưng trong bảng features",
         "Trích rút tự động từ điểm ảnh"],
    ], widths=[5.0, 6.5, 4.5], caption="Siêu dữ liệu được quản lý trong CSDL", size=10)
    r.h(3, "Lược đồ quan hệ")
    r.table(["Bảng", "Cột chính", "Vai trò"], [
        ["images", "image_id (PK), filename (UNIQUE), path, width, height, format, file_size, sha1, source_title, "
                   "source_url, author, license, genus, group_label, orig_width, orig_height, fg_coverage, "
                   "ingested_at", "Siêu dữ liệu của từng ảnh"],
        ["feature_sets", "set_name (PK), dim, description, extractor_version, metric, layout_json",
         "Danh mục bộ đặc trưng: số chiều, độ đo, bố cục khối"],
        ["features", "image_id (FK), set_name (FK), dim, vector BLOB; PK(image_id, set_name)",
         "Vector đặc trưng của mỗi ảnh theo từng bộ"],
        ["feature_stats", "set_name, key (mean | std | scale), value BLOB", "Thống kê chuẩn hoá"],
        ["indexes", "name (PK), kind, meta_json, payload BLOB, built_at", "Chỉ mục đa chiều đã tuần tự hoá"],
        ["image_clusters", "index_name, image_id (FK), cluster_id", "Ảnh thuộc cụm K-means nào"],
        ["query_log", "query_id, query_path, backend, weights_json, top_k, results_json, elapsed_ms, created_at",
         "Nhật ký truy vấn"],
    ], widths=[2.6, 8.4, 5.0], caption="Lược đồ CSDL đặc trưng", size=9.5)
    r.p("Quan hệ: images 1–n features n–1 feature_sets; khoá ngoại có ON DELETE CASCADE nên xoá một ảnh sẽ xoá "
        "luôn vector và phân cụm của nó. Chỉ mục B-tree của SQLite trên features(set_name), images(genus) và "
        "image_clusters(index_name, cluster_id) phục vụ nạp ma trận theo bộ, lọc theo chi và lấy thành viên cụm. "
        "Trích đoạn lệnh tạo bảng:")
    r.code("CREATE TABLE features (\n"
           "    image_id INT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,\n"
           "    set_name TEXT NOT NULL REFERENCES feature_sets(set_name),\n"
           "    dim INT NOT NULL, vector BLOB NOT NULL, PRIMARY KEY (image_id, set_name));\n"
           "CREATE INDEX idx_features_set ON features(set_name);\n"
           "CREATE INDEX idx_images_genus ON images(genus);", size=9)
    _sample_record(r, ctx)
    sets_rows = [[f["set_name"], str(f["dim"]), f["metric"], str(dbs["feature_sets"][f["set_name"]]["count"]),
                  ", ".join(f"{n}({d})" for n, d in f["layout"])] for f in ctx["feature_sets"]]
    r.table(["Bộ", "Chiều", "Độ đo", "Số vector", "Bố cục khối"], sets_rows, widths=[1.8, 1.4, 1.8, 2.0, 9.0],
            caption="Nội dung bảng feature_sets sau khi trích rút", size=9.5, align_center_cols={1, 2, 3})
    r.p(f"CSDL hiện có {dbs['images']} ảnh, {sum(v['count'] for v in dbs['feature_sets'].values())} vector đặc trưng "
        f"thuộc {len(dbs['feature_sets'])} bộ, dung lượng tệp SQLite {dbs['db_size_bytes'] / 1024 / 1024:.1f} MB.")
    r.h(3, "Các thao tác quản trị")
    r.table(["Thao tác", "Hàm (src/treedb/db.py)", "Mô tả"], [
        ["Thêm / cập nhật ảnh", "upsert_image", "INSERT … ON CONFLICT(filename) DO UPDATE"],
        ["Ghi vector", "put_feature, put_features_bulk", "Ghi theo lô trong một giao dịch"],
        ["Đọc vector", "get_feature, get_feature_matrix", "Một ảnh, hoặc cả bộ thành ma trận N×d"],
        ["Thống kê", "put_stat, get_stats", "mean/std/scale của từng bộ"],
        ["Chỉ mục", "put_index, get_index, set_clusters, get_cluster_members", "Lưu/nạp cây và cụm"],
        ["Truy vấn quan hệ", "list_images, get_images, summary", "Siêu dữ liệu, thống kê CSDL"],
        ["Nhật ký", "log_query", "Ghi mỗi lượt truy vấn"],
    ], widths=[3.5, 6.0, 6.5], caption="Các thao tác trên CSDL", size=10)


def _sample_record(r: Report, ctx: dict):
    conn = db.connect()
    img = db.get_image(conn, ctx["example_json"][ctx["examples"][1]["stem"]]["results"][0]["image_id"])
    vecs = {f["set_name"]: db.get_feature(conn, img["image_id"], f["set_name"]) for f in ctx["feature_sets"]}
    conn.close()
    keys = ["image_id", "filename", "width", "height", "file_size", "source_title", "author", "license", "genus",
            "group_label", "fg_coverage"]
    rows = [[k, str(round(img[k], 4) if isinstance(img[k], float) else img[k])] for k in keys]
    rows += [[f"features[{s}] ({v.size} chiều)", ", ".join(f"{x:.3f}" for x in v[:6]) + ", …"]
             for s, v in vecs.items()]
    r.table(["Trường", "Giá trị"], rows, widths=[5.0, 11.0],
            caption=f"Ví dụ một bản ghi trong CSDL: ảnh {img['filename']}", size=9.5)


def sec_norm_index(r: Report, ctx: dict):
    im = ctx["index_meta"]
    r.h(2, "3.4. Chuẩn hoá và kết hợp các bộ đặc trưng")
    r.p("Các bộ có số chiều và thang đo khác nhau nên không cộng khoảng cách trực tiếp được. Với mỗi bộ s, hệ "
        "thống lưu mean và std theo từng chiều (z-score cho các bộ dùng L2: x' = (x − μ)/σ) và scale_s = khoảng "
        "cách trung bình giữa hai ảnh ngẫu nhiên trong CSDL (ước lượng trên 4 000 cặp). Khoảng cách chuẩn hoá "
        "d̃_s = d_s / scale_s có kỳ vọng ≈ 1 với cặp ảnh ngẫu nhiên, nên kết hợp tuyến tính được: "
        "D = Σ_s w_s·d̃_s, Σ_s w_s = 1. Độ tương đồng trả về cho người dùng là sim = 1/(1 + D) ∈ (0, 1]. Trọng số "
        "mặc định: " + ", ".join(f"{k} = {v}" for k, v in config.DEFAULT_WEIGHTS.items()) + ".")
    r.h(2, "3.5. Cấu trúc chỉ mục đa chiều")
    dim_e = sum(f["dim"] for f in ctx["feature_sets"])
    r.p(f"Các chỉ mục được dựng trên không gian nhúng chung E ({dim_e} chiều): nối vector đã chuẩn hoá của mọi bộ, "
        "mỗi khối nhân √w_s / scale_s, nhờ đó khoảng cách Euclid trong E xấp xỉ D. Cơ sở lý thuyết ở mục 2.5 "
        "[1, bài 7; bài 8, tr. 26–28], [2], [3], [4]. Mọi chỉ mục đều theo nguyên tắc "
        "lọc rồi tinh chỉnh: chỉ mục chọn nhanh một tập ứng viên, sau đó ứng viên được xếp hạng lại bằng D chính xác.")
    r.table(["Chỉ mục", "Không gian", "Cách dựng", "Cách tìm"], [
        ["Cây k-d", f"PCA {im['kdtree']['pca_dim']} chiều (giữ {pct(im['kdtree']['explained_variance'])} phương "
                    "sai)", "Chia đôi luân phiên theo từng trục tại trung vị (SciPy cKDTree)",
         f"Lấy {config.CANDIDATE_POOL} láng giềng gần nhất làm ứng viên"],
        ["Cây R", f"PCA {im['rtree']['pca_dim']} chiều ({pct(im['rtree']['explained_variance'])})",
         f"Nạp lô Sort-Tile-Recursive, mỗi nút lưu MBR, lá ≤ 16 điểm ({im['rtree']['n_leaves']} lá, cao "
         f"{im['rtree']['height']} tầng)", "Tốt nhất trước: hàng đợi ưu tiên theo khoảng cách nhỏ nhất tới MBR"],
        ["Cây tứ phân PR", f"PCA 2 chiều ({pct(im['quadtree']['explained_variance'])})",
         f"Ô có > 8 điểm chia thành 4 ô NW, NE, SW, SE ({im['quadtree']['n_leaves']} lá, sâu "
         f"{im['quadtree']['depth']} mức)", "Tốt nhất trước trên các ô vuông"],
        [f"K-means (K = {im['kmeans']['k']})", f"E đầy đủ {dim_e} chiều",
         "Thuật toán Lloyd, khởi tạo k-means++, cài bằng NumPy", "Quét ≥ 2 cụm có tâm gần vector truy vấn nhất"],
    ], widths=[2.6, 3.6, 5.4, 4.4], caption="Các cấu trúc chỉ mục đa chiều", size=9.5)
    r.p(f"Kích thước các cụm K-means: {im['kmeans']['sizes']}. Chỉ mục được tuần tự hoá vào bảng indexes, phân cụm "
        "ghi vào bảng image_clusters.")
    r.figure(FIG / "06_clusters.png", "Phân bố ảnh trong không gian PCA: theo cụm K-means (trái) và theo nhóm hình "
             "thái (phải)", 16)


def sec_content3(r: Report, ctx: dict):
    r.h(1, "3. Trích rút đặc trưng và hệ CSDL quản trị đặc trưng (Nội dung 3)")
    sec_overview(r, ctx)
    sec_algorithms(r, ctx)
    sec_database(r, ctx)
    sec_norm_index(r, ctx)
