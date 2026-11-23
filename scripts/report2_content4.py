"""Bao cao 2 - muc 4: he thong tim kiem anh tuong dong (noi dung 4a, 4b, 4c cua de bai)."""
from __future__ import annotations

import numpy as np

from build_docs import FIG, f3, pct, sec_filter_feedback
from docx_helpers import Report
from report2_worked import worked_example
from treedb.index import distance

# Vi du bieu do muc xam cua bai giang [1, bai 11, tr. 10]: 3 anh xam 8x8, 3 bit (8 muc)
LECTURE_H = {"H1": [8, 8, 8, 8, 8, 8, 8, 8], "H2": [6, 6, 8, 8, 10, 10, 7, 9], "H3": [10, 10, 2, 2, 10, 10, 10, 10]}

SET_VI = {"color": "màu sắc", "shape": "hình dạng", "texture": "kết cấu", "deep": "embedding"}
HUE_NAMES = ["đỏ – cam", "vàng – lục vàng", "lục", "lục – lơ", "lơ – lam", "lam", "tím", "hồng – đỏ"]  # 45°/o
LEVEL3 = ["thấp", "vừa", "cao"]


def _bin_name(j: int) -> str:
    """O mau j = 9h + 3s + v cua bieu do HSV 8x3x3 -> mo ta bang chu."""
    h, s, v = j // 9, (j // 3) % 3, j % 3
    return f"{j} (H {HUE_NAMES[h]}, S {LEVEL3[s]}, V {LEVEL3[v]})"


# ---- 4a ---------------------------------------------------------------------------------
def sec_architecture(r: Report, ctx: dict):
    k = ctx["eval"]["k"]
    r.h(2, "4.1. Sơ đồ khối, chức năng các khối và quy trình thực hiện (yêu cầu 4a)")
    r.p("Hệ thống được thiết kế theo mô hình truy vấn ảnh dựa trên nội dung hai giai đoạn của bài giảng "
        "[1, bài 1, tr. 24]. Giai đoạn ngoại tuyến (mục 3) đưa ảnh vào CSDL và dựng chỉ mục. Giai đoạn "
        f"trực tuyến nhận một ảnh cây mới và trả về {k} ảnh giống nhất. Hai giai đoạn dùng chung khối tiền xử lý, "
        "tách nền và trích rút, nên vector truy vấn được so sánh trong đúng không gian của vector CSDL.")
    r.figure(FIG / "04_block_diagram.png", "Sơ đồ khối hệ thống: giai đoạn ngoại tuyến (trên) và trực tuyến (dưới)",
             17)
    r.table(["Khối", "Chức năng", "Dữ liệu vào", "Dữ liệu ra"], [
        ["1. Thu thập ảnh", "Duyệt category Commons, lọc theo quy tắc, tải ảnh, ghi siêu dữ liệu",
         "Danh sách category", "Ảnh thô + raw_manifest.csv"],
        ["2. Tiền xử lý", "Xoay EXIF, RGB, 512×512, khử trùng lặp pHash, áp danh sách kiểm duyệt", "Ảnh thô",
         "Ảnh 512×512 + manifest.csv"],
        ["3. Tách nền", "SegFormer (lớp tree) + lọc hình thái, dự phòng GrabCut; ghép cây lên nền trắng",
         "Ảnh RGB 512×512", "Ảnh đã tách nền, mặt nạ M"],
        ["4. Trích rút", "Tính 4 bộ đặc trưng", "Ảnh đã tách nền + M", "4 vector: " + ", ".join(
            f"{f['set_name']} ({f['dim']})" for f in ctx["feature_sets"])],
        ["5. CSDL SQLite", "Lưu siêu dữ liệu và vector", "Bản ghi ảnh + vector", "Bảng images, features, "
                                                                              "feature_sets"],
        ["6. Chuẩn hoá & chỉ mục", "mean/std/scale; PCA + cây k-d, cây R, cây tứ phân; K-means",
         "Ma trận đặc trưng", "feature_stats, indexes, image_clusters"],
        ["Q1. Tiền xử lý + tách nền", "Như khối 2–3 cho ảnh mới (mọi kích thước)", "Ảnh truy vấn",
         "Ảnh 512×512 đã tách nền + M"],
        ["Q2. Trích rút", "Như khối 4", "Ảnh + M", "Vector q_s của từng bộ"],
        ["Q3. Chọn ứng viên", "linear: toàn bộ; cây k-d/R/tứ phân: láng giềng trên PCA; K-means: ≥ 2 cụm gần "
                              "nhất; lọc siêu dữ liệu nếu có", "q, chỉ mục", "Tập ứng viên"],
        ["Q4. Tính khoảng cách", "d_s (L1/L2/cosine), d̃_s = d_s/scale_s, D = Σ w_s·d̃_s",
         "q, vector ứng viên, thống kê, trọng số", "d_s, d̃_s, D của từng ứng viên"],
        ["Q5. Xếp hạng", f"sim = 1/(1+D), sắp giảm dần, lấy top-{k}, ghi query_log", "D của ứng viên",
         f"{k} ảnh giống nhất + giá trị trung gian"],
    ], widths=[3.0, 4.8, 3.6, 4.6], caption="Chức năng và dữ liệu vào/ra của từng khối", size=9)
    r.p("Bảng dưới đây ánh xạ các khối của hệ thống vào năm khối của sơ đồ tra cứu dữ liệu đa phương tiện trong bài giảng "
        "[1, bài 1, tr. 26–27]:")
    r.table(["Khối theo bài giảng", "Khối của hệ thống", "Luồng dữ liệu (theo bài giảng)"], [
        ["1. Insert into the database", "1–3, 5: thu thập, tiền xử lý, tách nền, ghi SQLite",
         "MM-Objects + relational data → MM-Database"],
        ["2. Extraction of features", "4, 6: trích rút, thống kê chuẩn hoá, chỉ mục đặc trưng (Feature index)",
         "Raw data → Feature values"],
        ["3. Query preparation", "Q1, Q2: chuẩn hoá, phân đoạn (tách nền), trích rút; chọn chiến lược tìm kiếm",
         "Query → Query plan & feature values"],
        ["4. Similarity computation & query processing", "Q3, Q4: chọn ứng viên qua chỉ mục, lọc siêu dữ "
         "liệu, tính D", "Feature values + raw & relational data → Result data"],
        ["5. Result preparation", "Q5: xếp hạng top-5, ảnh thu nhỏ, bảng giá trị trung gian", "Result data → Result"],
        ["Relevance feedback", "Đánh dấu R/N trên kết quả, cập nhật vector truy vấn (mục 4.4)",
         "Result preparation → Query preparation"],
    ], widths=[4.4, 6.6, 5.0], caption="Đối chiếu sơ đồ khối của hệ thống với sơ đồ trong bài giảng", size=9.5)
    r.h(3, "Quy trình thực hiện một truy vấn")
    r.numbered([
        f"Người dùng đưa vào một ảnh cây mới (tải lên giao diện web hoặc đường dẫn dòng lệnh), chọn chiến lược "
        f"tìm kiếm, số kết quả K (mặc định {k}) và trọng số các bộ.",
        "Ảnh được chuẩn hoá về 512×512 và tách nền giống hệt ảnh trong CSDL.",
        "Trích rút 4 vector đặc trưng bằng chính các hàm đã dùng khi xây CSDL.",
        "Nạp ma trận đặc trưng và thống kê từ SQLite (một lần khi khởi động), chọn tập ứng viên.",
        "Với mỗi ứng viên, tính khoảng cách từng bộ, chuẩn hoá, kết hợp theo trọng số thành D.",
        f"Sắp xếp theo sim = 1/(1+D) giảm dần, trả về {k} ảnh đầu cùng bảng giá trị trung gian, ghi nhật ký.",
    ])


# ---- 4b ---------------------------------------------------------------------------------
def _feature_values(r: Report, e: dict, q: dict):
    F = q["query"]["features"]
    b, s, t = F["color"]["blocks"], F["shape"]["blocks"], F["texture"]["blocks"]
    rows = [["Màu: hsv_fg (12 ô đầu / 72)", ", ".join(f"{x:.3f}" for x in b["hsv_fg"][:12])],
            ["Màu: moments (mean H,S,V; std H,S,V; skew H,S,V)", ", ".join(f"{x:.3f}" for x in b["moments"])],
            ["Hình dạng: region (area, h/w, extent, solidity, ecc, major, minor, orient, dx, dy)",
             ", ".join(f"{x:.3f}" for x in s["region"])],
            ["Hình dạng: hu (7 moment, log)", ", ".join(f"{x:.2f}" for x in s["hu"])],
            ["Hình dạng: grid 8×8 (64 bit, trái → phải, trên → dưới)",
             " ".join("".join("1" if x > 0.5 else "0" for x in s["grid"][i:i + 8]) for i in range(0, 64, 8))],
            ["Hình dạng: profile (16 dải, ngọn → gốc)", ", ".join(f"{x:.2f}" for x in s["profile"])],
            ["Kết cấu: glcm (contrast, dissim, homog, energy, corr × d = 1, 3)",
             ", ".join(f"{x:.3f}" for x in t["glcm"])],
            ["Kết cấu: tamura (coarseness, contrast, directionality)", ", ".join(f"{x:.3f}" for x in t["tamura"])],
            ["Embedding: 8 chiều đầu / 1280", ", ".join(f"{x:.4f}" for x in F["deep"]["blocks"]["mobilenetv2_gap"][:8])]]
    r.table(["Đặc trưng (khối)", "Giá trị"], rows, widths=[5.5, 10.5],
            caption=f"Một số giá trị đặc trưng của ảnh truy vấn {e['query']}", size=9)


def _ranking_table(r: Report, q: dict, k: int, title: str):
    sets = q["sets"]
    body = [[str(x["rank"]), x["filename"], x["genus"] or "—"] + [f"{x['per_set'][s]['norm']:.3f}" for s in sets]
            + [f"{x['distance']:.3f}", f"{x['similarity']:.3f}"] for x in q["results"]]
    r.table(["Hạng", "Ảnh", "Chi"] + [f"d̃ {s}" for s in sets] + ["D", "sim"], body, caption=title, size=9.5,
            align_center_cols=set(range(3, 5 + len(sets))))


def _worked(r: Report, e: dict, q: dict):
    w = worked_example(q, 1)
    res, S = w["result"], w["sets"]
    r.h(3, f"Tính độ tương đồng từng bước: {e['query']} so với {res['filename']} (hạng 1)")
    r.p("Các giá trị dưới đây được tính lại từ vector trong CSDL và khớp với kết quả hệ thống trả về (sai lệch "
        "< 10⁻³), nhằm minh hoạ cụ thể công thức ở mục 3.4.")
    c = S["color"]
    r.p(f"Bước 1 – Màu sắc, độ đo L1: d_color = Σ|q_j − x_j| trên cả {sum(1 for _ in c['blocks'])} khối. Tám ô "
        "hsv_fg có chênh lệch lớn nhất:")
    r.table(["Ô màu j", "q_j", "x_j", "|q_j − x_j|"],
            [[_bin_name(j), f"{a:.4f}", f"{b:.4f}", f"{d:.4f}"] for j, a, b, d in c["bins"]],
            widths=[7.0, 3.0, 3.0, 3.0], caption="Chênh lệch biểu đồ màu hsv_fg (8 ô lớn nhất)", size=9.5,
            align_center_cols={1, 2, 3})
    r.p("Tổng theo khối: " + ", ".join(f"{n} = {v:.4f}" for n, v in c["blocks"]) + f" → d_color = {c['raw']:.4f}. "
        "Khối hsv_grid đóng góp nhiều nhất vì nó so 9 biểu đồ của 9 vùng (tổng mỗi vùng bằng 1), nên phản ánh cả "
        "việc màu nằm ở đâu trên tán.")
    for step, s in ((2, "shape"), (3, "texture")):
        v = S[s]
        r.p(f"Bước {step} – {SET_VI[s].capitalize()}, độ đo L2 trên z-score: mỗi chiều chuẩn hoá x' = (x − μ)/σ "
            "rồi d = √Σ(q'_j − x'_j)². Tổng bình phương theo khối: "
            + ", ".join(f"{n} = {val:.3f}" for n, val in v["blocks"])
            + f" → d_{s} = √{sum(val for _, val in v['blocks']):.3f} = {v['raw']:.4f}.")
    d = S["deep"]
    r.p(f"Bước 4 – Embedding, độ đo cosine: hai vector đã chuẩn hoá (‖q‖ = {d['nq']:.3f}, ‖x‖ = {d['nx']:.3f}), "
        f"q·x = {d['dot']:.4f} → d_deep = 1 − cos(q, x) = {d['raw']:.4f}.")
    r.p("Bước 5 – Chuẩn hoá theo scale, nhân trọng số và cộng:")
    r.table(["Bộ", "Độ đo", "d_s (thô)", "scale_s", "d̃_s = d_s/scale_s", "w_s", "w_s·d̃_s"],
            [[s, v["metric"], f"{v['raw']:.4f}", f"{v['scale']:.4f}", f"{v['norm']:.4f}", f"{v['weight']:.2f}",
              f"{v['contrib']:.4f}"] for s, v in S.items()]
            + [["Tổng", "", "", "", "", "1,00", f"D = {w['D']:.4f}"]],
            widths=[2.0, 1.8, 2.2, 2.0, 3.2, 1.6, 3.2], caption=f"Kết hợp khoảng cách cho ảnh {res['filename']}",
            size=9.5, align_center_cols={1, 2, 3, 4, 5, 6})
    best = min(S, key=lambda s: S[s]["norm"])
    r.p(f"Độ tương đồng sim = 1/(1 + D) = 1/(1 + {w['D']:.4f}) = {w['sim']:.4f}. Bộ {SET_VI[best]} cho d̃ nhỏ nhất "
        f"({S[best]['norm']:.3f}), tức là bộ thấy hai ảnh giống nhau rõ nhất so với một cặp ảnh ngẫu nhiên (d̃ ≈ 1).")


def sec_intermediate(r: Report, ctx: dict):
    k = ctx["eval"]["k"]
    ex, ej = ctx["examples"], ctx["example_json"]
    r.h(2, "4.2. Minh hoạ kết quả trung gian của quá trình truy vấn (yêu cầu 4b)")
    r.p("Các ảnh truy vấn thử được tách riêng khi xây bộ dữ liệu, không nằm trong CSDL. Mỗi ví dụ dưới đây được "
        "sinh bằng scripts/make_figures.py và lưu đầy đủ giá trị trung gian trong results/query_examples/.")
    _lecture_check(r)
    e = ex[1]
    q = ej[e["stem"]]
    r.h(3, f"Ví dụ 1: ảnh {e['query']} (chi thật {e['genus']}), quét tuyến tính")
    r.p(f"Sau tách nền, vùng cây chiếm {pct(q['query']['mask_coverage'])} diện tích ảnh. Giá trị đặc trưng trích "
        "rút được:")
    _feature_values(r, e, q)
    r.figure(FIG / f"12_{e['stem']}_features.png", f"Biểu diễn trực quan các đặc trưng của ảnh {e['query']}", 17)
    _ranking_table(r, q, k, f"Khoảng cách chuẩn hoá từng bộ, D và độ tương đồng của top-{k} cho {e['query']}")
    r.figure(FIG / f"12_{e['stem']}_result.png", f"Top-{k} cho ảnh {e['query']} và đóng góp của từng bộ đặc trưng", 17)
    _worked(r, e, q)
    tm = q["timing_ms"]
    r.p("Thời gian xử lý truy vấn (ms): " + ", ".join(f"{a} = {b:.1f}" for a, b in tm.items()) + ".")
    e = ex[0]
    q = ej[e["stem"]]
    r.h(3, f"Ví dụ 2: ảnh {e['query']} (chi thật {e['genus']}), một truy vấn chưa tốt")
    _ranking_table(r, q, k, f"Top-{k} cho {e['query']}")
    r.figure(FIG / f"11_{e['stem']}_result.png", f"Top-{k} cho ảnh {e['query']}", 17)
    r.p(f"Không ảnh nào trong top-{k} cùng chi {e['genus']}: kết quả là các cây lá rộng có tán tròn, màu lục tương "
        "tự. Bạch dương chỉ khác rõ ở thân trắng, mà thân chỉ chiếm phần nhỏ diện tích nên đóng góp ít vào biểu đồ "
        "màu và kết cấu. Đây đúng là trường hợp bài giảng nêu: so sánh màu “có thể không tốt lắm” khi hai đối tượng "
        "khác nhau có màu giống nhau [1, bài 2, tr. 18], và “nhiều thông tin nội dung của ảnh sẽ bị mất nếu chỉ sử "
        "dụng thuộc tính bậc thấp để mô tả” [1, bài 2, tr. 16].")
    for i, e in enumerate(ex[2:4], start=3):
        q = ej[e["stem"]]
        info = q["index_info"]
        r.h(3, f"Ví dụ {i}: ảnh {e['query']} (chi thật {e['genus']}), dùng chỉ mục {e['backend']}")
        if e["backend"] == "kdtree":
            r.p(f"Cây k-d trên {info['pca_dim']} thành phần PCA trả về {q['n_candidates']}/{q['n_database']} ứng "
                f"viên trong {q['timing_ms']['candidates']:.2f} ms; khoảng cách PCA của 5 ứng viên gần nhất: "
                + ", ".join(f"{x:.3f}" for x in info["tree_dist"][:5]) + ". Các ứng viên sau đó được xếp hạng lại "
                "bằng D.")
        else:
            cd = sorted(info["centroid_dist"].items(), key=lambda kv: kv[1])
            r.p("Khoảng cách từ vector truy vấn tới các tâm cụm: " + ", ".join(f"cụm {c} = {v:.3f}" for c, v in cd)
                + f". Hệ thống quét các cụm {info['clusters_used']} ({q['n_candidates']}/{q['n_database']} ảnh) "
                "rồi xếp hạng chính xác.")
        _ranking_table(r, q, k, f"Top-{k} cho {e['query']} (chỉ mục {e['backend']})")
        r.figure(FIG / f"1{i}_{e['stem']}_result.png", f"Top-{k} cho ảnh {e['query']} (chỉ mục {e['backend']})", 17)


def _lecture_check(r: Report):
    """Chay ham distance() cua he thong tren vi du H1/H2/H3 cua bai giang de kiem chung cai dat."""
    H = {k: np.asarray(v, np.float32) for k, v in LECTURE_H.items()}
    pairs = [("H1", "H2"), ("H1", "H3"), ("H2", "H3")]
    rows = [[f"{a} – {b}"] + [f"{distance(m, H[a], H[b][None])[0]:.3f}" for m in ("l1", "l2")] for a, b in pairs]
    best = min(pairs, key=lambda ab: distance("l2", H[ab[0]], H[ab[1]][None])[0])
    r.h(3, "Kiểm chứng hàm khoảng cách bằng ví dụ của bài giảng")
    r.p("Bài giảng cho ba ảnh xám 8×8 mã hoá 3 bit: H1 = [8, 8, 8, 8, 8, 8, 8, 8], H2 = [6, 6, 8, 8, 10, 10, 7, 9], "
        "H3 = [10, 10, 2, 2, 10, 10, 10, 10] và hỏi hai ảnh nào gần nhau nhất theo L2 [1, bài 11, tr. 10]. Chạy "
        "chính hàm distance() của hệ thống (src/treedb/index.py) trên ba biểu đồ này:")
    r.table(["Cặp ảnh", "L1", "L2"], rows, widths=[4.0, 3.0, 3.0], caption="Khoảng cách giữa các biểu đồ trong ví dụ "
            "của bài giảng, tính bằng mã nguồn của hệ thống", size=10.5, align_center_cols={1, 2})
    r.p(f"Theo cả L1 và L2, {best[0]} và {best[1]} gần nhau nhất. Tính tay cho cặp H1 – H2: hiệu từng ô là "
        "2, 2, 0, 0, 2, 2, 1, 1 nên L1 = 10 và L2 = √(4·4 + 2·1) = √18 ≈ 4,243, khớp với kết quả của hàm.")


# ---- 4c ---------------------------------------------------------------------------------
def sec_evaluation(r: Report, ctx: dict):
    ev = ctx["eval"]
    k, fs, be = ev["k"], ev["feature_sets"], ev["backends"]
    rb = ev["random_baseline"]
    r.h(2, "4.3. Đánh giá kết quả (yêu cầu 4c)")
    r.h(3, "Phương pháp")
    r.p("Theo bài giảng, hệ thống được đánh giá trên hai mặt [1, bài 1, tr. 44–45]: sự hiệu quả về kết quả "
        "(effectiveness) và độ hiệu quả về tài nguyên (efficiency: thời gian phản hồi). Đánh giá là ngoại tuyến, dùng "
        "các truy vấn cố định và nhãn liên quan [1, bài 1, tr. 51], theo cách leave-one-out: lần lượt lấy từng ảnh "
        f"có nhãn thuộc các chi có ≥ 5 ảnh làm truy vấn ({ev['n_queries']} truy vấn; "
        + ", ".join(f"{g} {c}" for g, c in ev["genera"].items())
        + f"), tìm top-{k} trong {ev['n_database'] - 1} ảnh còn lại. Một ảnh kết quả là liên quan khi cùng chi với "
        "ảnh truy vấn. Với mỗi truy vấn: ca = số ảnh liên quan trong top-5, fa = 5 − ca, fd = số ảnh liên quan "
        "không được trả về, cd = phần còn lại của CSDL [1, bài 1, tr. 47–48]; tỉ lệ đúng P@5 = ca/(ca + fa) = ca/5 "
        "và độ phủ R@5 = ca/(ca + fd) [1, bài 1, tr. 49]. Nhóm dùng thêm P@1 và mAP@5 (mở rộng, mục 2.6) để đánh "
        f"giá thứ tự xếp hạng, và P@{k} theo nhóm hình thái (lá rộng / lá kim). Đường cơ sở ngẫu nhiên: P@{k} = "
        f"{f3(rb['p_at_k'])} theo chi, {f3(rb['group_p_at_k'])} theo nhóm.")
    _quality_matrix(r, ev)
    rows = [[n, f3(v["p_at_k"]), f3(v["p_at_1"]), f3(v["ap_at_k"]), f3(v["group_p_at_k"]), f"{v['ms_per_query']:.1f}"]
            for n, v in fs.items()]
    rows.append(["(ngẫu nhiên)", f3(rb["p_at_k"]), "—", "—", f3(rb["group_p_at_k"]), "—"])
    r.table(["Cấu hình đặc trưng", f"P@{k} chi", "P@1", f"mAP@{k}", f"P@{k} nhóm", "ms/truy vấn"], rows,
            widths=[5.2, 2.0, 2.0, 2.0, 2.2, 2.4], caption="Độ chính xác theo bộ đặc trưng và cách kết hợp (quét "
            "tuyến tính)", size=9.5, align_center_cols={1, 2, 3, 4, 5})
    rows = [[b, f3(v["recall_vs_linear"]), f3(v["p_at_k"]), f3(v["ap_at_k"]), f"{v['ms_per_query']:.2f}"]
            for b, v in be.items()]
    r.table(["Chiến lược", f"Recall@{k} so với linear", f"P@{k} chi", f"mAP@{k}", "ms/truy vấn"], rows,
            widths=[3, 4, 3, 3, 3], caption=f"So sánh các chỉ mục ({be['linear']['n_queries']} truy vấn, trọng số "
            "mặc định)", size=9.5, align_center_cols={1, 2, 3, 4})
    r.figure(FIG / "07_evaluation.png", "Độ chính xác theo bộ đặc trưng (trái) và theo chỉ mục (phải)", 17)
    sec_filter_feedback(r, ctx, num="4.4")
    _remarks(r, ctx)


def _quality_matrix(r: Report, ev: dict):
    """Ma tran chat luong [1, bai 1, tr. 48] cong don tren moi truy van, cau hinh ket hop mac dinh."""
    k, q, n = ev["k"], ev["n_queries"], ev["n_database"]
    ca = round(ev["feature_sets"]["combined(default)"]["p_at_k"] * k * q)
    rel = sum(c * (c - 1) for c in ev["genera"].values())   # moi truy van co (so anh cung chi - 1) anh lien quan
    g_max, n_max = max(ev["genera"].items(), key=lambda kv: kv[1])
    fa, fd = k * q - ca, rel - ca
    cd = q * (n - 1) - ca - fa - fd
    r.table(["Cộng trên " + str(q) + " truy vấn", "Hệ thống trả về (liên quan)", "Hệ thống loại (không liên quan)"],
            [["Liên quan (cùng chi)", f"ca = {ca}", f"fd = {fd}"],
             ["Không liên quan", f"fa = {fa}", f"cd = {cd}"]],
            widths=[5.0, 5.5, 5.5], caption="Ma trận chất lượng của cấu hình kết hợp (top-5, cộng dồn mọi truy vấn)",
            size=10.5, align_center_cols={1, 2})
    r.p(f"Từ ma trận: P = ca/(ca + fa) = {ca}/{ca + fa} = {ca / (ca + fa):.3f}; R = ca/(ca + fd) = {ca}/{rel} = "
        f"{ca / rel:.3f}. Độ phủ thấp là tất yếu khi chỉ trả về 5 ảnh: chi {g_max} có {n_max} ảnh nên một truy vấn thuộc "
        f"chi này có {n_max - 1} ảnh liên quan và R@5 tối đa chỉ 5/{n_max - 1} ≈ {5 / (n_max - 1):.3f}. Đúng như bài giảng nhận xét, “cả tỉ lệ đúng và độ phủ chỉ có ý "
        "nghĩa khi chúng được sử dụng đồng thời” [1, bài 1, tr. 50] và độ phủ khó ước lượng hơn tỉ lệ đúng "
        "[1, bài 1, tr. 51]; với hệ trả về top-5, tỉ lệ đúng phản ánh trực tiếp chất lượng người dùng nhìn thấy.")


def _remarks(r: Report, ctx: dict):
    ev = ctx["eval"]
    k, fs, be, rb = ev["k"], ev["feature_sets"], ev["backends"], ev["random_baseline"]
    comb, hc = fs["combined(default)"], fs["handcrafted(color+shape+texture)"]
    r.h(2, "4.5. Nhận xét")
    r.bullets([
        f"Mọi bộ đặc trưng đều vượt đường cơ sở ngẫu nhiên P@{k} = {f3(rb['p_at_k'])}: màu sắc "
        f"{f3(fs['color']['p_at_k'])}, kết cấu {f3(fs['texture']['p_at_k'])}, hình dạng {f3(fs['shape']['p_at_k'])}, "
        f"embedding {f3(fs['deep']['p_at_k'])}. Giả thiết “nếu hai hình ảnh có mầu sắc giống nhau thì nội dung của "
        "chúng cũng giống nhau” [1, bài 2, tr. 16] chỉ đúng một phần: màu là bộ thủ công mạnh nhất nhưng một mình "
        "chưa đủ.",
        f"Kết hợp có lợi: ba bộ thủ công gộp lại đạt P@{k} = {f3(hc['p_at_k'])}, cao hơn từng bộ riêng; thêm "
        f"embedding đạt {f3(comb['p_at_k'])} (P@1 = {f3(comb['p_at_1'])}, mAP@{k} = {f3(comb['ap_at_k'])}), gấp "
        f"{comb['p_at_k'] / rb['p_at_k']:.1f} lần ngẫu nhiên. Theo nhóm lá rộng/lá kim đạt "
        f"{f3(comb['group_p_at_k'])}.",
        f"Hình dạng yếu nhất khi dùng riêng ({f3(fs['shape']['p_at_k'])}) vì một chi có nhiều dáng (tuổi, cắt tỉa) "
        "và hình bao chỉ là hình chiếu 2D, nhưng vẫn bổ sung thông tin khi kết hợp.",
        f"Cây k-d giữ {pct(be['kdtree']['recall_vs_linear'])} kết quả của quét tuyến tính với thời gian xếp hạng "
        f"giảm khoảng {be['linear']['ms_per_query'] / be['kdtree']['ms_per_query']:.0f} lần; K-means giữ "
        f"{pct(be['kmeans']['recall_vs_linear'])}. Cây R ({pct(be['rtree']['recall_vs_linear'])}) và cây tứ phân "
        f"({pct(be['quadtree']['recall_vs_linear'])}) kém hơn vì chỉ làm việc trên "
        f"{ctx['index_meta']['rtree']['pca_dim']} và 2 chiều PCA: chúng hợp với dữ liệu ít chiều hơn là vector "
        "hàng trăm chiều, đúng như bài giảng lưu ý cây R hợp với dữ liệu “có ít thuộc tính (thường tối đa 10 thuộc "
        "tính)” [1, bài 7, tr. 50] và cây tứ phân tốn dung lượng khi nhiều chiều [1, bài 7, tr. 48]. Cấu trúc cây "
        "còn phụ thuộc thứ tự chọn nút [1, bài 7, tr. 29], nên độ phủ dưới 100% so với quét tuyến tính là bản "
        "chất của việc lọc trước, không phải lỗi cài đặt.",
        f"Với {ev['n_database']} ảnh, quét tuyến tính chỉ mất ~{be['linear']['ms_per_query']:.0f} ms nên là chế độ "
        "mặc định; thời gian một truy vấn hoàn chỉnh khoảng 0,6–1,2 giây, phần lớn dành cho tách nền và trích rút.",
        "Hạn chế: nhãn chi lấy tự động có thể sai; tách nền đôi khi gộp hàng cây phía sau vào vùng cây; đặc trưng "
        "thủ công nhạy với ánh sáng và mùa; mạng MobileNetV2 không được huấn luyện riêng cho cây. Khoảng cách ngữ "
        "nghĩa vẫn còn ở mức truyền đạt (mục 2.3); theo bài giảng, siêu dữ liệu, đặc trưng hoá và cơ chế phản hồi là "
        "các công cụ thu hẹp khoảng cách này [1, bài 1, tr. 29], và phản hồi liên quan đã tăng P@5 ở mục 4.4.",
    ])


def sec_content4(r: Report, ctx: dict):
    r.h(1, "4. Hệ thống tìm kiếm ảnh tương đồng (Nội dung 4)")
    sec_architecture(r, ctx)
    sec_intermediate(r, ctx)
    sec_evaluation(r, ctx)
