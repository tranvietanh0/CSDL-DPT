"""Cac muc bao cao dua tren phan tich dinh luong (results/feature_analysis.json), dung chung cho
bao cao tong (build_docs.py) va Bao cao 1 - noi dung 1, 2 (build_report1.py).
"""
from __future__ import annotations

from docx_helpers import Report
from treedb import config

FIG = config.FIG_DIR
SET_VI = {"color": "Màu sắc", "shape": "Hình dạng", "texture": "Kết cấu / bố cục", "deep": "Embedding sâu"}
BLOCK_VI = {
    "hsv_global": "biểu đồ HSV toàn ảnh", "hsv_grid": "biểu đồ HSV theo lưới 3×3", "hsv_fg": "biểu đồ HSV vùng cây",
    "hsv_bg": "biểu đồ HSV nền", "moments": "moment màu", "region": "thước đo vùng", "hu": "moment Hu",
    "grid": "lưới nhị phân 8×8", "fourier": "hệ số Fourier đường bao", "profile": "profile bề rộng",
    "glcm": "GLCM", "lbp": "LBP", "edge": "hướng cạnh + mật độ cạnh", "tamura": "Tamura", "dct": "DCT tần số thấp",
    "mobilenetv2_gap": "MobileNetV2 GAP",
}


def _f(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}".replace(".", ",")


def sec_quant_dataset(r: Report, fa: dict, num: str):
    """Muc dinh luong diem giong / khac theo chi va theo nhom la rong / la kim."""
    r.h(2, f"{num}. Định lượng điểm giống và khác nhau theo chi cây")
    keys = fa["key_features"]
    genera = list(fa["genera"])
    r.p(f"Để kiểm chứng các nhận xét ở trên bằng số liệu, nhóm thống kê trung bình một số thuộc tính bậc thấp "
        f"trên {fa['n_labeled']} ảnh có nhãn chi thuộc {len(genera)} chi có từ 9 ảnh trở lên. Các thuộc tính này "
        f"được tính trên mặt nạ vùng cây (trừ khi ghi khác); sắc độ H tính theo độ (0–360°), S và V trong [0, 1].")
    head = ["Chi", "Số ảnh"] + ["Diện tích", "Cao/rộng", "Solidity", "Tâm sai", "H (°)", "S", "V", "Mật độ cạnh"]
    pick = keys[:8]
    rows = []
    for g in genera + ["__all__"]:
        v = fa["per_genus"][g]
        rows.append([g if g != "__all__" else "Toàn bộ CSDL", str(v["n"])]
                    + [_f(v[k][0], 0 if k.startswith("Sắc độ H") else 2) for k in pick])
    r.table(head, rows, widths=[2.3, 1.3, 1.6, 1.6, 1.6, 1.5, 1.4, 1.3, 1.3, 1.9],
            caption="Trung bình một số thuộc tính của vùng cây theo chi", size=9, align_center_cols=set(range(1, 10)))
    pg = fa["per_genus"]
    asp = {g: pg[g]["Tỉ lệ cao/rộng (HCN cơ bản)"][0] for g in genera}
    hue = {g: pg[g]["Sắc độ H trung bình vùng cây (°)"][0] for g in genera}
    sol = {g: pg[g]["Solidity (độ đặc tán)"][0] for g in genera}
    tall, wide = max(asp, key=asp.get), min(asp, key=asp.get)
    r.bullets([
        ("Khác biệt về dáng: ", f"tỉ lệ cao/rộng trung bình cao nhất ở {tall} ({_f(asp[tall], 2)}) và thấp nhất ở "
                                f"{wide} ({_f(asp[wide], 2)}); solidity (độ đặc của tán) cao nhất ở "
                                f"{max(sol, key=sol.get)} ({_f(max(sol.values()), 2)}), thấp nhất ở "
                                f"{min(sol, key=sol.get)} ({_f(min(sol.values()), 2)})."),
        ("Khác biệt về màu: ", f"sắc độ trung bình vùng cây dao động từ {_f(min(hue.values()), 0)}° "
                               f"({min(hue, key=hue.get)}) đến {_f(max(hue.values()), 0)}° ({max(hue, key=hue.get)}), "
                               "tức từ vàng lục (lá vàng, cây trụi lá lộ cành nâu) đến lục lam (lá kim, lá xanh đậm)."),
        ("Giống nhau trong cùng chi nhưng chồng lấn giữa các chi: ", "độ lệch chuẩn trong từng chi (Hình bên dưới) "
                                                                    "lớn gần bằng khoảng cách giữa các trung bình, "
                                                                    "nên không một thuộc tính đơn lẻ nào đủ để phân "
                                                                    "biệt chi: cần kết hợp nhiều bộ đặc trưng."),
    ])
    gr = fa["groups"]
    rows = [[k, _f(gr["broadleaf"][k][0], 3), _f(gr["conifer"][k][0], 3)] for k in keys]
    r.table(["Thuộc tính", f"Lá rộng (n={gr['broadleaf']['n']})", f"Lá kim (n={gr['conifer']['n']})"], rows,
            widths=[7, 4, 4], caption="So sánh trung bình giữa nhóm lá rộng và lá kim", size=9.5,
            align_center_cols={1, 2})
    r.p("Lá kim tối hơn (V thấp hơn), tán kém đặc và mật độ cạnh thấp hơn (kết cấu mịn, đều) so với lá rộng; đây "
        "là các khác biệt mà bộ đặc trưng màu và kết cấu khai thác.")
    r.figure(FIG / "15_genus_feature_boxplots.png", "Phân bố bốn thuộc tính theo chi: hộp = khoảng tứ phân vị, "
             "vạch giữa = trung vị", 16)


def sec_info_value(r: Report, fa: dict, num: str):
    """Muc gia tri thong tin dinh luong cua tung khoi / bo dac trung."""
    r.h(2, f"{num}. Đánh giá định lượng giá trị thông tin của các đặc trưng")
    r.p("Giá trị thông tin của một đặc trưng là mức độ nó giúp phân biệt các cây khác loại trong khi vẫn giữ các "
        "cây cùng loại ở gần nhau. Nhóm đo bằng ba chỉ số trên các ảnh có nhãn chi, mỗi khối dùng đúng độ đo "
        "khoảng cách của bộ chứa nó:")
    r.bullets([
        ("Tỉ số Fisher: ", "trung bình theo từng chiều (đã chuẩn hoá z-score) của tỉ số phương sai giữa các chi / "
                           "phương sai trong cùng chi. Càng lớn, chiều đó càng tách các chi."),
        ("Tỉ số tách: ", "khoảng cách trung bình giữa hai ảnh khác chi chia cho khoảng cách trung bình giữa hai ảnh "
                         "cùng chi. Lớn hơn 1 nghĩa là đặc trưng đặt ảnh cùng chi gần nhau hơn."),
        ("Độ chính xác 1-NN: ", f"tỉ lệ ảnh mà ảnh gần nhất (bỏ chính nó) có cùng chi. Mức ngẫu nhiên (hoán vị nhãn) "
                                f"là {_f(fa['nn1_chance'], 3)}."),
    ])
    xa = "nn1_acc_xauthor" in fa["sets"][next(iter(fa["sets"]))]
    if xa:
        r.bullets([("1-NN khác tác giả: ", "như 1-NN nhưng không cho phép ảnh gần nhất là ảnh của cùng tác giả. Bộ dữ "
                                           "liệu có nhiều loạt ảnh chụp cùng một cây (mục 1.5.2), nên chỉ số này phản "
                                           "ánh đúng hơn khả năng nhận ra cây cùng chi nhưng là cây khác.")])
    rows = []
    for b in sorted(fa["blocks"], key=lambda x: -x["nn1_acc"]):
        rows.append([SET_VI[b["set"]], f"{b['block']} ({BLOCK_VI.get(b['block'], '')})", str(b["dim"]),
                     b["metric"].upper(), _f(b["fisher"]), _f(b["sep_ratio"]), _f(b["nn1_acc"])]
                    + ([_f(b["nn1_acc_xauthor"])] if xa else []))
    r.table(["Bộ", "Khối", "Chiều", "Độ đo", "Fisher", "Tỉ số tách", "1-NN"] + (["1-NN khác TG"] if xa else []), rows,
            widths=[2.4, 4.6, 1.3, 1.5, 1.5, 1.6, 1.4] + ([1.8] if xa else []),
            caption="Giá trị thông tin của từng khối đặc trưng (xếp theo độ chính xác 1-NN)", size=9,
            align_center_cols=set(range(2, 8)))
    sets = fa["sets"]
    r.table(["Bộ đặc trưng", "Chiều", "Fisher", "Tỉ số tách", "1-NN"] + (["1-NN khác TG"] if xa else []),
            [[SET_VI[s], str(v["dim"]), _f(v["fisher"]), _f(v["sep_ratio"]), _f(v["nn1_acc"])]
             + ([_f(v["nn1_acc_xauthor"])] if xa else []) for s, v in sets.items()],
            widths=[4.0, 1.8, 2.2, 2.2, 2.2] + ([2.6] if xa else []), caption="Giá trị thông tin của cả bộ đặc trưng",
            size=9.5, align_center_cols=set(range(1, 6)))
    blocks = sorted(fa["blocks"], key=lambda x: -x["nn1_acc"])
    best_fisher = max(fa["blocks"], key=lambda x: x["fisher"])
    weak = [b for b in fa["blocks"] if b["sep_ratio"] < 1.03]
    r.p("Nhận xét:", bold=True)
    r.bullets([
        f"Mọi bộ đặc trưng đều có tỉ số tách > 1 và 1-NN cao hơn mức ngẫu nhiên ({_f(fa['nn1_chance'], 3)}), tức "
        f"đều mang thông tin về loài cây. Khối có 1-NN cao nhất là {blocks[0]['block']} "
        f"({SET_VI[blocks[0]['set']].lower()}, {_f(blocks[0]['nn1_acc'])}), gấp "
        f"{_f(blocks[0]['nn1_acc'] / max(fa['nn1_chance'], 1e-9), 1)} lần ngẫu nhiên.",
        f"Khối có tỉ số Fisher lớn nhất là {best_fisher['block']} ({_f(best_fisher['fisher'])}): ít chiều nhưng mỗi "
        f"chiều tách chi tốt, nên là đặc trưng “tìm sự khác biệt” rất hiệu quả trên mỗi chiều lưu trữ.",
        "Biểu đồ màu vùng cây (hsv_fg) giữ thông tin tốt hơn biểu đồ nền (hsv_bg): tách vật thể khỏi nền đúng như "
        "lý thuyết khắc phục hiệu ứng che mặt của nền.",
        "Các khối hình dạng có 1-NN thấp hơn khi dùng riêng vì cùng một chi có nhiều dáng (tuổi cây, cắt tỉa, gió) "
        "và hình bao chỉ là hình chiếu 2D; chúng vẫn bổ sung thông tin độc lập với màu (không phụ thuộc mùa, ánh "
        "sáng) nên được giữ trong tổ hợp có trọng số."
        + (f" Các khối có tỉ số tách gần 1 ({', '.join(b['block'] for b in weak)}) mang ít thông tin về chi và "
           f"chủ yếu giúp tìm ảnh cùng bối cảnh/bố cục." if weak else ""),
    ] + ([_xauthor_note(fa)] if xa else []))


def _xauthor_note(fa: dict) -> str:
    sets = fa["sets"]
    best = max(sets, key=lambda s: sets[s]["nn1_acc_xauthor"])
    blocks = sorted(fa["blocks"], key=lambda b: b["nn1_acc"] - b["nn1_acc_xauthor"])
    small, big = blocks[:3], blocks[-2:][::-1]
    top = max(fa["blocks"], key=lambda b: b["nn1_acc_xauthor"])
    fmt = lambda bs: ", ".join(f"{b['block']} ({_f(b['nn1_acc'])} → {_f(b['nn1_acc_xauthor'])})" for b in bs)  # noqa: E731
    return (f"Khi loại ảnh cùng tác giả, độ chính xác 1-NN của mọi bộ đều giảm, tức một phần kết quả 1-NN đến từ việc "
            f"tìm lại chính cây đó trong cùng loạt ảnh. Giảm mạnh nhất là {fmt(big)}: ảnh cùng loạt chụp có cùng ánh "
            f"sáng, mùa và bối cảnh nên màu và embedding gần nhau; giảm ít nhất là {fmt(small)}. Dù vậy mọi bộ vẫn "
            f"trên mức ngẫu nhiên ({_f(fa['nn1_chance'])}); bộ cao nhất là {SET_VI[best].lower()} "
            f"({_f(sets[best]['nn1_acc_xauthor'])}), khối cao nhất là {top['block']} ({_f(top['nn1_acc_xauthor'])}).")


def sec_example(r: Report, fa: dict, num: str):
    """Vi du minh hoa: A, B cung chi (khac cay), C khac chi -> gia tri dac trung va khoang cach."""
    ex = fa["example"]
    g = ex["genera"]
    r.h(2, f"{num}. Ví dụ minh hoạ: đặc trưng tìm sự tương đồng và sự khác biệt")
    r.p(f"Chọn ba ảnh trong CSDL: A và B cùng chi {g[0]} (hai cây khác nhau, khác tác giả; B là ảnh cùng chi có "
        f"hình dạng gần A nhất), C thuộc chi {g[2]} (ảnh tiêu biểu nhất của chi theo hình dạng). Bảng dưới liệt kê "
        "giá trị một số đặc trưng và khoảng cách giữa từng cặp theo mỗi bộ đặc trưng.")
    r.figure(FIG / "16_similarity_triplet.png", f"Ba ảnh minh hoạ: A, B ({g[0]}) và C ({g[2]})", 15)
    rows = [[k] + [_f(v, 1 if k.startswith("Sắc độ H") else 3) for v in vals] for k, vals in ex["values"].items()]
    r.table(["Đặc trưng", "Ảnh A", "Ảnh B", "Ảnh C"], rows, widths=[7, 3, 3, 3],
            caption="Giá trị đặc trưng của ba ảnh minh hoạ", size=9.5, align_center_cols={1, 2, 3})
    d = ex["distances"]
    rows = [[SET_VI[s], v["metric"].upper(), _f(v["AB"]), _f(v["AC"]), _f(v["BC"]),
             "đúng" if v["AB"] < min(v["AC"], v["BC"]) else "không"] for s, v in d.items()]
    r.table(["Bộ đặc trưng", "Độ đo", "d(A,B)", "d(A,C)", "d(B,C)", "A–B gần nhất?"], rows,
            widths=[3.6, 1.8, 2.2, 2.2, 2.2, 3], caption="Khoảng cách giữa các cặp ảnh theo từng bộ đặc trưng",
            size=9.5, align_center_cols={1, 2, 3, 4, 5})
    ok = [SET_VI[s].lower() for s, v in d.items() if v["AB"] < min(v["AC"], v["BC"])]
    bad = [SET_VI[s].lower() for s, v in d.items() if not v["AB"] < min(v["AC"], v["BC"])]
    r.p(f"Theo các bộ {', '.join(ok)}, hai cây cùng chi A–B gần nhau hơn so với cây khác chi C: các đặc trưng này "
        "vừa giúp tìm sự tương đồng (A–B nhỏ) vừa tìm sự khác biệt (A–C, B–C lớn)."
        + (f" Riêng bộ {', '.join(bad)} cho kết quả ngược lại ở ví dụ này, minh hoạ rằng không bộ đặc trưng nào "
           "đúng trong mọi trường hợp và vì sao hệ thống kết hợp nhiều bộ theo trọng số." if bad else ""))
