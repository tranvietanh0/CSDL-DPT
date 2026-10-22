"""Demo hệ thống: giao diện Streamlit cho CSDL ảnh cây (CBIR) - chạy: streamlit run app/streamlit_app.py

Bố cục theo kiểu bài báo / công cụ thực nghiệm: đầu trang (tiêu đề, nhóm, tóm tắt), các mục đánh số,
hình và bảng có chú thích, công thức hiển thị bằng LaTeX, tên chi viết nghiêng.
"""
from __future__ import annotations

import html
import importlib
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BACKENDS = ("linear", "kdtree", "rtree", "quadtree", "kmeans")
BACKEND_LABELS = {"linear": "Quét tuyến tính (linear scan)", "kdtree": "Cây k-d trên PCA",
                  "rtree": "Cây R (nạp lô STR)", "quadtree": "Cây tứ phân PR", "kmeans": "Phân cụm K-means"}
SET_LABELS = {"color": "Màu sắc", "shape": "Hình dạng", "texture": "Kết cấu", "deep": "Embedding sâu"}
FILTER_LABELS = {"genus": "Chi (genus)", "group_label": "Nhóm hình thái", "license": "Giấy phép"}
FEEDBACK_CHOICES = ("·", "R", "N")   # chưa đánh giá / liên quan (R) / không liên quan (N) – ký hiệu Rocchio
GRID_COLS = 5
LOAD_ERRORS = (RuntimeError, ImportError, OSError, sqlite3.Error)  # CSDL trống -> báo lỗi thân thiện


def _mod(name: str):
    """Nạp một mô-đun treedb (import trễ để trang không chết khi gói chưa sẵn sàng)."""
    return importlib.import_module(f"treedb.{name}")


def esc(x) -> str:
    return html.escape(str(x))


@st.cache_resource(show_spinner="Đang nạp CSDL đặc trưng…")
def load_backend():
    """Một Searcher dùng chung cho cả phiên. Kết nối phải dùng được giữa các luồng, vì Streamlit chạy
    mỗi lượt tương tác ở một luồng khác còn sqlite3 mặc định buộc kết nối với luồng đã tạo ra nó."""
    from treedb.search import Searcher

    module = _mod("db")
    conn = sqlite3.connect(str(_mod("config").DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row  # phần còn lại giống db.connect()
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    module.init_schema(conn)
    return Searcher(conn=conn), conn


def stop_with_load_error(exc: Exception) -> None:
    """Dừng trang kèm hướng dẫn khắc phục (thay cho traceback)."""
    if isinstance(exc, RuntimeError):  # FeatureSpace báo trong CSDL chưa có bộ đặc trưng nào
        st.error("CSDL đặc trưng đang trống hoặc chưa được tạo nên chưa thể tìm kiếm – hãy chạy lần lượt:\n\n"
                 "```\npython scripts/extract_features.py\npython scripts/build_index.py\n```")
    else:  # lỗi môi trường: thiếu gói, chặn DLL, tệp CSDL hỏng...
        st.error("Không nạp được treedb hoặc tệp CSDL – kiểm tra `pip install -r requirements.txt` và db/tree_features.sqlite.")
    st.caption(f"Chi tiết kỹ thuật: {type(exc).__name__}: {exc}")
    st.stop()


@st.cache_data(show_spinner=False)
def sample_images(size: int = 30) -> list[dict]:  # 30 ảnh lấy ngẫu nhiên để chọn làm ảnh truy vấn
    """Lấy ngẫu nhiên (ổn định giữa các lần chạy lại) một số ảnh trong CSDL."""  # seed cố định = tái lập
    rows = [dict(r) for r in _mod("db").list_images(_mod("db").connect())]
    random.Random(42).shuffle(rows)
    return [{k: r.get(k) for k in ("image_id", "filename", "path", "genus")} for r in rows[:size]]


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    path = Path(_mod("config").RESULTS_DIR) / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class Numbering:
    """Đánh số hình / bảng theo thứ tự xuất hiện trong trang (như trong bài báo)."""

    def __init__(self):
        self.fig = self.tab = 0

    def figure(self, text: str) -> str:
        self.fig += 1
        return f'<div class="cap"><b>Hình {self.fig}.</b> {text}</div>'

    def table(self, text: str) -> str:
        self.tab += 1
        return f'<div class="cap"><b>Bảng {self.tab}.</b> {text}</div>'


def inject_style() -> None:
    st.html(f"<style>{(Path(__file__).with_name('style.css')).read_text(encoding='utf-8')}</style>")


def section(title: str) -> None:
    st.html(f'<div class="sec">{esc(title)}</div>')


# ---- Thanh bên: tham số thực nghiệm --------------------------------------------------------
def side_section(title: str) -> None:
    st.sidebar.html(f'<div class="side-sec">{esc(title)}</div>')


def weight_sliders(names: list[str]) -> dict:
    """Một slider cho mỗi bộ đặc trưng; trả về trọng số đã chuẩn hoá (tổng = 1)."""
    if not names:
        return {}
    defaults = _mod("config").DEFAULT_WEIGHTS  # trọng số mặc định trong cấu hình dự án
    raw = {n: st.sidebar.slider(f"w({SET_LABELS.get(n, n).lower()})", 0.0, 1.0, float(defaults.get(n, 0.0)), 0.05,
                                key=f"w_{n}") for n in names}
    total = sum(raw.values())
    if total <= 0.0:
        st.sidebar.warning("Tổng trọng số bằng 0 – tạm dùng trọng số đều nhau cho mọi bộ.")
        return {n: 1.0 / len(raw) for n in raw}
    return {n: v / total for n, v in raw.items()}  # tổng = 1


def filter_widgets(searcher) -> dict:
    """Lọc kết quả theo siêu dữ liệu quan hệ (truy vấn kết hợp thuộc tính + nội dung)."""
    values = searcher.filter_values()
    return {f: set(st.sidebar.multiselect(label, values.get(f, []), placeholder="không lọc"))
            for f, label in FILTER_LABELS.items()}


def render_sidebar(searcher) -> dict:
    """Vẽ thanh bên và trả về cấu hình truy vấn người dùng đã chọn."""
    st.sidebar.html('<div class="side-title">Tham số thực nghiệm</div>'
                    '<div class="side-sub">Thay đổi rồi chạy lại truy vấn để so sánh.</div>')
    qp = st.query_params  # ?backend=kmeans&debug=1 -> giá trị mặc định (phục vụ demo / chụp màn hình)
    side_section("Chọn ứng viên")
    backend = st.sidebar.selectbox("Cấu trúc chỉ mục", BACKENDS, format_func=BACKEND_LABELS.get,
                                   index=BACKENDS.index(qp.get("backend")) if qp.get("backend") in BACKENDS else 0)
    top_k = st.sidebar.slider("Số kết quả K", 1, 10, int(_mod("config").TOP_K))
    side_section("Trọng số kết hợp w")
    weights = weight_sliders(list(searcher.space.sets))
    st.sidebar.caption("Chuẩn hoá: " + ", ".join(f"{v:.2f}" for v in weights.values()) + " (Σw = 1)")
    side_section("Ràng buộc siêu dữ liệu")
    filters = filter_widgets(searcher)
    side_section("Hiển thị")
    debug = st.sidebar.toggle("Vẽ biểu đồ giá trị trung gian", value=qp.get("debug") == "1")
    return {"backend": backend, "top_k": top_k, "weights": weights, "show_debug": debug,
            "filters": {k: v for k, v in filters.items() if v}}


def render_header(conn, searcher) -> None:
    summary = _mod("db").summary(conn) or {}
    sets = summary.get("feature_sets", {})
    ev = load_json("evaluation.json")
    comb = ev.get("feature_sets", {}).get("combined(default)", {})
    kv = [("N", f"{int(summary.get('images') or 0)} ảnh"), ("Số chi có nhãn", len(searcher.filter_values().get("genus", []))),
          ("Bộ đặc trưng", " + ".join(f"{s} ({v['dim']})" for s, v in sets.items())),
          ("P@5 (leave-one-out)", f"{comb.get('p_at_k', 0):.3f}" if comb else "—")]
    st.html('<div class="paper-head"><div class="venue">Hệ cơ sở dữ liệu đa phương tiện · Bài tập lớn</div>'
            '<h1>Hệ CSDL lưu trữ và tìm kiếm ảnh cây dựa trên nội dung</h1>'
            '<div class="authors">Nhóm 27 <span>· D23 – Lớp 2 · Học viện Công nghệ Bưu chính Viễn thông</span></div>'
            '<div class="abstract"><b>Tóm tắt.</b> Ảnh truy vấn được chuẩn hoá về 512×512, tách vùng cây bằng '
            'phân đoạn ngữ nghĩa, rồi mô tả bằng bốn bộ đặc trưng: màu sắc (biểu đồ HSV, luỹ tích, PWH), hình dạng '
            '(moment Hu, lưới 8×8, Fourier), kết cấu (GLCM, LBP, Tamura, DCT) và embedding sâu MobileNetV2. '
            'Khoảng cách từng bộ được chuẩn hoá và kết hợp tuyến tính có trọng số; ứng viên được chọn bằng quét '
            'tuyến tính hoặc một trong bốn cấu trúc chỉ mục; kết quả có thể ràng buộc theo siêu dữ liệu và tinh '
            'chỉnh bằng phản hồi liên quan (Rocchio).</div><div class="kv">'
            + "".join(f"<span>{esc(k)}: <b>{esc(v)}</b></span>" for k, v in kv) + "</div></div>")


# ---- Ảnh truy vấn -------------------------------------------------------------------------
def save_upload(upload) -> str:
    """Ghi ảnh tải lên vào results/tmp và trả về đường dẫn tạm."""
    tmp_dir = Path(_mod("config").RESULTS_DIR) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"upload_{time.strftime('%Y%m%d_%H%M%S')}_{Path(upload.name).name or 'query.png'}"
    path.write_bytes(upload.getbuffer())
    return str(path)


def pick_query_source() -> tuple[str | None, int | None]:
    """Trả về (đường dẫn ảnh truy vấn, image_id cần loại khỏi kết quả hoặc None)."""
    mode = st.segmented_control("Nguồn ảnh", ["Tải ảnh lên", "Ảnh trong CSDL"], default="Tải ảnh lên",
                                label_visibility="collapsed")
    if mode != "Ảnh trong CSDL":
        upload = st.file_uploader("Ảnh cây (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
        if upload is None:
            st.html('<div class="note">Yêu cầu: một cây trưởng thành, chụp ngang, nằm giữa khung hình.</div>')
            return None, None
        st.image(upload, width="stretch")
        return save_upload(upload), None
    if not (rows := sample_images()):
        st.info("CSDL chưa có ảnh nào để chọn.")
        return None, None
    pick = st.selectbox("Ảnh trong CSDL", range(len(rows)),
                        format_func=lambda i: f"{rows[i]['filename']} — {rows[i]['genus'] or 'chưa rõ chi'}")
    row = rows[pick]
    path = Path(str(row["path"]))
    if not path.exists():  # đường dẫn lưu trong CSDL không còn đúng -> ghép lại từ IMG_DIR
        path = Path(_mod("config").IMG_DIR) / str(row["filename"])
    st.image(str(path), width="stretch")
    exclude = st.checkbox("Loại chính ảnh này khỏi kết quả (leave-one-out)", value=True,
                          help="Ảnh vốn nằm trong CSDL: nếu không loại, nó luôn đứng hạng 1 với độ tương đồng 1,0.")
    return str(path), (int(row["image_id"]) if exclude else None)


def render_query(res: dict, num: Numbering) -> None:
    """Ảnh truy vấn + vùng cây và bảng thông số truy vấn."""
    q = res["query"]
    if "rgb" in q:
        a, b = st.columns(2)
        a.image(q["rgb"], width="stretch")
        b.image(_mod("segment").overlay(q["rgb"], q["mask"]), width="stretch")
        st.html(num.figure("Ảnh truy vấn sau chuẩn hoá 512×512 (trái) và mặt nạ vùng cây dùng cho đặc trưng "
                           "vật thể (phải, tô đỏ)."))
    total = res["timing_ms"].get("total", sum(res["timing_ms"].values()))
    params = pd.DataFrame([
        {"thông số": "Tỉ lệ vùng cây", "giá trị": f"{q.get('mask_coverage', 0):.1%}"},
        {"thông số": "Cấu trúc chỉ mục", "giá trị": BACKEND_LABELS.get(res["backend"], res["backend"])},
        {"thông số": "Ứng viên đã xét / |CSDL|", "giá trị": f"{res['n_candidates']} / {res['n_database']}"},
        {"thông số": "Trọng số w", "giá trị": ", ".join(f"{s} {w:.2f}" for s, w in res["weights"].items())},
        {"thông số": "Thời gian xử lý", "giá trị": f"{total:.0f} ms"},
    ] + ([{"thông số": "Ràng buộc", "giá trị": "; ".join(f"{FILTER_LABELS[k]} ∈ {{{', '.join(v)}}}"
                                                         for k, v in res["filters"].items())}]
         if res.get("filters") else []))
    st.html(num.table("Thông số của lần truy vấn."))
    st.dataframe(params, hide_index=True, width="stretch")


# ---- Kết quả ------------------------------------------------------------------------------
def results_table(res: dict) -> pd.DataFrame:
    """Bảng: hạng, tệp, chi, nhóm, độ tương đồng, D và d/scale, w·d/scale theo từng bộ."""
    rows = []
    for r in res["results"]:
        row = {"hạng": r["rank"], "tệp": r["filename"], "chi": r["genus"] or "", "nhóm": r["group"] or "",
               "sim": round(r["similarity"], 4), "D": round(r["distance"], 4)}
        for s in res["sets"]:
            if r["per_set"].get(s):
                row[f"d̃ {s}"] = round(r["per_set"][s]["norm"], 4)
                row[f"w·d̃ {s}"] = round(r["per_set"][s]["contrib"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def result_card(r: dict, sets: list[str], round_no: int) -> str:
    """Một ô kết quả (ảnh + số liệu); trả về đánh giá phản hồi người dùng chọn."""
    with st.container(border=True):
        genus = f"<i>{esc(r['genus'])}</i>" if r["genus"] else "<span class='note'>chưa rõ chi</span>"
        st.html(f'<div class="res-head"><span class="rk">Hạng {r["rank"]}</span>{genus}</div>')
        if Path(str(r["path"])).exists():
            st.image(str(r["path"]), width="stretch")
        else:
            st.warning(f"Không tìm thấy tệp ảnh: {r['path']}")
        rows = "".join(f"<tr><td>d̃ {esc(s)}</td><td>{r['per_set'][s]['norm']:.3f}</td></tr>"
                       for s in sets if s in r["per_set"])
        st.html(f'<table class="res-tab"><tr class="main"><td>sim</td><td>{r["similarity"]:.4f}</td></tr>'
                f'<tr><td>D</td><td>{r["distance"]:.4f}</td></tr>{rows}'
                f'<tr><td colspan="2" class="note">{esc(r["filename"])}</td></tr></table>')
        return st.segmented_control("Đánh giá", FEEDBACK_CHOICES, default=FEEDBACK_CHOICES[0],
                                    key=f"fb_{round_no}_{r['image_id']}", label_visibility="collapsed") \
            or FEEDBACK_CHOICES[0]


def render_results(res: dict, num: Numbering) -> dict[int, str]:
    """Lưới kết quả (kèm ô đánh giá phản hồi); trả về {image_id: đánh giá}."""
    if not res["results"]:
        st.warning("Không có kết quả nào (CSDL quá nhỏ, ràng buộc quá chặt hoặc đã loại hết ứng viên).")
        return {}
    marks = {}
    round_no = len(st.session_state.get("feedback_history", []))
    items = res["results"]
    for start in range(0, len(items), GRID_COLS):
        for col, r in zip(st.columns(GRID_COLS), items[start:start + GRID_COLS]):
            with col:
                marks[r["image_id"]] = result_card(r, res["sets"], round_no)
    st.html(num.figure(f"{len(items)} ảnh có độ tương đồng cao nhất, xếp giảm dần theo sim = 1/(1 + D); d̃ là "
                       "khoảng cách đã chuẩn hoá của từng bộ đặc trưng. Ô dưới mỗi ảnh dùng để đánh giá phản "
                       "hồi: R – liên quan, N – không liên quan."))
    return marks


def render_feedback(searcher, res: dict, marks: dict[int, str]) -> None:
    """Truy vấn lại theo công thức Rocchio với các ảnh người dùng vừa đánh dấu."""
    rel = [i for i, m in marks.items() if m == "R"]
    non = [i for i, m in marks.items() if m == "N"]
    hist = st.session_state.get("feedback_history") or []
    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        st.latex(r"q' = \frac{\alpha\,q + \beta\,\bar{R} - \gamma\,\bar{N}}{\alpha + \beta - \gamma},"
                 r"\quad \alpha = 1,\ \beta = 0.75,\ \gamma = 0.25")
        st.caption(f"Đã đánh dấu |R| = {len(rel)}, |N| = {len(non)}"
                   + (f" · đã thực hiện {len(hist)} vòng phản hồi" if hist else ""))
    if right.button("Truy vấn lại với phản hồi", disabled=not (rel or non), width="stretch"):
        opts = st.session_state.get("last_options") or {}
        new = searcher.refine(res, rel, non, top_k=opts.get("top_k", 5), backend=opts.get("backend", "linear"),
                              exclude_ids=st.session_state.get("last_exclude"), filters=opts.get("filters"))
        st.session_state.setdefault("feedback_history", []).append({"relevant": rel, "nonrelevant": non})
        st.session_state["last_result"] = new
        st.rerun()


def render_details(res: dict, show_debug: bool, num: Numbering) -> None:
    """Các tab chi tiết: khoảng cách, công thức, thời gian, giá trị trung gian, chỉ mục, đánh giá."""
    tabs = st.tabs(["Bảng khoảng cách", "Công thức", "Thời gian xử lý", "Giá trị trung gian", "Chỉ mục",
                    "Đánh giá thực nghiệm", "Dạng văn bản"])
    with tabs[0]:
        st.html(num.table("Khoảng cách theo từng bộ đặc trưng: d̃ = d/scale và đóng góp w·d̃ vào D."))
        st.dataframe(results_table(res), hide_index=True, width="stretch")
    with tabs[1]:
        st.latex(r"\tilde d_s(q, x) = \frac{d_s(q, x)}{\mathrm{scale}_s},\qquad "
                 r"D(q, x) = \sum_s w_s\,\tilde d_s(q, x),\qquad \mathrm{sim}(q, x) = \frac{1}{1 + D(q, x)}")
        st.latex(r"d_{L1}(X, Y) = \sum_j |x_j - y_j|,\quad d_{L2}(X, Y) = \sqrt{\sum_j (x_j - y_j)^2},\quad "
                 r"d_{\cos}(X, Y) = 1 - \frac{X \cdot Y}{\|X\|\,\|Y\|}")
        st.caption("Độ đo: màu sắc – L1 trên biểu đồ; hình dạng, kết cấu – L2 trên z-score; embedding – cosine. "
                   "scale_s là khoảng cách trung bình giữa hai ảnh ngẫu nhiên trong CSDL.")
    with tabs[2]:
        t = pd.DataFrame([{"bước": k, "ms": round(v, 2)} for k, v in res["timing_ms"].items()])
        st.bar_chart(t[t["bước"] != "total"], x="bước", y="ms", color="#1f3a5f", horizontal=True)
        st.html(num.figure("Thời gian từng bước của lần truy vấn (ms)."))
    with tabs[3]:
        if show_debug and "rgb" in res["query"]:
            render_debug_plots(res, num)
        else:
            st.caption("Bật “Vẽ biểu đồ giá trị trung gian” ở thanh bên để xem biểu đồ HSV, lưới hình dạng, LBP…")
        for name, feature in res["query"]["features"].items():
            if "blocks" not in feature:
                continue
            with st.expander(f"Vector đặc trưng – {SET_LABELS.get(name, name)} ({feature['vector'].size} chiều)"):
                # các khối có số chiều khác nhau -> dùng Series để pandas tự căn hàng (NaN ở ô thiếu)
                st.dataframe(pd.DataFrame({n: pd.Series([round(float(x), 4) for x in v])
                                           for n, v in feature["blocks"].items()}))
    with tabs[4]:
        if res["backend"] == "linear":
            st.caption("Quét tuyến tính không dùng chỉ mục – mọi ảnh trong CSDL đều được tính khoảng cách.")
        else:
            st.json(res["index_info"], expanded=False)
    with tabs[5]:
        render_evaluation(num)
    with tabs[6]:
        st.code(_mod("search").explain(res))


def render_evaluation(num: Numbering) -> None:
    ev = load_json("evaluation.json")
    if not ev:
        st.caption("Chưa có results/evaluation.json – chạy python scripts/run_evaluation.py.")
        return
    k = ev["k"]
    st.html(num.table(f"Độ chính xác leave-one-out trên {ev['n_queries']} truy vấn có nhãn chi (quét tuyến tính); "
                      f"ngẫu nhiên P@{k} = {ev['random_baseline']['p_at_k']:.3f}."))
    st.dataframe(pd.DataFrame([{"cấu hình": n, f"P@{k}": round(v["p_at_k"], 3), "P@1": round(v["p_at_1"], 3),
                                f"mAP@{k}": round(v["ap_at_k"], 3), f"P@{k} nhóm": round(v["group_p_at_k"], 3)}
                               for n, v in ev["feature_sets"].items()]), hide_index=True, width="stretch")
    st.html(num.table("So sánh các cấu trúc chọn ứng viên (trọng số mặc định)."))
    st.dataframe(pd.DataFrame([{"chỉ mục": BACKEND_LABELS.get(b, b), f"recall@{k} so với tuyến tính":
                                round(v["recall_vs_linear"], 3), f"P@{k}": round(v["p_at_k"], 3),
                                "ms / truy vấn": round(v["ms_per_query"], 2)} for b, v in ev["backends"].items()]),
                 hide_index=True, width="stretch")
    if fb := ev.get("feedback"):
        st.caption(f"Phản hồi liên quan (mô phỏng 1 vòng, {fb['n_queries']} truy vấn): P@{k} "
                   f"{fb['p_at_k_before']:.3f} → {fb['p_at_k_after']:.3f}.")


def render_debug_plots(res: dict, num: Numbering) -> None:
    from treedb import visualize

    tmp_dir = Path(_mod("config").RESULTS_DIR) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    st.image(str(visualize.plot_query_features(res, tmp_dir / f"query_features_{stamp}.png")), width="stretch")
    st.html(num.figure("Giá trị trung gian của các bộ đặc trưng trích rút từ ảnh truy vấn."))


def run_query(searcher, options: dict, path: str, exclude_id: int | None) -> dict | None:
    """Chạy một truy vấn; trả về kết quả hoặc None nếu lỗi."""
    try:
        with st.spinner("Đang phân đoạn, trích rút đặc trưng và so khớp…"):
            return searcher.query(path, top_k=options["top_k"], backend=options["backend"],
                                  weights=options["weights"], exclude_ids={exclude_id} if exclude_id else None,
                                  filters=options["filters"] or None)
    except LOAD_ERRORS + (ValueError,) as exc:
        st.error(f"Không truy vấn được ảnh này: {type(exc).__name__}: {exc}")
        return None


def remember(result: dict, options: dict, exclude: set[int] | None) -> None:
    st.session_state.update(last_result=result, last_options=options, feedback_history=[], last_exclude=exclude)


def main() -> None:
    st.set_page_config(page_title="CBIR ảnh cây – Nhóm 27", layout="wide")
    inject_style()
    try:
        searcher, conn = load_backend()
    except LOAD_ERRORS as exc:
        stop_with_load_error(exc)
    options = render_sidebar(searcher)
    render_header(conn, searcher)
    num = Numbering()

    left, right = st.columns([1, 2], gap="large")
    with left:
        section("1. Ảnh truy vấn")
        query_path, exclude_id = pick_query_source()
        if st.button("Thực hiện truy vấn", type="primary", disabled=query_path is None, width="stretch"):
            if (result := run_query(searcher, options, query_path, exclude_id)) is not None:
                remember(result, options, {exclude_id} if exclude_id else None)
    auto = st.query_params.get("auto")  # ?auto=query_01.jpg -> tự chạy một ảnh trong data/queries (phục vụ demo)
    if auto and "last_result" not in st.session_state:
        auto_path = Path(_mod("config").QUERY_DIR) / str(auto)
        if auto_path.exists() and (result := run_query(searcher, options, str(auto_path), None)) is not None:
            remember(result, options, None)
    res = st.session_state.get("last_result")
    with right:
        section("2. Tiền xử lý và thông số")
        if res is None:
            st.html('<div class="empty">Chọn ảnh truy vấn ở mục 1 rồi bấm <b>Thực hiện truy vấn</b>.</div>')
            return
        render_query(res, num)
    if any((st.session_state.get("last_options") or {}).get(k) != options[k] for k in ("backend", "top_k", "weights", "filters")):
        st.info("Tham số vừa thay đổi – bấm “Thực hiện truy vấn” để chạy lại (kết quả dưới đây của lần trước).")
    section(f"3. Kết quả: {len(res['results'])} ảnh tương đồng nhất")
    marks = render_results(res, num)
    section("4. Phản hồi liên quan")
    render_feedback(searcher, res, marks)
    section("5. Chi tiết tính toán")
    render_details(res, options["show_debug"], num)


main()
