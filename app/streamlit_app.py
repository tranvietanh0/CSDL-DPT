"""Demo hệ thống: giao diện Streamlit cho CSDL ảnh cây (CBIR) - chạy: streamlit run app/streamlit_app.py"""
from __future__ import annotations

import html
import importlib
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
BACKEND_LABELS = {"linear": "Quét toàn bộ (chính xác nhất)", "kdtree": "Cây k-d trên PCA", "rtree": "Cây R",
                  "quadtree": "Cây tứ phân", "kmeans": "Phân cụm K-means"}
SET_LABELS = {"color": "Màu sắc", "shape": "Hình dạng", "texture": "Kết cấu", "deep": "Embedding sâu"}
FILTER_LABELS = {"genus": "Chi (genus)", "group_label": "Nhóm hình thái", "license": "Giấy phép"}
FEEDBACK_CHOICES = ("–", "✓", "✗")
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


def inject_style() -> None:
    st.html(f"<style>{(Path(__file__).with_name('style.css')).read_text(encoding='utf-8')}</style>")


def section(num: str, title: str) -> None:
    st.html(f'<div class="section-title"><span class="num">{esc(num)}</span>{esc(title)}</div>')


# ---- Thanh bên: tham số truy vấn ---------------------------------------------------------
def side_section(title: str) -> None:
    st.sidebar.html(f'<div class="side-section">{esc(title)}</div>')


def weight_sliders(names: list[str]) -> dict:
    """Một slider cho mỗi bộ đặc trưng; trả về trọng số đã chuẩn hoá (tổng = 1)."""
    if not names:
        return {}
    defaults = _mod("config").DEFAULT_WEIGHTS  # trọng số mặc định trong cấu hình dự án
    raw = {n: st.sidebar.slider(SET_LABELS.get(n, n), 0.0, 1.0, float(defaults.get(n, 0.0)), 0.05, key=f"w_{n}")
           for n in names}
    total = sum(raw.values())
    if total <= 0.0:
        st.sidebar.warning("Tổng trọng số bằng 0 – tạm dùng trọng số đều nhau cho mọi bộ.")
        return {n: 1.0 / len(raw) for n in raw}
    return {n: v / total for n, v in raw.items()}  # tổng = 1


def filter_widgets(searcher) -> dict:
    """Lọc kết quả theo siêu dữ liệu quan hệ (truy vấn kết hợp thuộc tính + nội dung)."""
    values = searcher.filter_values()
    return {f: set(st.sidebar.multiselect(label, values.get(f, []), placeholder="Tất cả"))
            for f, label in FILTER_LABELS.items()}


def render_sidebar(searcher) -> dict:
    """Vẽ thanh bên và trả về cấu hình truy vấn người dùng đã chọn."""
    st.sidebar.html('<div class="side-brand"><div class="logo">🌳</div><div><div class="name">TreeDB</div>'
                    '<div class="sub">Hệ CSDL ảnh cây · Nhóm 27</div></div></div>')
    qp = st.query_params  # ?backend=kmeans&debug=1 -> giá trị mặc định (phục vụ demo / chụp màn hình)
    side_section("Tìm kiếm")
    backend = st.sidebar.selectbox("Phương pháp chọn ứng viên", BACKENDS, format_func=BACKEND_LABELS.get,
                                   index=BACKENDS.index(qp.get("backend")) if qp.get("backend") in BACKENDS else 0)
    top_k = st.sidebar.slider("Số kết quả (top-K)", 1, 10, int(_mod("config").TOP_K))
    side_section("Trọng số bộ đặc trưng")
    weights = weight_sliders(list(searcher.space.sets))
    side_section("Lọc theo siêu dữ liệu")
    filters = filter_widgets(searcher)
    side_section("Hiển thị")
    debug = st.sidebar.toggle("Vẽ biểu đồ giá trị trung gian", value=qp.get("debug") == "1")
    return {"backend": backend, "top_k": top_k, "weights": weights, "show_debug": debug,
            "filters": {k: v for k, v in filters.items() if v}}


def render_hero(conn, searcher) -> None:
    summary = _mod("db").summary(conn) or {}
    sets = summary.get("feature_sets", {})
    chips = [f"<b>{int(summary.get('images') or 0):,}</b> ảnh cây",
             f"<b>{len(sets)}</b> bộ đặc trưng · {sum(v['dim'] for v in sets.values()):,} chiều",
             f"<b>{len(BACKENDS)}</b> cấu trúc chỉ mục",
             f"<b>{len(searcher.filter_values().get('genus', []))}</b> chi cây có nhãn"]
    st.html('<div class="hero"><h1>🌳 Tìm kiếm ảnh cây theo nội dung</h1>'
            '<p>Tải lên một ảnh cây, hệ thống tách vùng cây, trích rút màu sắc, hình dạng, kết cấu và embedding '
            'sâu rồi trả về những cây giống nhất trong CSDL. Có thể lọc theo siêu dữ liệu và tinh chỉnh kết quả '
            'bằng phản hồi liên quan.</p><div class="chips">'
            + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div></div>")


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
    mode = st.segmented_control("Nguồn ảnh", ["📤 Tải lên", "🗂️ Từ CSDL"], default="📤 Tải lên",
                                label_visibility="collapsed")
    if mode != "🗂️ Từ CSDL":
        upload = st.file_uploader("Kéo thả hoặc chọn ảnh cây (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
        if upload is None:
            st.html('<div class="hint">Ảnh nên có một cây trưởng thành nằm giữa khung hình.</div>')
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
    exclude = st.checkbox("Loại chính ảnh này khỏi kết quả", value=True,
                          help="Ảnh vốn nằm trong CSDL: nếu không loại, nó luôn đứng hạng 1 với độ tương đồng 1,0.")
    return str(path), (int(row["image_id"]) if exclude else None)


def render_query(res: dict) -> None:
    """Ảnh truy vấn + vùng cây và các số liệu tóm tắt."""
    q = res["query"]
    if "rgb" in q:
        a, b = st.columns(2)
        a.image(q["rgb"], caption="Ảnh truy vấn đã chuẩn hoá", width="stretch")
        b.image(_mod("segment").overlay(q["rgb"], q["mask"]), caption="Vùng cây được tách (đỏ)", width="stretch")
    m = st.columns(3)
    m[0].metric("Vùng cây", f"{q.get('mask_coverage', 0):.0%}")
    m[1].metric("Ứng viên đã xét", f"{res['n_candidates']}/{res['n_database']}")
    m[2].metric("Thời gian", f"{res['timing_ms'].get('total', sum(res['timing_ms'].values())):.0f} ms")
    st.caption(BACKEND_LABELS.get(res["backend"], res["backend"]) + " · trọng số "
               + ", ".join(f"{SET_LABELS.get(s, s).lower()} {w:.2f}" for s, w in res["weights"].items()))


# ---- Kết quả ------------------------------------------------------------------------------
def results_table(res: dict) -> pd.DataFrame:
    """Bảng: hạng, tệp, chi, nhóm, độ tương đồng, D và d/scale, w·d/scale theo từng bộ."""
    rows = []
    for r in res["results"]:
        row = {"hạng": r["rank"], "tệp": r["filename"], "chi": r["genus"] or "", "nhóm": r["group"] or "",
               "độ tương đồng": round(r["similarity"], 4), "D (khoảng cách)": round(r["distance"], 4)}
        for s in res["sets"]:
            if r["per_set"].get(s):
                row[f"d/scale {s}"] = round(r["per_set"][s]["norm"], 4)
                row[f"w·d/scale {s}"] = round(r["per_set"][s]["contrib"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def result_card(r: dict, round_no: int) -> str:
    """Một thẻ kết quả; trả về đánh giá phản hồi người dùng chọn."""
    with st.container(border=True):
        st.html(f'<div class="res-head"><span class="rank{" top" if r["rank"] == 1 else ""}">#{r["rank"]}</span>'
                f'<span class="genus">{esc(r["genus"] or "chưa rõ chi")}</span></div>')
        if Path(str(r["path"])).exists():
            st.image(str(r["path"]), width="stretch")
        else:
            st.warning(f"Không tìm thấy tệp ảnh: {r['path']}")
        st.html(f'<div class="res-meta"><span class="lbl">Độ tương đồng</span><span class="val">{r["similarity"]:.3f}</span></div>'
                f'<div class="simbar"><span style="width:{max(0.0, min(1.0, r["similarity"])) * 100:.0f}%"></span></div>'
                f'<div class="fname">{esc(r["filename"])}</div>')
        return st.segmented_control("Phản hồi", FEEDBACK_CHOICES, default=FEEDBACK_CHOICES[0],
                                    key=f"fb_{round_no}_{r['image_id']}", label_visibility="collapsed") \
            or FEEDBACK_CHOICES[0]


def render_results(res: dict) -> dict[int, str]:
    """Lưới thẻ kết quả (kèm ô đánh giá phản hồi); trả về {image_id: đánh giá}."""
    if not res["results"]:
        st.warning("Không có kết quả nào (CSDL quá nhỏ, bộ lọc quá chặt hoặc đã loại hết ứng viên).")
        return {}
    marks = {}
    round_no = len(st.session_state.get("feedback_history", []))
    items = res["results"]
    for start in range(0, len(items), GRID_COLS):
        for col, r in zip(st.columns(GRID_COLS), items[start:start + GRID_COLS]):
            with col:
                marks[r["image_id"]] = result_card(r, round_no)
    return marks


def render_feedback(searcher, res: dict, marks: dict[int, str]) -> None:
    """Truy vấn lại theo công thức Rocchio với các ảnh người dùng vừa đánh dấu."""
    rel = [i for i, m in marks.items() if m == FEEDBACK_CHOICES[1]]
    non = [i for i, m in marks.items() if m == FEEDBACK_CHOICES[2]]
    with st.container(border=True):
        left, right = st.columns([3, 1], vertical_alignment="center")
        hist = st.session_state.get("feedback_history") or []
        left.markdown(f"**Phản hồi liên quan** · đã chọn {len(rel)} ✓, {len(non)} ✗"
                      + (f" · đã thực hiện {len(hist)} vòng" if hist else ""))
        left.caption("Đánh dấu ✓ (liên quan) / ✗ (không liên quan) dưới các ảnh rồi truy vấn lại: vector truy vấn "
                     "dịch về phía ảnh ✓ và ra xa ảnh ✗ theo công thức Rocchio q' = q + 0,75·tb(✓) − 0,25·tb(✗).")
        if right.button("🔁 Truy vấn lại", disabled=not (rel or non), width="stretch"):
            opts = st.session_state.get("last_options") or {}
            new = searcher.refine(res, rel, non, top_k=opts.get("top_k", 5), backend=opts.get("backend", "linear"),
                                  exclude_ids=st.session_state.get("last_exclude"), filters=opts.get("filters"))
            st.session_state.setdefault("feedback_history", []).append({"relevant": rel, "nonrelevant": non})
            st.session_state["last_result"] = new
            st.rerun()


def render_details(res: dict, show_debug: bool) -> None:
    """Các tab chi tiết: bảng khoảng cách, thời gian, giá trị trung gian, chỉ mục, văn bản."""
    tabs = st.tabs(["📊 Bảng khoảng cách", "⏱️ Thời gian xử lý", "🧬 Giá trị trung gian", "🌲 Chỉ mục", "📝 Dạng văn bản"])
    with tabs[0]:
        st.dataframe(results_table(res), hide_index=True, width="stretch")
        st.caption("d/scale: khoảng cách của từng bộ đã chuẩn hoá; w·d/scale: đóng góp vào D = Σ w·d/scale; "
                   "độ tương đồng sim = 1/(1 + D).")
    with tabs[1]:
        t = pd.DataFrame([{"bước": k, "thời gian (ms)": round(v, 2)} for k, v in res["timing_ms"].items()])
        st.bar_chart(t[t["bước"] != "total"], x="bước", y="thời gian (ms)", color="#2f6f45", horizontal=True)
        st.dataframe(t, hide_index=True, width="stretch")
    with tabs[2]:
        if show_debug and "rgb" in res["query"]:
            render_debug_plots(res)
        else:
            st.caption("Bật “Vẽ biểu đồ giá trị trung gian” ở thanh bên để xem biểu đồ HSV, lưới hình dạng, LBP…")
        for name, feature in res["query"]["features"].items():
            if "blocks" not in feature:
                continue
            with st.expander(f"Giá trị khối đặc trưng – {SET_LABELS.get(name, name)}"):
                # các khối có số chiều khác nhau -> dùng Series để pandas tự căn hàng (NaN ở ô thiếu)
                st.dataframe(pd.DataFrame({n: pd.Series([round(float(x), 4) for x in v])
                                           for n, v in feature["blocks"].items()}))
    with tabs[3]:
        if res["backend"] == "linear":
            st.caption("Quét toàn bộ không dùng chỉ mục – mọi ảnh trong CSDL đều được tính khoảng cách.")
        else:
            st.json(res["index_info"], expanded=False)
    with tabs[4]:
        st.code(_mod("search").explain(res))


def render_debug_plots(res: dict) -> None:
    from treedb import visualize

    tmp_dir = Path(_mod("config").RESULTS_DIR) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    st.image(str(visualize.plot_query_features(res, tmp_dir / f"query_features_{stamp}.png")), width="stretch")
    if res["results"]:
        st.image(str(visualize.plot_query_result(res, tmp_dir / f"query_result_{stamp}.png")), width="stretch")


def run_query(searcher, options: dict, path: str, exclude_id: int | None) -> dict | None:
    """Chạy một truy vấn; trả về kết quả hoặc None nếu lỗi."""
    try:
        with st.spinner("Đang tách vùng cây, trích rút đặc trưng và so khớp…"):
            return searcher.query(path, top_k=options["top_k"], backend=options["backend"],
                                  weights=options["weights"], exclude_ids={exclude_id} if exclude_id else None,
                                  filters=options["filters"] or None)
    except LOAD_ERRORS + (ValueError,) as exc:
        st.error(f"Không truy vấn được ảnh này: {type(exc).__name__}: {exc}")
        return None


def remember(result: dict, options: dict, exclude: set[int] | None) -> None:
    st.session_state.update(last_result=result, last_options=options, feedback_history=[], last_exclude=exclude)


def main() -> None:
    st.set_page_config(page_title="TreeDB – tìm kiếm ảnh cây", page_icon="🌳", layout="wide")
    inject_style()
    try:
        searcher, conn = load_backend()
    except LOAD_ERRORS as exc:
        stop_with_load_error(exc)
    options = render_sidebar(searcher)
    render_hero(conn, searcher)

    left, right = st.columns([1, 2], gap="large")
    with left, st.container(border=True):
        section("1", "Ảnh truy vấn")
        query_path, exclude_id = pick_query_source()
        if st.button("🔍 Tìm cây giống nhất", type="primary", disabled=query_path is None, width="stretch"):
            if (result := run_query(searcher, options, query_path, exclude_id)) is not None:
                remember(result, options, {exclude_id} if exclude_id else None)
    auto = st.query_params.get("auto")  # ?auto=query_01.jpg -> tự chạy một ảnh trong data/queries (phục vụ demo)
    if auto and "last_result" not in st.session_state:
        auto_path = Path(_mod("config").QUERY_DIR) / str(auto)
        if auto_path.exists() and (result := run_query(searcher, options, str(auto_path), None)) is not None:
            remember(result, options, None)
    res = st.session_state.get("last_result")
    with right, st.container(border=True):
        section("2", "Phân tích ảnh truy vấn")
        if res is None:
            st.html('<div class="empty"><div class="big">🌿</div>Chọn hoặc tải ảnh cây ở bên trái rồi bấm '
                    '<b>Tìm cây giống nhất</b>.</div>')
            return
        render_query(res)
        if res.get("filters"):
            st.caption("Đang lọc: " + "; ".join(f"{FILTER_LABELS[k]} ∈ {{{', '.join(v)}}}"
                                                for k, v in res["filters"].items()))
    if any((st.session_state.get("last_options") or {}).get(k) != options[k] for k in ("backend", "top_k", "weights", "filters")):
        st.info("Tham số vừa thay đổi – bấm “Tìm cây giống nhất” để chạy lại (kết quả dưới đây của lần trước).")
    section("3", f"{len(res['results'])} cây giống nhất")
    render_feedback(searcher, res, render_results(res))
    section("4", "Chi tiết tính toán")
    render_details(res, options["show_debug"])


main()
