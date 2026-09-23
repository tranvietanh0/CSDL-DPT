"""Demo hệ thống: giao diện Streamlit cho CSDL ảnh cây (CBIR) - chạy: streamlit run app/streamlit_app.py"""
from __future__ import annotations

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

BACKENDS = ("linear", "kdtree", "kmeans")
LOAD_ERRORS = (RuntimeError, ImportError, OSError, sqlite3.Error)  # CSDL trống -> báo lỗi thân thiện


def _mod(name: str):
    """Nạp một mô-đun treedb (import trễ để trang không chết khi gói chưa sẵn sàng)."""
    return importlib.import_module(f"treedb.{name}")


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


# ---- Thanh bên: thông tin CSDL + tham số truy vấn ---------------------------------------
def weight_sliders(names: list[str]) -> dict:
    """Một slider cho mỗi bộ đặc trưng; trả về trọng số đã chuẩn hoá (tổng = 1)."""
    if not names:
        return {}
    defaults = _mod("config").DEFAULT_WEIGHTS  # trọng số mặc định trong cấu hình dự án
    raw = {n: st.sidebar.slider(f"Trọng số {n}", 0.0, 1.0, float(defaults.get(n, 0.0)), 0.05) for n in names}
    total = sum(raw.values())
    if total <= 0.0:
        st.sidebar.warning("Tổng trọng số bằng 0 – tạm dùng trọng số đều nhau cho mọi bộ.")
        return {n: 1.0 / len(raw) for n in raw}
    norm = {n: v / total for n, v in raw.items()}
    st.sidebar.caption("Trọng số sau chuẩn hoá: " + ", ".join(f"{k}={v:.2f}" for k, v in norm.items()))
    return norm  # tổng = 1


def render_sidebar(conn) -> dict:
    """Vẽ thanh bên và trả về cấu hình truy vấn người dùng đã chọn."""
    module = _mod("db")
    st.sidebar.title("Hệ CSDL ảnh cây – tìm kiếm theo nội dung")
    summary = module.summary(conn) or {}
    n_images = summary.get("images")  # None = CSDL trống
    st.sidebar.metric("Số ảnh trong CSDL", "—" if n_images is None else f"{int(n_images):,}")
    sets = module.list_feature_sets(conn)
    st.sidebar.caption("Bộ đặc trưng: " + " · ".join(f"{f['set_name']} {f['dim']} chiều ({f['metric']})" for f in sets))
    st.sidebar.divider()
    backend = st.sidebar.selectbox("Phương pháp chọn ứng viên", BACKENDS)
    top_k = st.sidebar.slider("Số kết quả (top-K)", 1, 10, int(_mod("config").TOP_K))
    weights = weight_sliders([fs["set_name"] for fs in sets])
    debug = st.sidebar.checkbox("Hiển thị giá trị trung gian", value=False)
    return {"backend": backend, "top_k": top_k, "weights": weights, "show_debug": debug}


def save_upload(upload) -> str:
    """Ghi ảnh tải lên vào results/tmp và trả về đường dẫn tạm."""
    tmp_dir = Path(_mod("config").RESULTS_DIR) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"upload_{time.strftime('%Y%m%d_%H%M%S')}_{Path(upload.name).name or 'query.png'}"
    path.write_bytes(upload.getbuffer())
    return str(path)


def pick_query_source() -> tuple[str | None, int | None]:
    """Trả về (đường dẫn ảnh truy vấn, image_id cần loại khỏi kết quả hoặc None)."""
    upload = st.file_uploader("Tải ảnh cây lên (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
    if upload is not None:
        return save_upload(upload), None
    st.caption("Chưa có ảnh tải lên – có thể chọn một ảnh đã có trong CSDL:")
    if not (rows := sample_images()):
        st.info("CSDL chưa có ảnh nào để chọn.")
        return None, None
    pick = st.selectbox("Hoặc chọn một ảnh trong CSDL", range(len(rows)),
                        format_func=lambda i: f"{rows[i]['filename']} — {rows[i]['genus'] or 'chưa rõ chi'}")
    row = rows[pick]
    st.info("Ảnh này vốn nằm trong CSDL: nếu không loại trừ, nó sẽ đứng hạng 1 với độ tương đồng 1.0.")
    exclude = st.checkbox("Loại ảnh truy vấn khỏi kết quả", value=True)
    path = Path(str(row["path"]))
    if not path.exists():  # đường dẫn lưu trong CSDL không còn đúng -> ghép lại từ IMG_DIR
        path = Path(_mod("config").IMG_DIR) / str(row["filename"])
    return str(path), (int(row["image_id"]) if exclude else None)


def render_query(res: dict) -> None:
    """Ảnh truy vấn + mask, thông tin truy vấn và bảng thời gian."""
    q, cols = res["query"], st.columns(2)
    cols[0].image(q["rgb"], caption="Ảnh truy vấn (đã chuẩn hoá)", width="stretch")
    cols[1].image(_mod("segment").overlay(q["rgb"], q["mask"]), caption=f"Vùng cây (mask) – chiếm {q['mask_coverage']:.1%}", width="stretch")
    st.caption(f"Backend: {res['backend']} · ứng viên {res['n_candidates']}/{res['n_database']} · trọng số "
               + ", ".join(f"{s}={w:.2f}" for s, w in res["weights"].items()))
    st.dataframe(pd.DataFrame([{"bước": k, "thời gian (ms)": round(v, 2)} for k, v in res["timing_ms"].items()]), hide_index=True)


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


def render_results(res: dict) -> None:
    """Lưới ảnh kết quả + bảng số liệu chi tiết."""
    if not res["results"]:
        st.warning("Không có kết quả nào (CSDL quá nhỏ hoặc đã loại hết ứng viên).")
        return
    for col, r in zip(st.columns(len(res["results"])), res["results"]):
        caption = f"#{r['rank']} · sim={r['similarity']:.3f} · {r['filename']} · {r['genus'] or 'chưa rõ chi'}"
        if Path(str(r["path"])).exists():
            col.image(str(r["path"]), caption=caption, width="stretch")
        else:
            col.warning(f"Không tìm thấy tệp ảnh: {r['path']}")
    st.dataframe(results_table(res), hide_index=True)


def render_debug(res: dict) -> None:
    """Hình minh hoạ, giá trị khối đặc trưng và thông tin chỉ mục (giá trị trung gian)."""
    from treedb import visualize

    tmp_dir = Path(_mod("config").RESULTS_DIR) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    st.image(str(visualize.plot_query_features(res, tmp_dir / f"query_features_{stamp}.png")), width="stretch")
    if res["results"]:
        st.image(str(visualize.plot_query_result(res, tmp_dir / f"query_result_{stamp}.png")), width="stretch")
    for name, feature in res["query"]["features"].items():
        with st.expander(f"Giá trị khối đặc trưng – {name}"):
            # các khối có số chiều khác nhau -> dùng Series để pandas tự căn hàng (NaN ở ô thiếu)
            st.dataframe(pd.DataFrame({n: pd.Series([round(float(x), 4) for x in v])
                                       for n, v in feature["blocks"].items()}))
    if res["backend"] in ("kdtree", "kmeans"):
        with st.expander(f"Thông tin chỉ mục ({res['backend']})"):
            st.json(res["index_info"])


def run_query(searcher, options: dict, path: str, exclude_id: int | None) -> dict | None:
    """Chạy một truy vấn; trả về kết quả hoặc None nếu lỗi."""
    try:
        return searcher.query(path, top_k=options["top_k"], backend=options["backend"], weights=options["weights"],
                              exclude_ids={exclude_id} if exclude_id else None)
    except LOAD_ERRORS + (ValueError,) as exc:
        st.error(f"Không truy vấn được ảnh này: {type(exc).__name__}: {exc}")
        return None


def main() -> None:
    st.set_page_config(page_title="Demo hệ thống tìm kiếm ảnh cây", layout="wide")
    st.title("Demo hệ thống: tìm kiếm ảnh cây theo nội dung")
    st.caption("Ảnh truy vấn được chuẩn hoá, phân đoạn vật thể, trích rút đặc trưng rồi so khớp có trọng số.")
    try:
        searcher, conn = load_backend()
    except LOAD_ERRORS as exc:
        stop_with_load_error(exc)
    options = render_sidebar(conn)
    st.subheader("1. Ảnh truy vấn")
    query_path, exclude_id = pick_query_source()
    if st.button("Tìm kiếm", type="primary", disabled=query_path is None):
        if (result := run_query(searcher, options, query_path, exclude_id)) is not None:
            st.session_state.update(last_result=result, last_options=options)
    auto = st.query_params.get("auto")  # ?auto=query_01.jpg -> tự chạy một ảnh trong data/queries (phục vụ demo)
    if auto and "last_result" not in st.session_state:
        auto_path = Path(_mod("config").QUERY_DIR) / str(auto)
        if auto_path.exists() and (result := run_query(searcher, options, str(auto_path), None)) is not None:
            st.session_state.update(last_result=result, last_options=options)
    res = st.session_state.get("last_result")
    if res is None:
        st.caption("Chọn ảnh truy vấn rồi bấm “Tìm kiếm”.")
        return
    if any((st.session_state.get("last_options") or {}).get(k) != options[k] for k in ("backend", "top_k", "weights")):
        st.info("Tham số vừa thay đổi – bấm “Tìm kiếm” để chạy lại (kết quả dưới đây của lần trước).")
    st.subheader("2. Kết quả truy vấn")
    render_query(res)
    render_results(res)
    if options["show_debug"]:
        render_debug(res)
    with st.expander("Bảng chi tiết (text)"):
        st.code(_mod("search").explain(res))


main()
