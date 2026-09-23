"""Ve hinh minh hoa cho bao cao: mau du lieu, phan doan, so do khoi, dac trung, ket qua truy van, danh gia."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

from . import config, db  # noqa: E402
from .features import load_rgb, split_blocks  # noqa: E402
from .segment import foreground_mask, overlay  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 9


def _save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------------------
# 1. Bo du lieu
# ---------------------------------------------------------------------------------------
def plot_dataset_samples(out, n: int = 24, seed: int = 1):
    rows = list(csv.DictReader(open(config.MANIFEST, encoding="utf-8")))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    cols = 6
    r = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(r, cols, figsize=(cols * 2.2, r * 2.4))
    for ax, i in zip(axes.flat, pick):
        row = rows[i]
        ax.imshow(Image.open(config.IMG_DIR / row["filename"]))
        ax.set_title(f"{row['filename']}\n{row['genus'] or '(chưa rõ chi)'}", fontsize=7)
        ax.axis("off")
    for ax in axes.flat[len(pick):]:
        ax.axis("off")
    fig.suptitle(f"Mẫu ảnh trong bộ dữ liệu ({config.IMAGE_SIZE}×{config.IMAGE_SIZE}, cây nằm giữa ảnh)")
    return _save(fig, out)


def plot_dataset_stats(conn, out):
    ids, C = db.get_feature_matrix(conn, "color")
    _, S = db.get_feature_matrix(conn, "shape")
    imgs = db.get_images(conn, ids.tolist())
    lay_c = {n: d for n, d in db.list_feature_sets(conn)[0]["layout"]} if False else None  # noqa: F841
    # color: moments = 9 chieu cuoi (mean H,S,V ...)
    mom = C[:, -9:]
    hue, sat, val = mom[:, 0] * 360, mom[:, 1], mom[:, 2]
    cov = np.array([imgs[int(i)].get("fg_coverage") or 0 for i in ids])
    aspect = S[:, 1]
    solidity = S[:, 3]
    groups = [imgs[int(i)].get("group_label") or "unknown" for i in ids]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    axes[0, 0].hist(hue, bins=36, color="#4c9a2a")
    axes[0, 0].set_title("Sắc độ (Hue) trung bình vùng cây (°)")
    axes[0, 1].hist(sat, bins=30, color="#e0a800")
    axes[0, 1].set_title("Độ bão hoà (S) trung bình vùng cây")
    axes[0, 2].hist(val, bins=30, color="#555")
    axes[0, 2].set_title("Độ sáng (V) trung bình vùng cây")
    axes[1, 0].hist(cov, bins=30, color="#2a7fbf")
    axes[1, 0].set_title("Tỉ lệ diện tích vùng cây / ảnh")
    axes[1, 1].hist(np.clip(aspect, 0, 3), bins=30, color="#b04a9a")
    axes[1, 1].set_title("Tỉ lệ cao/rộng của hình chữ nhật cơ bản")
    cnt = {g: groups.count(g) for g in sorted(set(groups))}
    axes[1, 2].bar(list(cnt), list(cnt.values()), color=["#4c9a2a", "#1f5f3f", "#999"][:len(cnt)])
    axes[1, 2].set_title("Số ảnh theo nhóm hình thái")
    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle("Thống kê điểm giống / khác nhau giữa các ảnh cây trong bộ dữ liệu")
    fig.tight_layout()
    return _save(fig, out)


def plot_segmentation_examples(out, filenames: list[str]):
    fig, axes = plt.subplots(2, len(filenames), figsize=(2.6 * len(filenames), 5.4))
    for j, fn in enumerate(filenames):
        rgb = load_rgb(config.IMG_DIR / fn)
        m = foreground_mask(rgb)
        axes[0, j].imshow(rgb)
        axes[0, j].set_title(fn, fontsize=8)
        axes[1, j].imshow(overlay(rgb, m))
        axes[1, j].set_title(f"vùng cây = {m.mean():.1%}", fontsize=8)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.suptitle("Phân đoạn vật thể (GrabCut + loại bầu trời): ảnh gốc (trên) và vùng cây (dưới, đỏ)")
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------------------------------
# 2. So do khoi he thong
# ---------------------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc="#e8f1fb", ec="#2a5d9f", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def _arrow(ax, x1, y1, x2, y2, text="", fs=7, dy=0.075):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#333"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center", va="bottom", fontsize=fs, color="#333",
                bbox=dict(fc="white", ec="none", pad=1))


def plot_block_diagram(out):
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # --- giai doan ngoai tuyen ---
    ax.text(0.02, 0.965, "GIAI ĐOẠN NGOẠI TUYẾN (xây dựng CSDL)", fontsize=10, weight="bold", color="#1f4e79")
    ax.add_patch(FancyBboxPatch((0.01, 0.55), 0.98, 0.40, boxstyle="round,pad=0.01", fc="none", ec="#1f4e79",
                                ls="--", lw=1))
    y = 0.70
    _box(ax, 0.03, y, 0.15, 0.13, "1. Thu thập ảnh\n(Wikimedia Commons,\nlọc theo quy tắc)")
    _box(ax, 0.22, y, 0.15, 0.13, "2. Tiền xử lý\n(EXIF, RGB, resize,\ncenter-crop 512×512,\nkhử trùng lặp)")
    _box(ax, 0.41, y, 0.15, 0.13, "3. Phân đoạn\nvật thể / nền\n(GrabCut, loại trời)")
    _box(ax, 0.60, y, 0.17, 0.13, "4. Trích rút đặc trưng\nMàu (369) · Hình dạng (113)\nKết cấu (85) · Embedding (1280)")
    _box(ax, 0.81, y, 0.16, 0.13, "5. CSDL SQLite\nimages · feature_sets\nfeatures · feature_stats", fc="#fff2cc",
         ec="#b8860b")
    _arrow(ax, 0.18, y + 0.065, 0.22, y + 0.065, "ảnh thô + metadata")
    _arrow(ax, 0.37, y + 0.065, 0.41, y + 0.065, "ảnh chuẩn")
    _arrow(ax, 0.56, y + 0.065, 0.60, y + 0.065, "ảnh + mask")
    _arrow(ax, 0.77, y + 0.065, 0.81, y + 0.065, "vector đặc trưng")
    _box(ax, 0.60, 0.575, 0.17, 0.10, "6. Chuẩn hoá & chỉ mục\nz-score, scale; PCA + kd-tree;\nphân cụm K-means", fc="#e2f0d9",
         ec="#3a7d2c")
    ax.plot([0.89, 0.89, 0.79], [y, 0.61, 0.61], color="#333", lw=1.1)
    _arrow(ax, 0.79, 0.61, 0.77, 0.61, "", dy=0)
    ax.text(0.84, 0.62, "ma trận đặc trưng", ha="center", fontsize=7, color="#333")
    ax.text(0.685, 0.56, "→ bảng indexes, image_clusters", ha="center", fontsize=7, color="#333")

    # --- giai doan truc tuyen ---
    ax.text(0.02, 0.475, "GIAI ĐOẠN TRỰC TUYẾN (truy vấn)", fontsize=10, weight="bold", color="#7b2d00")
    ax.add_patch(FancyBboxPatch((0.01, 0.03), 0.98, 0.45, boxstyle="round,pad=0.01", fc="none", ec="#7b2d00",
                                ls="--", lw=1))
    y2 = 0.27
    _box(ax, 0.03, y2, 0.12, 0.13, "Ảnh truy vấn\n(cây mới)", fc="#fde9d9", ec="#c0504d")
    _box(ax, 0.18, y2, 0.14, 0.13, "Q1. Tiền xử lý +\nphân đoạn\n(giống bước 2-3)")
    _box(ax, 0.35, y2, 0.15, 0.13, "Q2. Trích rút đặc trưng\n(giống bước 4)")
    _box(ax, 0.53, y2, 0.16, 0.13, "Q3. Chọn ứng viên\nlinear / kd-tree / K-means\n(cắt tỉa)")
    _box(ax, 0.72, y2, 0.14, 0.13, "Q4. Tính khoảng cách\nL1 · L2 · cosine\nchuẩn hoá, trọng số\nD = Σ wₛ·dₛ/scaleₛ")
    _box(ax, 0.88, y2, 0.10, 0.13, "Q5. Xếp hạng\nsim = 1/(1+D)\nTop-5", fc="#e2f0d9", ec="#3a7d2c")
    _arrow(ax, 0.15, y2 + 0.065, 0.18, y2 + 0.065)
    _arrow(ax, 0.32, y2 + 0.065, 0.35, y2 + 0.065, "ảnh + mask")
    _arrow(ax, 0.50, y2 + 0.065, 0.53, y2 + 0.065, "vector q")
    _arrow(ax, 0.69, y2 + 0.065, 0.72, y2 + 0.065, "tập ứng viên")
    _arrow(ax, 0.86, y2 + 0.065, 0.88, y2 + 0.065, "D", dy=0.075)
    _box(ax, 0.35, 0.06, 0.34, 0.10, "CSDL đặc trưng + chỉ mục (đọc)", fc="#fff2cc", ec="#b8860b")
    _arrow(ax, 0.61, 0.16, 0.61, y2)
    ax.plot([0.66, 0.66, 0.79], [0.16, 0.22, 0.22], color="#333", lw=1.1)
    _arrow(ax, 0.79, 0.22, 0.79, y2)
    _box(ax, 0.75, 0.06, 0.23, 0.10, "Kết quả: 5 ảnh + độ tương đồng +\ngiá trị trung gian (từng bộ đặc trưng)\nghi query_log",
         fc="#fde9d9", ec="#c0504d")
    _arrow(ax, 0.93, y2, 0.93, 0.16)
    fig.suptitle("Sơ đồ khối hệ thống lưu trữ và tìm kiếm ảnh cây theo nội dung (CBIR)", fontsize=12)
    return _save(fig, out)


# ---------------------------------------------------------------------------------------
# 3. Dac trung cua mot anh (ket qua trung gian)
# ---------------------------------------------------------------------------------------
def plot_query_features(res: dict, out):
    q = res["query"]
    rgb, mask = q["rgb"], q["mask"]
    F = q["features"]
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb)
    ax.set_title("Ảnh truy vấn (đã chuẩn hoá)")
    ax.axis("off")
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(overlay(rgb, mask))
    ax.set_title(f"Vùng cây ({mask.mean():.1%})")
    ax.axis("off")
    if "color" in F:
        b = F["color"]["blocks"]
        ax = fig.add_subplot(gs[0, 2:])
        ax.bar(range(72), b["hsv_global"], color="#4c9a2a", width=1.0)
        ax.set_title("Màu: biểu đồ HSV toàn cục 8×3×3 = 72 ô (chuẩn hoá L1)")
        ax.set_xlabel("ô màu j (H·9 + S·3 + V)")
        ax = fig.add_subplot(gs[1, 0])
        ax.bar(range(72), b["hsv_fg"], color="#2a7f2a", width=1.0, label="vật thể")
        ax.bar(range(72), -b["hsv_bg"], color="#5b9bd5", width=1.0, label="nền")
        ax.set_title("Màu: biểu đồ vật thể (+) / nền (−)")
        ax.legend(fontsize=7)
    if "shape" in F:
        b = F["shape"]["blocks"]
        ax = fig.add_subplot(gs[1, 1])
        ax.imshow(b["grid"].reshape(8, 8), cmap="Greens", vmin=0, vmax=1)
        ax.set_title("Hình dạng: lưới 8×8 (>15% → 1)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(b["profile"], range(16), marker="o", color="#1f5f3f")
        ax.invert_yaxis()
        ax.set_title("Hình dạng: profile bề rộng (đỉnh→gốc)")
        ax.set_xlabel("tỉ lệ bề rộng")
        ax.grid(alpha=0.3)
        ax = fig.add_subplot(gs[1, 3])
        names = ["area", "h/w", "extent", "solid.", "ecc.", "major", "minor", "orient", "dx", "dy"]
        ax.bar(names, b["region"], color="#b04a9a")
        ax.set_title("Hình dạng: đo vùng + tâm sai")
        ax.tick_params(axis="x", labelrotation=60, labelsize=7)
    if "texture" in F:
        b = F["texture"]["blocks"]
        ax = fig.add_subplot(gs[2, 0])
        ax.bar(range(28), b["lbp"], color="#555", width=1.0)
        ax.set_title("LBP (10 + 18 ô)")
        ax = fig.add_subplot(gs[2, 1])
        ax.bar(range(8), b["edge"][:8], color="#c0504d")
        ax.set_title(f"Hướng cạnh (8 ô), mật độ={b['edge'][8]:.2f}")
        ax = fig.add_subplot(gs[2, 2])
        d = np.zeros(36)
        d[1:] = b["dct"]
        ax.imshow(d.reshape(6, 6), cmap="coolwarm")
        ax.set_title("DCT 6×6 tần số thấp (bỏ DC)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[2, 3])
        names = ["contrast", "dissim.", "homog.", "energy", "corr."]
        g = b["glcm"].reshape(5, 2)
        ax.bar(np.arange(5) - 0.2, g[:, 0], width=0.4, label="d=1")
        ax.bar(np.arange(5) + 0.2, g[:, 1], width=0.4, label="d=3")
        ax.set_xticks(range(5))
        ax.set_xticklabels(names, fontsize=7)
        tam = [round(float(x), 2) for x in b["tamura"]]
        ax.set_title(f"GLCM; Tamura={tam}", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("Giá trị đặc trưng trung gian của ảnh truy vấn", fontsize=12)
    return _save(fig, out)


def plot_query_result(res: dict, out):
    q = res["query"]
    rows = res["results"]
    sets = [s for s in res["sets"] if s in rows[0]["per_set"]]
    n = len(rows)
    fig = plt.figure(figsize=(2.5 * (n + 1), 6.2))
    gs = fig.add_gridspec(2, n + 1, height_ratios=[1.35, 1], hspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    if "rgb" in q:
        ax.imshow(q["rgb"])
    else:
        ax.imshow(Image.open(db.get_image(db.connect(), q["image_id"])["path"]))
    ax.set_title("Ảnh truy vấn", color="#c0504d", weight="bold")
    ax.axis("off")
    for j, r in enumerate(rows, start=1):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(Image.open(r["path"]))
        ax.set_title(f"#{r['rank']}  sim={r['similarity']:.3f}\n{r['filename']}  {r['genus'] or ''}", fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(n)
    bottom = np.zeros(n)
    colors = {"color": "#4c9a2a", "shape": "#b04a9a", "texture": "#e0a800", "deep": "#2a7fbf"}
    for s in sets:
        c = np.array([r["per_set"][s]["contrib"] for r in rows])
        ax.bar(x, c, bottom=bottom, color=colors.get(s, "#999"), label=f"{s} (w={res['weights'][s]:.2f})")
        bottom += c
    for i, r in enumerate(rows):
        ax.text(i, bottom[i] + 0.01, f"D={r['distance']:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{r['rank']}" for r in rows])
    ax.set_ylabel("khoảng cách kết hợp D = Σ wₛ·dₛ/scaleₛ")
    ax.set_title(f"Đóng góp của từng bộ đặc trưng vào khoảng cách (backend={res['backend']}, "
                 f"ứng viên={res['n_candidates']}/{res['n_database']})", fontsize=9)
    ax.legend(fontsize=8, ncol=len(sets))
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


# ---------------------------------------------------------------------------------------
# 4. Danh gia va phan cum
# ---------------------------------------------------------------------------------------
def plot_evaluation(report: dict, out):
    fs = report["feature_sets"]
    names = list(fs)
    k = report["k"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(names))
    ax = axes[0]
    ax.bar(x - 0.3, [fs[n]["p_at_k"] for n in names], width=0.2, label=f"P@{k} (cùng chi)", color="#2a7fbf")
    ax.bar(x - 0.1, [fs[n]["p_at_1"] for n in names], width=0.2, label="P@1", color="#4c9a2a")
    ax.bar(x + 0.1, [fs[n]["ap_at_k"] for n in names], width=0.2, label=f"mAP@{k}", color="#e0a800")
    ax.bar(x + 0.3, [fs[n]["group_p_at_k"] for n in names], width=0.2, label=f"P@{k} (cùng nhóm)", color="#b04a9a")
    ax.axhline(report["random_baseline"]["p_at_k"], color="#2a7fbf", ls="--", lw=1, label="ngẫu nhiên (chi)")
    ax.axhline(report["random_baseline"]["group_p_at_k"], color="#b04a9a", ls=":", lw=1, label="ngẫu nhiên (nhóm)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.set_title(f"Độ chính xác truy vấn theo bộ đặc trưng ({report['n_queries']} truy vấn có nhãn chi)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    be = report["backends"]
    bn = list(be)
    ax.bar(np.arange(len(bn)) - 0.2, [be[b]["recall_vs_linear"] for b in bn], width=0.4, color="#2a7fbf",
           label=f"recall@{k} so với linear")
    ax.bar(np.arange(len(bn)) + 0.2, [be[b]["p_at_k"] for b in bn], width=0.4, color="#4c9a2a", label=f"P@{k}")
    for i, b in enumerate(bn):
        ax.text(i, 1.02, f"{be[b]['ms_per_query']:.2f} ms", ha="center", fontsize=8)
    ax.set_xticks(range(len(bn)))
    ax.set_xticklabels(bn)
    ax.set_ylim(0, 1.15)
    ax.set_title("So sánh backend chọn ứng viên")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out)


def plot_clusters(conn, out):
    from .index import FeatureSpace, KDTreeIndex, KMeansIndex
    space = FeatureSpace(conn)
    kd = KDTreeIndex(conn)
    km = KMeansIndex(conn)
    Z = kd.Z[:, :2]
    imgs = db.get_images(conn, space.ids.tolist())
    groups = np.array([imgs[int(i)].get("group_label") or "unknown" for i in space.ids])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc = axes[0].scatter(Z[:, 0], Z[:, 1], c=km.labels, cmap="tab10", s=12)
    axes[0].set_title(f"Phân cụm K-means (K={km.meta['k']}) trên không gian PCA (2 thành phần đầu)")
    axes[0].legend(*sc.legend_elements(), title="cụm", fontsize=7, loc="best")
    for g, c in (("broadleaf", "#4c9a2a"), ("conifer", "#1f5f3f"), ("unknown", "#bbb")):
        m = groups == g
        axes[1].scatter(Z[m, 0], Z[m, 1], s=12, c=c, label=f"{g} ({m.sum()})")
    axes[1].set_title("Cùng không gian, tô màu theo nhóm hình thái")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out)


def plot_feature_dims(conn, out):
    fs = db.list_feature_sets(conn)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colors = {"color": "#4c9a2a", "shape": "#b04a9a", "texture": "#e0a800", "deep": "#2a7fbf"}
    y = 0
    for s in fs:
        left = 0
        for name, d in s["layout"]:
            ax.barh(y, d, left=left, color=colors.get(s["set_name"], "#999"), edgecolor="white")
            if d >= 20:
                ax.text(left + d / 2, y, f"{name} ({d})", ha="center", va="center", fontsize=7, color="white")
            left += d
        ax.text(left + 5, y, f"{s['set_name']}: {s['dim']} chiều, {s['metric']}", va="center", fontsize=8)
        y += 1
    ax.set_yticks([])
    ax.set_xscale("symlog")
    ax.set_xlabel("số chiều (thang symlog)")
    ax.set_title("Cấu trúc các bộ đặc trưng lưu trong CSDL")
    fig.tight_layout()
    return _save(fig, out)


def load_report(path=None) -> dict:
    return json.loads(Path(path or config.RESULTS_DIR / "evaluation.json").read_text(encoding="utf-8"))
