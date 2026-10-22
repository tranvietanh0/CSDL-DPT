"""Phan tich dinh luong noi dung 1-2 (giong / khac giua cac chi, gia tri thong tin cua tung khoi dac trung).

    python scripts/analyze_features.py      -> results/feature_analysis.json + docs/figures/15_*.png, 16_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treedb import analysis, config, db  # noqa: E402

OUT = config.RESULTS_DIR / "feature_analysis.json"
FIG_BOX = config.FIG_DIR / "15_genus_feature_boxplots.png"
FIG_TRIPLET = config.FIG_DIR / "16_similarity_triplet.png"
BOX_FEATURES = ["Tỉ lệ cao/rộng (HCN cơ bản)", "Solidity (độ đặc tán)", "Sắc độ H trung bình vùng cây (°)",
                "Mật độ cạnh"]


def plot_boxplots(conn, rep: dict, out: Path) -> Path:
    sets = {f["set_name"]: f for f in db.list_feature_sets(conn)}
    imgs = {i["image_id"]: i for i in db.list_images(conn)}
    genera = list(rep["genera"])
    fig, axes = plt.subplots(1, len(BOX_FEATURES), figsize=(16, 4.2))
    for ax, label in zip(axes, BOX_FEATURES):
        s, b, k, _, mul = next(f for f in analysis.KEY_FEATURES if f[3] == label)
        start = sum(d for n, d in sets[s]["layout"][:[n for n, _ in sets[s]["layout"]].index(b)])
        ids, X = db.get_feature_matrix(conn, s)
        g = np.array([imgs[int(i)].get("genus") or "" for i in ids])
        data = [X[g == gn, start + k] * mul for gn in genera]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(genera) + 1), genera, rotation=60, fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Phân bố một số đặc trưng theo chi (genus): khác biệt giữa các chi, tương đồng trong cùng chi")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_triplet(conn, rep: dict, out: Path) -> Path:
    ex = rep["example"]
    imgs = db.get_images(conn, ex["image_ids"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, iid, gn, tag in zip(axes, ex["image_ids"], ex["genera"], "ABC"):
        ax.imshow(Image.open(config.IMG_DIR / imgs[iid]["filename"]))   # cot path co the la duong dan may khac
        ax.set_title(f"Ảnh {tag}: {imgs[iid]['filename']} ({gn})", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    conn = db.connect()
    rep = analysis.analyze(conn)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[analysis] {OUT}  ({rep['n_labeled']} anh co nhan, {len(rep['genera'])} chi)")
    for b in sorted(rep["blocks"], key=lambda r: -r["nn1_acc"]):
        print(f"  {b['set']:8s} {b['block']:16s} dim={b['dim']:5d} fisher={b['fisher']:.3f} "
              f"sep={b['sep_ratio']:.3f} 1nn={b['nn1_acc']:.3f} 1nn_xauthor={b['nn1_acc_xauthor']:.3f}")
    print("[fig]", plot_boxplots(conn, rep, FIG_BOX))
    print("[fig]", plot_triplet(conn, rep, FIG_TRIPLET))
    conn.close()


if __name__ == "__main__":
    main()
