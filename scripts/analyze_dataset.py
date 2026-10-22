"""Danh gia chat luong va tinh can bang cua bo du lieu (muc 1.3.2, 1.5, 2.2, 2.3 cua Bao cao 1).

    python scripts/analyze_dataset.py   -> results/dataset_quality.json
                                         + docs/figures/17_genus_representatives.png
                                         + docs/figures/18_dataset_limitations.png
                                         + docs/figures/19_genus_distribution.png

Tinh lai mat na vung cay cua tung anh trong CSDL (SegFormer, can onnxruntime) de do:
  - tan cham mep anh (ti le chieu dai canh anh bi mat na cham vao) -> tan bi khung hinh cat,
  - chieu cao / chieu rong khung bao cua cay so voi anh -> ti le vat trong khung,
  - anh goc doc / ngang, anh den trang (do bao hoa trung binh rat thap),
  - can bang nhan chi (IR), ti le anh co anh gan nhat (embedding sau) cung tac gia.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treedb import config, db, segformer  # noqa: E402
from treedb.analysis import _block_slices  # noqa: E402
from treedb.features import load_rgb  # noqa: E402
from treedb.segment import foreground_mask, mask_bbox  # noqa: E402

OUT = config.RESULTS_DIR / "dataset_quality.json"
FIG_REPR = config.FIG_DIR / "17_genus_representatives.png"
FIG_LIMIT = config.FIG_DIR / "18_dataset_limitations.png"
FIG_DIST = config.FIG_DIR / "19_genus_distribution.png"

EDGE_BAND = 3            # so hang / cot sat mep anh duoc xet
TOP_CUT = 0.05           # mat na cham > 5% chieu dai mep tren -> tan bi cat o dinh
SMALL_TREE = 0.25        # cay thap hon 1/4 chieu cao anh
GRAY_SAT = 0.04          # do bao hoa trung binh < 4% -> anh den trang / gan don sac
MIN_GENUS_BALANCE = 5    # chi co >= 5 anh moi tinh vao he so mat can bang
# Nhom hien thi cho bang doi sanh theo chi: Acacia va Vachellia (keo savan) gop lam mot
DISPLAY_GENERA = ["Populus", "Quercus", "Pinus", "Juglans", "Betula", "Alnus", "Tilia", "Acacia/Vachellia"]
GENUS_VI = {"Populus": "dương", "Quercus": "sồi", "Pinus": "thông", "Juglans": "óc chó", "Betula": "bạch dương",
            "Alnus": "tống quán sủi", "Tilia": "đoan", "Acacia/Vachellia": "keo"}
CONIFER = {"Pinus", "Picea", "Abies", "Larix", "Cedrus", "Cupressus", "Taxus", "Juniperus", "Sequoia",
           "Sequoiadendron", "Araucaria", "Thuja", "Metasequoia", "Taxodium", "Pseudotsuga", "Tsuga", "Cryptomeria"}

# Mau theo nhom hinh thai (da kiem tra do tach mau cho nguoi mu mau); chu dung mau muc, khong dung mau du lieu
C_BROAD, C_CONIF, C_UNK = "#2a78d6", "#eb6834", "#a3a29c"
INK, INK_2 = "#0b0b0b", "#52514e"


def display_genus(g: str) -> str:
    return "Acacia/Vachellia" if g in ("Acacia", "Vachellia") else g


def measure(rgb: np.ndarray) -> dict:
    mask, method = foreground_mask(rgb, return_method=True)
    H, W = mask.shape
    t, l, b, r = mask_bbox(mask)
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].mean() / 255.0
    return {"method": method,
            "top": float(mask[:EDGE_BAND].any(0).mean()), "bottom": float(mask[-EDGE_BAND:].any(0).mean()),
            "left": float(mask[:, :EDGE_BAND].any(1).mean()), "right": float(mask[:, -EDGE_BAND:].any(1).mean()),
            "height": (b - t) / H, "width": (r - l) / W, "coverage": float(mask.mean()), "saturation": float(sat)}, mask


def deep_neighbors(conn) -> tuple[np.ndarray, np.ndarray]:
    ids, X = db.get_feature_matrix(conn, "deep")
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -1)
    return ids, S


def main():
    if not segformer.is_available():
        sys.exit("Can SegFormer (pip install onnxruntime) de tinh mat na vung cay")
    conn = db.connect()
    imgs = db.list_images(conn)
    per_image, masks = {}, {}
    for k, im in enumerate(imgs):
        m, mask = measure(load_rgb(config.IMG_DIR / im["filename"]))
        m["portrait"] = int(im["orig_height"]) > int(im["orig_width"])
        per_image[im["filename"]] = m
        masks[im["filename"]] = mask
        if k % 100 == 0:
            print(f"[mask] {k}/{len(imgs)}", flush=True)

    names = [im["filename"] for im in imgs]
    P = per_image
    top_cut = [n for n in names if P[n]["top"] > TOP_CUT]
    portrait = [n for n in names if P[n]["portrait"]]
    small = [n for n in names if P[n]["height"] < SMALL_TREE]
    gray = [n for n in names if P[n]["saturation"] < GRAY_SAT]
    hgt = np.array([P[n]["height"] for n in names])
    cov = np.array([P[n]["coverage"] for n in names])

    # Ti le khung bao cao/rong cua cay (khoi region cua bo hinh dang)
    sets = {f["set_name"]: f for f in db.list_feature_sets(conn)}
    sids, S = db.get_feature_matrix(conn, "shape")
    reg = S[:, _block_slices(sets["shape"]["layout"])["region"]]
    by_id = {im["image_id"]: im for im in imgs}
    aspect = reg[:, 1]

    # Can bang nhan
    genus = Counter(im.get("genus") or "" for im in imgs)
    labeled = {g: c for g, c in genus.items() if g}
    big = {g: c for g, c in labeled.items() if c >= MIN_GENUS_BALANCE}
    groups = Counter(im.get("group_label") or "unknown" for im in imgs)

    # Anh gan nhat theo embedding sau cung tac gia (loat anh cung mot cay / cung nguoi chup)
    dids, Sim = deep_neighbors(conn)
    author = np.array([by_id[int(i)].get("author") or "" for i in dids])
    nn = Sim.argmax(1)
    same_author_nn = float((author[nn] == author).mean())

    # Thong ke hinh dang theo nhom chi hien thi (gom Acacia + Vachellia)
    gstats = {}
    dg = np.array([display_genus(by_id[int(i)].get("genus") or "") for i in sids])
    fname = np.array([by_id[int(i)]["filename"] for i in sids])
    for g in DISPLAY_GENERA:
        sel = dg == g
        gstats[g] = {"n": int(sel.sum()), "aspect": float(aspect[sel].mean()), "solidity": float(reg[sel, 3].mean()),
                     "eccentricity": float(reg[sel, 4].mean()), "area": float(reg[sel, 0].mean()),
                     "height": float(np.mean([P[f]["height"] for f in fname[sel]]))}

    rep = {
        "n": len(names),
        "thresholds": {"top_cut": TOP_CUT, "small_tree": SMALL_TREE, "gray_saturation": GRAY_SAT},
        "segmentation": dict(Counter(P[n]["method"] for n in names)),
        "crown_cut_top": len(top_cut),
        "portrait": len(portrait),
        "crown_cut_top_portrait": sum(n in set(top_cut) for n in portrait),
        "crown_cut_top_landscape": sum(n in set(top_cut) for n in names if not P[n]["portrait"]),
        "small_tree": len(small),
        "below_third": int((hgt < 1 / 3).sum()),
        "grayscale": len(gray),
        "height_pct": {str(q): float(np.percentile(hgt, q)) for q in (0, 5, 25, 50, 75, 95, 100)},
        "coverage_pct": {str(q): float(np.percentile(cov, q)) for q in (0, 5, 50, 95, 100)},
        "aspect_pct": {str(q): float(np.percentile(aspect, q)) for q in (0, 5, 25, 50, 75, 95, 100)},
        "aspect_in_06_16": float(((aspect >= 0.6) & (aspect <= 1.6)).mean()),
        "labeled": sum(labeled.values()), "n_genera": len(labeled),
        "genus_counts": dict(sorted(labeled.items(), key=lambda kv: -kv[1])),
        "balance": {"min_images": MIN_GENUS_BALANCE, "n_genera": len(big), "n_images": sum(big.values()),
                    "max": max(big.values()), "max_genus": max(big, key=big.get),
                    "min": min(big.values()), "min_genus": min(big, key=big.get),
                    "ir": max(big.values()) / min(big.values())},
        "groups": dict(groups),
        "same_author_nn": same_author_nn,
        "genus_stats": gstats,
        "images": P,
    }
    rep["examples"] = plot_limitations(names, P, Sim, dids, author, by_id, FIG_LIMIT)
    rep["representatives"] = plot_representatives(dg, fname, reg, P, masks, gstats, FIG_REPR)
    plot_distribution(labeled, groups, FIG_DIST)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    conn.close()
    print(f"[quality] {OUT}")
    print(f"  tan cham mep tren: {len(top_cut)}/{len(names)} (anh goc doc {rep['crown_cut_top_portrait']}/"
          f"{len(portrait)}, ngang {rep['crown_cut_top_landscape']}/{len(names) - len(portrait)})")
    print(f"  cay < 1/4 anh: {len(small)}; den trang: {len(gray)}; anh gan nhat cung tac gia: {same_author_nn:.1%}")
    print(f"  IR (chi >= {MIN_GENUS_BALANCE} anh): {rep['balance']['ir']:.1f}")
    for p in (FIG_REPR, FIG_LIMIT, FIG_DIST):
        print("[fig]", p)


def _tile(ax, path, title, color=INK):
    ax.imshow(load_rgb(path))
    ax.set_title(title, fontsize=9, color=color)
    ax.axis("off")


def plot_limitations(names, P, Sim, dids, author, by_id, out) -> dict:
    """3 hang vi du: tan bi cat o dinh, cay qua nho trong khung, nhieu anh cua cung mot cay."""
    # anh goc doc (nguyen nhan chinh), tan cham mep tren nhung khong cham kin hai ben (tranh mat na dinh vao rung nen)
    cut = sorted((n for n in names if P[n]["portrait"] and max(P[n]["left"], P[n]["right"]) < 0.5),
                 key=lambda n: -P[n]["top"])[:5]
    small = sorted(names, key=lambda n: P[n]["height"])[:5]
    # anh co 4 lang gieng gan nhat cung tac gia va giong nhat -> mot loat anh cung cay
    order = np.argsort(-Sim, axis=1)[:, :4]
    score = np.where((author[order] == author[:, None]).all(1), np.take_along_axis(Sim, order, 1).mean(1), -1)
    c = int(score.argmax())
    series = [by_id[int(dids[i])]["filename"] for i in [c, *order[c]]]
    rows = [("Tán bị cắt (ảnh gốc dọc)", cut, lambda n: f"{n[5:9]} · chạm mép trên {P[n]['top']:.0%}"),
            ("Cây nhỏ trong khung hình", small, lambda n: f"{n[5:9]} · cao {P[n]['height']:.0%} ảnh"),
            ("Nhiều ảnh của cùng một cây", series, lambda n: f"{n[5:9]}")]
    fig, axes = plt.subplots(3, 5, figsize=(13, 8.4))
    for r, (label, files, cap) in enumerate(rows):
        for k, n in enumerate(files):
            _tile(axes[r, k], config.IMG_DIR / n, cap(n))
        axes[r, 0].text(-0.08, 0.5, label, transform=axes[r, 0].transAxes, rotation=90, ha="right", va="center",
                        fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return {"crown_cut": cut, "small": small, "series": series}


def plot_representatives(dg, fname, reg, P, masks, gstats, out) -> dict:
    """Moi nhom chi 1 anh tieu bieu: trong cac anh cay tron ven, chon anh co dang (cao/rong, solidity, tam sai,
    dien tich) gan trung binh cua chi nhat. Hang tren: anh + vien mat na; hang duoi: hinh bong de so sanh dang tan."""
    feats = reg[:, [1, 3, 4, 0]]
    Z = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    reps = {}
    for g in DISPLAY_GENERA:
        members = np.where(dg == g)[0]
        cand = [k for k in members
                if P[fname[k]]["top"] <= 0.01 and max(P[fname[k]]["left"], P[fname[k]]["right"]) <= 0.10
                and 0.45 <= P[fname[k]]["height"] <= 0.95 and P[fname[k]]["saturation"] >= GRAY_SAT]
        cand = cand or list(members)
        center = Z[members].mean(0)
        reps[g] = str(fname[min(cand, key=lambda k: float(((Z[k] - center) ** 2).sum()))])
    fig, axes = plt.subplots(2, len(DISPLAY_GENERA), figsize=(16, 4.9))
    for k, g in enumerate(DISPLAY_GENERA):
        n = reps[g]
        rgb = load_rgb(config.IMG_DIR / n).copy()
        cnts, _ = cv2.findContours(masks[n].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cnts, -1, (235, 40, 40), 3)
        axes[0, k].imshow(rgb)
        axes[0, k].set_title(f"{g}\n({GENUS_VI[g]}) · {n[5:9]}", fontsize=9, color=INK)
        axes[0, k].axis("off")
        axes[1, k].imshow(~masks[n], cmap="gray", vmin=0, vmax=1)
        st = gstats[g]
        axes[1, k].set_title(f"cao/rộng {st['aspect']:.2f} · đặc {st['solidity']:.2f}".replace(".", ","),
                             fontsize=8.5, color=INK_2)
        axes[1, k].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return reps


def plot_distribution(labeled: dict, groups: Counter, out):
    """So anh theo chi (chi co >= 3 anh) + so anh chua co nhan; mau theo nhom hinh thai."""
    items = sorted(((g, c) for g, c in labeled.items() if c >= 3), key=lambda kv: kv[1])
    other = sum(c for g, c in labeled.items() if c < 3)
    n_other = sum(1 for c in labeled.values() if c < 3)
    rows = [(f"{n_other} chi khác (< 3 ảnh/chi)", other, C_UNK), *[(g, c, C_CONIF if g in CONIFER else C_BROAD)
                                                                 for g, c in items]]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [3, 1.25]})
    y = np.arange(len(rows))
    ax.barh(y, [c for _, c, _ in rows], height=0.62, color=[col for _, _, col in rows], edgecolor="white",
            linewidth=1.5)
    for yi, (_, c, _) in zip(y, rows):
        ax.text(c + 0.8, yi, str(c), va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [g for g, _, _ in rows], fontsize=9)
    ax.set_xlabel("Số ảnh", color=INK_2)
    ax.set_title(f"Số ảnh có nhãn theo chi ({sum(labeled.values())} ảnh, {len(labeled)} chi)", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_BROAD, C_CONIF, C_UNK)],
              labels=["lá rộng", "lá kim", "gộp nhiều chi"], loc="lower right", frameon=False)
    order = [("broadleaf", "lá rộng", C_BROAD), ("conifer", "lá kim", C_CONIF), ("unknown", "chưa có nhãn", C_UNK)]
    vals = [groups.get(k, 0) for k, _, _ in order]
    ax2.bar(range(3), vals, width=0.6, color=[c for _, _, c in order], edgecolor="white", linewidth=1.5)
    for x, v in enumerate(vals):
        ax2.text(x, v + 4, f"{v}\n({v / sum(vals):.0%})", ha="center", va="bottom", fontsize=9, color=INK)
    ax2.set_xticks(range(3), [lab for _, lab, _ in order], fontsize=9)
    ax2.set_ylim(0, max(vals) * 1.25)
    ax2.set_title("Số ảnh theo nhóm hình thái", fontsize=11)
    ax2.grid(axis="y", alpha=0.25)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
