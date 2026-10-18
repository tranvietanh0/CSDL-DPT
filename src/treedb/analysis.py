"""Phan tich dinh luong cho noi dung 1-2: diem giong / khac giua cac chi cay va gia tri thong tin cua dac trung.

Chi dung cac anh co nhan chi (genus) trong CSDL. Moi khoi dac trung duoc danh gia bang 3 chi so:
  - fisher    : trung binh theo chieu cua ti so phuong sai giua cac chi / phuong sai trong cung chi (z-score)
  - sep_ratio : khoang cach trung binh giua 2 anh khac chi / khoang cach trung binh giua 2 anh cung chi
  - nn1_acc   : do chinh xac 1-NN leave-one-out theo chi (anh gan nhat co cung chi voi anh dang xet)
Khoang cach dung dung do do cua bo dac trung (L1 cho mau, L2 tren z-score cho hinh dang / ket cau, cosine cho deep).
"""
from __future__ import annotations

import numpy as np

from . import db

# Cac chieu de dien giai (ten bo, ten khoi, vi tri trong khoi, nhan, he so doi don vi)
KEY_FEATURES = [
    ("shape", "region", 0, "Tỉ lệ diện tích vùng cây", 1.0),
    ("shape", "region", 1, "Tỉ lệ cao/rộng (HCN cơ bản)", 1.0),
    ("shape", "region", 3, "Solidity (độ đặc tán)", 1.0),
    ("shape", "region", 4, "Tâm sai", 1.0),
    ("color", "moments", 0, "Sắc độ H trung bình vùng cây (°)", 360.0),
    ("color", "moments", 1, "Độ bão hoà S trung bình", 1.0),
    ("color", "moments", 2, "Độ sáng V trung bình", 1.0),
    ("texture", "edge", 8, "Mật độ cạnh", 1.0),
    ("texture", "glcm", 0, "GLCM contrast (d=1)", 1.0),
]
MIN_GENUS_IMAGES = 9


def _load(conn):
    sets = {f["set_name"]: f for f in db.list_feature_sets(conn)}
    mats = {}
    for s in sets:
        ids, X = db.get_feature_matrix(conn, s)
        mats[s] = (ids, X)
    return sets, mats


def _block_slices(layout):
    out, i = {}, 0
    for name, d in layout:
        out[name] = slice(i, i + d)
        i += d
    return out


def _pairwise(X: np.ndarray, metric: str) -> np.ndarray:
    if metric == "l1":
        return np.abs(X[:, None, :] - X[None, :, :]).sum(-1)
    if metric == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        return 1.0 - Xn @ Xn.T
    sq = (X ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))


def _zscore(X: np.ndarray) -> np.ndarray:
    sd = X.std(0)
    sd[sd < 1e-6] = 1.0
    return (X - X.mean(0)) / sd


def _scores(X: np.ndarray, y: np.ndarray, metric: str) -> dict:
    Z = _zscore(X)
    mu = Z.mean(0)
    between = np.zeros(Z.shape[1])
    within = np.zeros(Z.shape[1])
    for g in np.unique(y):
        Zg = Z[y == g]
        between += len(Zg) * (Zg.mean(0) - mu) ** 2
        within += ((Zg - Zg.mean(0)) ** 2).sum(0)
    ok = within > 1e-9
    fisher = float((between[ok] / within[ok]).mean()) if ok.any() else 0.0
    D = _pairwise(Z if metric == "l2" else X, metric)
    same = y[:, None] == y[None, :]
    off = ~np.eye(len(y), dtype=bool)
    intra, inter = float(D[same & off].mean()), float(D[~same].mean())
    np.fill_diagonal(D, np.inf)
    nn1 = float((y[D.argmin(1)] == y).mean())
    return {"dim": int(X.shape[1]), "fisher": fisher, "intra": intra, "inter": inter,
            "sep_ratio": inter / max(intra, 1e-9), "nn1_acc": nn1}


def analyze(conn) -> dict:
    sets, mats = _load(conn)
    imgs = {i["image_id"]: i for i in db.list_images(conn)}
    ref_ids = mats["color"][0]
    genus = np.array([imgs[int(i)].get("genus") or "" for i in ref_ids])
    counts = {g: int((genus == g).sum()) for g in np.unique(genus) if g}
    keep_genera = sorted(g for g, c in counts.items() if c >= MIN_GENUS_IMAGES)
    lab = np.isin(genus, keep_genera)
    y = genus[lab]
    rng = np.random.default_rng(0)
    chance = float(np.mean([(rng.permutation(y) == y).mean() for _ in range(200)]))

    blocks, set_scores = [], {}
    for s, info in sets.items():
        ids, X = mats[s]
        assert np.array_equal(ids, ref_ids), f"thu tu anh cua bo {s} khong khop"
        XL = X[lab]
        set_scores[s] = _scores(XL, y, info["metric"])
        for name, sl in _block_slices(info["layout"]).items():
            sc = _scores(XL[:, sl], y, info["metric"])
            blocks.append({"set": s, "block": name, "metric": info["metric"], **sc})

    # Thong ke mo ta theo chi cho cac dac trung de dien giai
    per_genus = {}
    for g in keep_genera + ["__all__"]:
        sel = lab & (genus == g) if g != "__all__" else np.ones(len(genus), bool)
        row = {"n": int(sel.sum())}
        for s, b, k, label, mul in KEY_FEATURES:
            sl = _block_slices(sets[s]["layout"])[b]
            v = mats[s][1][sel, sl.start + k] * mul
            row[label] = [float(v.mean()), float(v.std())]
        per_genus[g] = row
    groups = {}
    grp = np.array([imgs[int(i)].get("group_label") or "unknown" for i in ref_ids])
    for g in ("broadleaf", "conifer"):
        sel = grp == g
        groups[g] = {"n": int(sel.sum())}
        for s, b, k, label, mul in KEY_FEATURES:
            sl = _block_slices(sets[s]["layout"])[b]
            v = mats[s][1][sel, sl.start + k] * mul
            groups[g][label] = [float(v.mean()), float(v.std())]

    return {"n_labeled": int(lab.sum()), "genera": {g: counts[g] for g in keep_genera},
            "nn1_chance": chance, "blocks": blocks, "sets": set_scores,
            "per_genus": per_genus, "groups": groups,
            "key_features": [f[3] for f in KEY_FEATURES],
            "example": _example_triplet(sets, mats, genus,
                                        np.array([imgs[int(i)].get("author") or "" for i in ref_ids]))}


def _example_triplet(sets, mats, genus, author) -> dict:
    """A, B cung chi (Populus, B gan A nhat theo hinh dang, khac tac gia -> khac cay); C khac chi (Quercus)."""
    ids = mats["color"][0]
    Zs = _zscore(mats["shape"][1])
    pop, que = np.where(genus == "Populus")[0], np.where(genus == "Quercus")[0]
    Dp = _pairwise(Zs[pop], "l2")
    Dp[author[pop][:, None] == author[pop][None, :]] = np.inf      # bo cap cung tac gia (thuong la cung 1 cay)
    a_loc, b_loc = np.unravel_index(np.argmin(Dp), Dp.shape)
    a, b = pop[a_loc], pop[b_loc]
    Dq = _pairwise(Zs[que], "l2")
    c = que[int(np.argmin(Dq.mean(1)))]          # medoid cua Quercus theo hinh dang
    rows = [a, b, c]
    values = {}
    for s, bl, k, label, mul in KEY_FEATURES:
        sl = _block_slices(sets[s]["layout"])[bl]
        values[label] = [float(mats[s][1][r, sl.start + k] * mul) for r in rows]
    dists = {}
    for s, info in sets.items():
        X = mats[s][1]
        Xm = _zscore(X) if info["metric"] == "l2" else X
        D = _pairwise(Xm[rows], info["metric"])
        dists[s] = {"AB": float(D[0, 1]), "AC": float(D[0, 2]), "BC": float(D[1, 2]), "metric": info["metric"]}
    return {"image_ids": [int(ids[r]) for r in rows], "genera": [str(genus[r]) for r in rows],
            "values": values, "distances": dists}
