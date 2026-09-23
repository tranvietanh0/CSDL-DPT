"""Khong gian dac trung, do do khoang cach va cac cau truc chi muc (kd-tree, K-means).

- Do do khoang cach theo tung bo dac trung (slide bai 8, tr.18): L1, L2, cosine.
- Chuan hoa: bo dung L2 duoc z-score theo tung chieu; moi bo co "scale" = khoang cach trung binh
  giua 2 anh ngau nhien trong CSDL, de d/scale ~ 1 -> cac bo dac trung co the cong voi nhau theo trong so.
- Chi muc: cay k-d tren khong gian PCA (slide bai 7) va phan cum K-means (slide bai 8, tr.26-28)
  dung de cat tia ung vien (filter) truoc khi xep hang chinh xac (refine).
"""
from __future__ import annotations

import pickle
from datetime import datetime

import numpy as np
from scipy.spatial import cKDTree

from . import config, db

EPS = 1e-8
INDEX_KDTREE = "kdtree_pca"
INDEX_KMEANS = "kmeans"


def distance(metric: str, q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Khoang cach tu q (d,) den tung hang cua X (N,d)."""
    if metric == "l1":
        return np.abs(X - q).sum(axis=1)
    if metric == "l2":
        return np.sqrt(((X - q) ** 2).sum(axis=1))
    if metric == "cosine":
        qn = q / (np.linalg.norm(q) + EPS)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
        return 1.0 - Xn @ qn
    raise ValueError(f"metric khong ho tro: {metric}")


def fit_stats(X: np.ndarray, metric: str, n_pairs: int = 4000, seed: int = 0) -> dict[str, np.ndarray]:
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    Xp = (X - mean) / std if metric == "l2" else X
    rng = np.random.default_rng(seed)
    n = len(X)
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    keep = a != b
    d = np.array([distance(metric, Xp[i], Xp[j:j + 1])[0] for i, j in zip(a[keep], b[keep])])
    scale = np.array([max(float(d.mean()), EPS)], np.float32)
    return {"mean": mean, "std": std, "scale": scale}


class FeatureSpace:
    """Nap toan bo vector dac trung trong CSDL vao bo nho va tinh khoang cach ket hop."""

    def __init__(self, conn, sets: list[str] | None = None, weights: dict | None = None):
        self.conn = conn
        available = {s["set_name"]: s for s in db.list_feature_sets(conn)}
        self.sets = [s for s in (sets or list(available)) if s in available]
        if not self.sets:
            raise RuntimeError("CSDL chua co bo dac trung nao - hay chay scripts/extract_features.py")
        self.metric = {s: available[s]["metric"] for s in self.sets}
        self.layout = {s: available[s]["layout"] for s in self.sets}
        self.set_weights(weights)
        self.ids: np.ndarray | None = None
        self.X: dict[str, np.ndarray] = {}
        self.stats: dict[str, dict[str, np.ndarray]] = {}
        for s in self.sets:
            ids, X = db.get_feature_matrix(conn, s)
            if self.ids is None:
                self.ids = ids
            elif not np.array_equal(ids, self.ids):
                raise RuntimeError(f"Bo '{s}' co tap anh khac cac bo con lai - hay trich rut lai")
            self.X[s] = X
            st = db.get_stats(conn, s)
            if not all(k in st for k in ("mean", "std", "scale")):
                st = fit_stats(X, self.metric[s])
                for k, v in st.items():
                    db.put_stat(conn, s, k, v)
            self.stats[s] = st
        self.Xp = {s: self.prep(s, self.X[s]) for s in self.sets}
        self.pos = {int(i): k for k, i in enumerate(self.ids)}

    # ---- trong so ------------------------------------------------------------------------
    def set_weights(self, weights: dict | None):
        w = {s: float((weights or config.DEFAULT_WEIGHTS).get(s, 0.0)) for s in self.sets}
        tot = sum(w.values())
        if tot <= 0:
            w = {s: 1.0 for s in self.sets}
            tot = float(len(self.sets))
        self.weights = {s: v / tot for s, v in w.items()}

    # ---- chuan hoa ----------------------------------------------------------------------
    def prep(self, s: str, v: np.ndarray) -> np.ndarray:
        if self.metric[s] == "l2":
            return (v - self.stats[s]["mean"]) / self.stats[s]["std"]
        return v

    def scale(self, s: str) -> float:
        return float(self.stats[s]["scale"][0])

    # ---- khoang cach --------------------------------------------------------------------
    def set_distances(self, s: str, qvec: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        X = self.Xp[s] if rows is None else self.Xp[s][rows]
        return distance(self.metric[s], self.prep(s, qvec.astype(np.float32)), X)

    def combined_distances(self, feats: dict[str, np.ndarray], rows: np.ndarray | None = None) -> dict:
        """feats: {set: vector}. Tra ve raw[s], norm[s] (= raw/scale) va combined = sum w_s * norm[s]."""
        raw, norm = {}, {}
        n = len(self.ids) if rows is None else len(rows)
        combined = np.zeros(n, np.float64)
        for s in self.sets:
            if s not in feats:
                continue
            raw[s] = self.set_distances(s, feats[s], rows)
            norm[s] = raw[s] / self.scale(s)
            combined += self.weights[s] * norm[s]
        return {"raw": raw, "norm": norm, "combined": combined}

    # ---- khong gian nhung (cho chi muc) --------------------------------------------------
    def embed(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for s in self.sets:
            v = self.prep(s, feats[s].astype(np.float32))
            if self.metric[s] == "cosine":
                v = v / (np.linalg.norm(v) + EPS)
            parts.append(np.sqrt(self.weights[s]) / self.scale(s) * v)
        return np.concatenate(parts).astype(np.float32)

    def embed_matrix(self) -> np.ndarray:
        parts = []
        for s in self.sets:
            V = self.Xp[s]
            if self.metric[s] == "cosine":
                V = V / (np.linalg.norm(V, axis=1, keepdims=True) + EPS)
            parts.append(np.sqrt(self.weights[s]) / self.scale(s) * V)
        return np.concatenate(parts, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------------------
# Xay dung chi muc
# ---------------------------------------------------------------------------------------
class PCA:
    """Phan tich thanh phan chinh bang SVD (numpy) - giam chieu truoc khi dung cay k-d."""

    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = vt[:self.n_components]
        var = s ** 2 / max(len(X) - 1, 1)
        self.explained_variance_ratio_ = var[:self.n_components] / var.sum()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_.T


def kmeans(X: np.ndarray, k: int, n_init: int = 10, max_iter: int = 100, seed: int = 0):
    """K-means (Lloyd) voi khoi tao k-means++ - thuat toan slide bai 8 tr.26:
    (1) khoi tao K diem goc, (2) gan moi ban ghi vao cum gan nhat, (3) cap nhat diem goc = trung binh cum,
    lap den khi khong con thay doi hoac vuot so vong lap. Chay n_init lan, giu ket qua co inertia nho nhat."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = np.empty((k, X.shape[1]), np.float64)
        centers[0] = X[rng.integers(len(X))]
        d2 = ((X - centers[0]) ** 2).sum(axis=1)
        for j in range(1, k):                                  # k-means++
            probs = d2 / d2.sum() if d2.sum() > 0 else np.full(len(X), 1 / len(X))
            centers[j] = X[rng.choice(len(X), p=probs)]
            d2 = np.minimum(d2, ((X - centers[j]) ** 2).sum(axis=1))
        labels = None
        for _it in range(max_iter):
            dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = dist.argmin(axis=1)
            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                m = labels == j
                if m.any():
                    centers[j] = X[m].mean(axis=0)
        inertia = float(dist[np.arange(len(X)), labels].sum())
        if best is None or inertia < best[2]:
            best = (centers.copy(), labels.copy(), inertia)
    return best


def build_indexes(conn, space: FeatureSpace, k: int = config.KMEANS_K,
                  pca_dim: int = config.KDTREE_PCA_DIM, seed: int = 0) -> dict:
    E = space.embed_matrix().astype(np.float64)
    n_comp = int(min(pca_dim, E.shape[1], max(1, E.shape[0] - 1)))
    pca = PCA(n_comp).fit(E)
    Z = pca.transform(E).astype(np.float32)
    meta_kd = {"pca_dim": n_comp, "explained_variance": float(pca.explained_variance_ratio_.sum()),
               "n": int(len(Z)), "sets": space.sets, "weights": space.weights,
               "built_at": datetime.now().isoformat(timespec="seconds")}
    db.put_index(conn, INDEX_KDTREE, "kdtree", pickle.dumps({"pca": pca, "Z": Z, "ids": space.ids}), meta_kd)

    centers, labels, inertia = kmeans(E, min(k, len(E)), seed=seed)
    labels = labels.astype(int)
    sizes = np.bincount(labels, minlength=len(centers)).tolist()
    meta_km = {"k": int(len(centers)), "inertia": inertia, "sizes": sizes, "n": int(len(E)),
               "sets": space.sets, "weights": space.weights,
               "built_at": datetime.now().isoformat(timespec="seconds")}
    db.put_index(conn, INDEX_KMEANS, "kmeans",
                 pickle.dumps({"centroids": centers.astype(np.float32), "ids": space.ids, "labels": labels}),
                 meta_km)
    db.set_clusters(conn, INDEX_KMEANS, [(int(i), int(c)) for i, c in zip(space.ids, labels)])
    return {"kdtree": meta_kd, "kmeans": meta_km}


class KDTreeIndex:
    """Cay k-d (scipy cKDTree) tren toa do PCA cua khong gian nhung."""

    def __init__(self, conn):
        loaded = db.get_index(conn, INDEX_KDTREE)
        if loaded is None:
            raise RuntimeError("Chua co chi muc kd-tree - hay chay scripts/build_index.py")
        payload, self.meta = loaded
        obj = pickle.loads(payload)
        self.pca, self.Z, self.ids = obj["pca"], obj["Z"], obj["ids"]
        self.tree = cKDTree(self.Z)

    def candidates(self, space: FeatureSpace, feats: dict, n: int = config.CANDIDATE_POOL) -> dict:
        z = self.pca.transform(space.embed(feats)[None, :]).astype(np.float32)[0]
        d, idx = self.tree.query(z, k=min(n, len(self.Z)))
        idx = np.atleast_1d(idx)
        return {"rows": idx, "ids": self.ids[idx], "tree_dist": np.atleast_1d(d), "z": z}


class KMeansIndex:
    """Phan cum K-means: chon cum gan cau truy van nhat roi chi xet cac anh trong cum do."""

    def __init__(self, conn):
        loaded = db.get_index(conn, INDEX_KMEANS)
        if loaded is None:
            raise RuntimeError("Chua co chi muc k-means - hay chay scripts/build_index.py")
        payload, self.meta = loaded
        obj = pickle.loads(payload)
        self.centroids, self.ids, self.labels = obj["centroids"], obj["ids"], obj["labels"]

    def candidates(self, space: FeatureSpace, feats: dict, n: int = config.CANDIDATE_POOL,
                   min_clusters: int = config.KMEANS_MIN_CLUSTERS) -> dict:
        e = space.embed(feats)
        cd = np.sqrt(((self.centroids - e) ** 2).sum(axis=1))
        order = np.argsort(cd)
        rows: list[int] = []
        used = []
        for c in order:                       # luon xet it nhat min_clusters cum gan nhat, mo rong neu chua du n
            rows += np.where(self.labels == c)[0].tolist()
            used.append(int(c))
            if len(rows) >= n and len(used) >= min_clusters:
                break
        rows = np.array(rows, dtype=int)
        return {"rows": rows, "ids": self.ids[rows], "clusters": used,
                "centroid_dist": {int(c): float(cd[c]) for c in order}}
