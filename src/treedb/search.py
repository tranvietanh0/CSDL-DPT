"""He thong tim kiem anh tuong dong (giai doan truc tuyen - online).

Quy trinh mot truy van (slide bai 1, tr.26-28):
  anh vao -> chuan hoa (nhu CSDL) -> phan doan vat the -> trich rut dac trung (cung cach voi CSDL)
        -> chon ung vien (linear / kd-tree / k-means) -> tinh khoang cach tung bo -> chuan hoa
        -> ket hop theo trong so -> do tuong dong = 1/(1+D) -> xep hang giam dan -> top-K.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import config, db
from .features import extract_all, load_rgb, split_blocks
from .index import FeatureSpace, KDTreeIndex, KMeansIndex
from .segment import foreground_mask

BACKENDS = ("linear", "kdtree", "kmeans")


def similarity_from_distance(d: np.ndarray | float):
    return 1.0 / (1.0 + d)


class Searcher:
    def __init__(self, conn=None, sets: list[str] | None = None, weights: dict | None = None):
        self.conn = conn or db.connect()
        self.space = FeatureSpace(self.conn, sets, weights)
        self._kd: KDTreeIndex | None = None
        self._km: KMeansIndex | None = None

    # ---- chi muc (nap luoi) -----------------------------------------------------------
    @property
    def kd(self) -> KDTreeIndex:
        if self._kd is None:
            self._kd = KDTreeIndex(self.conn)
        return self._kd

    @property
    def km(self) -> KMeansIndex:
        if self._km is None:
            self._km = KMeansIndex(self.conn)
        return self._km

    # ---- truy van --------------------------------------------------------------------
    def query(self, image, top_k: int = config.TOP_K, backend: str = "linear", weights: dict | None = None,
              exclude_ids: set[int] | None = None, log: bool = True) -> dict:
        """image: duong dan anh hoac mang RGB uint8. Tra ve dict ket qua kem cac gia tri trung gian."""
        t_all = time.perf_counter()
        timing: dict[str, float] = {}
        if weights:
            self.space.set_weights(weights)

        t0 = time.perf_counter()
        if isinstance(image, (str, Path)):
            path, rgb = str(image), load_rgb(image)
        else:
            path, rgb = "<array>", np.asarray(image, dtype=np.uint8)
        timing["load_normalize"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        mask = foreground_mask(rgb)
        timing["segment"] = (time.perf_counter() - t0) * 1000

        feats_full, t_ext = extract_all(rgb, mask, sets=self.space.sets, with_timing=True)
        timing.update({f"extract_{k}": v for k, v in t_ext.items()})
        feats = {s: v for s, (v, _) in feats_full.items()}

        result = self._rank(feats, top_k, backend, exclude_ids, timing)
        result.update({
            "query": {"path": path, "rgb": rgb, "mask": mask, "mask_coverage": float(mask.mean()),
                      "features": {s: {"vector": v, "layout": lay, "blocks": split_blocks(v, lay)}
                                   for s, (v, lay) in feats_full.items()}},
        })
        result["timing_ms"]["total"] = (time.perf_counter() - t_all) * 1000
        if log:
            db.log_query(self.conn, path, backend, self.space.weights, top_k,
                         [{"rank": r["rank"], "image_id": r["image_id"], "similarity": r["similarity"]}
                          for r in result["results"]], result["timing_ms"]["total"])
        return result

    def query_by_id(self, image_id: int, top_k: int = config.TOP_K, backend: str = "linear",
                    weights: dict | None = None, exclude_self: bool = True) -> dict:
        """Truy van bang mot anh da co trong CSDL (dung dac trung da luu) - phuc vu danh gia."""
        if weights:
            self.space.set_weights(weights)
        row = self.space.pos[int(image_id)]
        feats = {s: self.space.X[s][row] for s in self.space.sets}
        excl = {int(image_id)} if exclude_self else None
        result = self._rank(feats, top_k, backend, excl, {})
        result["query"] = {"image_id": int(image_id), "features": {s: {"vector": v} for s, v in feats.items()}}
        return result

    # ---- xep hang --------------------------------------------------------------------
    def _rank(self, feats: dict, top_k: int, backend: str, exclude_ids: set[int] | None, timing: dict) -> dict:
        if backend not in BACKENDS:
            raise ValueError(f"backend phai thuoc {BACKENDS}")
        t0 = time.perf_counter()
        index_info: dict = {}
        if backend == "linear":
            rows = np.arange(len(self.space.ids))
        elif backend == "kdtree":
            pool = config.CANDIDATE_POOL + (len(exclude_ids) if exclude_ids else 0)
            c = self.kd.candidates(self.space, feats, pool)
            rows = c["rows"]
            index_info = {"pca_dim": self.kd.meta["pca_dim"], "candidate_ids": c["ids"].tolist(),
                          "tree_dist": c["tree_dist"].tolist()}
        else:
            pool = config.CANDIDATE_POOL + (len(exclude_ids) if exclude_ids else 0)
            c = self.km.candidates(self.space, feats, pool)
            rows = c["rows"]
            index_info = {"clusters_used": c["clusters"], "centroid_dist": c["centroid_dist"],
                          "candidate_ids": c["ids"].tolist()}
        if exclude_ids:
            keep = np.array([int(self.space.ids[r]) not in exclude_ids for r in rows], bool)
            rows = rows[keep]
        timing["candidates"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dist = self.space.combined_distances(feats, rows)
        order = np.argsort(dist["combined"], kind="stable")[:top_k]
        timing["distance_rank"] = (time.perf_counter() - t0) * 1000

        ids = [int(self.space.ids[rows[o]]) for o in order]
        meta = db.get_images(self.conn, ids)
        results = []
        for rank, o in enumerate(order, start=1):
            iid = int(self.space.ids[rows[o]])
            D = float(dist["combined"][o])
            results.append({
                "rank": rank, "image_id": iid, "filename": meta[iid]["filename"], "path": meta[iid]["path"],
                "genus": meta[iid].get("genus") or "", "group": meta[iid].get("group_label") or "",
                "distance": D, "similarity": float(similarity_from_distance(D)),
                "per_set": {s: {"raw": float(dist["raw"][s][o]), "norm": float(dist["norm"][s][o]),
                                "weight": self.space.weights[s],
                                "contrib": float(self.space.weights[s] * dist["norm"][s][o])}
                            for s in dist["raw"]},
            })
        return {"backend": backend, "weights": dict(self.space.weights), "sets": list(self.space.sets),
                "n_candidates": int(len(rows)), "n_database": int(len(self.space.ids)),
                "index_info": index_info, "timing_ms": timing, "results": results,
                "all_distances": {"rows": rows, "combined": dist["combined"]}}


def explain(result: dict) -> str:
    """Bang van ban mo ta ket qua truy van (dung cho CLI / bao cao)."""
    lines = []
    q = result.get("query", {})
    if "path" in q:
        lines.append(f"Anh truy van: {q['path']}  (ti le vung cay: {q['mask_coverage']:.2%})")
    lines.append(f"Backend: {result['backend']}  | ung vien: {result['n_candidates']}/{result['n_database']}"
                 f"  | trong so: " + ", ".join(f"{s}={w:.2f}" for s, w in result["weights"].items()))
    if q.get("features"):
        for s, f in q["features"].items():
            v = f["vector"]
            lines.append(f"  dac trung [{s}] dim={v.size}: " + ", ".join(f"{n}({d})" for n, d in f["layout"]))
    hdr = f"{'#':>2} {'anh':<14} {'chi':<12} {'sim':>6} {'D':>7} | " + " ".join(f"{s:>9}" for s in result["sets"])
    lines.append(hdr)
    for r in result["results"]:
        per = " ".join(f"{r['per_set'][s]['norm']:>9.3f}" for s in result["sets"] if s in r["per_set"])
        lines.append(f"{r['rank']:>2} {r['filename']:<14} {r['genus'][:12]:<12} {r['similarity']:>6.3f} "
                     f"{r['distance']:>7.3f} | {per}")
    t = result["timing_ms"]
    lines.append("Thoi gian (ms): " + ", ".join(f"{k}={v:.1f}" for k, v in t.items()))
    return "\n".join(lines)


def to_jsonable(result: dict) -> dict:
    """Loai bo mang lon (anh, mask) de ghi JSON."""
    out = {k: v for k, v in result.items() if k not in ("all_distances",)}
    q = dict(result.get("query", {}))
    q.pop("rgb", None)
    q.pop("mask", None)
    if "features" in q:
        q["features"] = {s: {"dim": int(f["vector"].size), "layout": f.get("layout"),
                             "blocks": {n: [round(float(x), 5) for x in b] for n, b in f.get("blocks", {}).items()}}
                         for s, f in q["features"].items()}
    out["query"] = q
    return out
