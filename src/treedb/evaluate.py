"""Danh gia he thong tim kiem (leave-one-out tren cac anh co nhan chi/loai).

Do do:
  P@K   : ti le anh trong top-K cung chi (genus) voi anh truy van
  P@1   : anh dau tien co cung chi hay khong
  AP@K  : average precision trong top-K (mAP = trung binh tren cac truy van)
  P@K (nhom): ti le anh trong top-K cung nhom hinh thai (la rong / la kim)
So sanh: tung bo dac trung rieng le, ket hop; backend linear / kd-tree / k-means (recall so voi linear, thoi gian).
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np

from . import config, db
from .search import Searcher

MIN_PER_GENUS = 5


def _labels(conn):
    imgs = db.list_images(conn)
    genus = {int(r["image_id"]): (r.get("genus") or "") for r in imgs}
    group = {int(r["image_id"]): (r.get("group_label") or "unknown") for r in imgs}
    return genus, group


def _metrics(rows: list[dict], q_genus: str, q_group: str, k: int) -> dict:
    rel = [r["genus"] == q_genus for r in rows[:k]]
    grp = [r["group"] == q_group for r in rows[:k]]
    hits, ap = 0, 0.0
    for i, r in enumerate(rel, start=1):
        if r:
            hits += 1
            ap += hits / i
    return {"p_at_k": float(np.mean(rel)) if rel else 0.0, "p_at_1": float(rel[0]) if rel else 0.0,
            "ap_at_k": ap / min(k, max(hits, 1)) if hits else 0.0,
            "group_p_at_k": float(np.mean(grp)) if grp else 0.0}


def evaluate_config(searcher: Searcher, queries: list[int], genus: dict, group: dict, weights: dict,
                    backend: str = "linear", k: int = config.TOP_K) -> dict:
    searcher.space.set_weights(weights)
    agg = Counter()
    t0 = time.perf_counter()
    per_query = {}
    for qid in queries:
        res = searcher.query_by_id(qid, top_k=k, backend=backend)
        m = _metrics(res["results"], genus[qid], group[qid], k)
        per_query[qid] = [r["image_id"] for r in res["results"]]
        for key, v in m.items():
            agg[key] += v
    n = max(len(queries), 1)
    out = {key: agg[key] / n for key in ("p_at_k", "p_at_1", "ap_at_k", "group_p_at_k")}
    out["ms_per_query"] = (time.perf_counter() - t0) * 1000 / n
    out["n_queries"] = len(queries)
    out["weights"] = dict(searcher.space.weights)
    out["backend"] = backend
    out["_top"] = per_query
    return out


def run(k: int = config.TOP_K, n_backend_queries: int = 150, seed: int = 0) -> dict:
    conn = db.connect()
    searcher = Searcher(conn)
    genus, group = _labels(conn)
    counts = Counter(g for g in genus.values() if g)
    eligible = sorted(g for g, c in counts.items() if c >= MIN_PER_GENUS)
    queries = sorted(i for i, g in genus.items() if g in eligible and i in searcher.space.pos)
    n_db = len(searcher.space.ids)
    # co so ngau nhien: xac suat 1 anh ngau nhien cung chi voi truy van
    rand_p = float(np.mean([(counts[genus[q]] - 1) / (n_db - 1) for q in queries])) if queries else 0.0
    grp_counts = Counter(group[i] for i in searcher.space.ids.tolist())
    rand_grp = float(np.mean([(grp_counts[group[q]] - 1) / (n_db - 1) for q in queries])) if queries else 0.0

    sets = searcher.space.sets
    configs = {s: {s: 1.0} for s in sets}
    configs["handcrafted(color+shape+texture)"] = {"color": 0.4, "shape": 0.3, "texture": 0.3}
    configs["combined(default)"] = dict(config.DEFAULT_WEIGHTS)
    configs["color+shape"] = {"color": 0.5, "shape": 0.5}
    if "deep" in sets:
        configs["deep+shape"] = {"deep": 0.6, "shape": 0.4}
        configs["deep+color+shape"] = {"deep": 0.5, "color": 0.25, "shape": 0.25}

    report: dict = {"n_database": n_db, "n_queries": len(queries), "k": k, "genera": {g: counts[g] for g in eligible},
                    "random_baseline": {"p_at_k": rand_p, "group_p_at_k": rand_grp}, "feature_sets": {}}
    for name, w in configs.items():
        r = evaluate_config(searcher, queries, genus, group, w, "linear", k)
        r.pop("_top")
        report["feature_sets"][name] = r
        print(f"[eval] {name:<32} P@{k}={r['p_at_k']:.3f} P@1={r['p_at_1']:.3f} mAP@{k}={r['ap_at_k']:.3f} "
              f"nhom-P@{k}={r['group_p_at_k']:.3f}  ({r['ms_per_query']:.1f} ms/truy van)", flush=True)

    # so sanh backend voi cau hinh mac dinh
    rng = np.random.default_rng(seed)
    sub = sorted(rng.choice(queries, size=min(n_backend_queries, len(queries)), replace=False).tolist())
    base = evaluate_config(searcher, sub, genus, group, config.DEFAULT_WEIGHTS, "linear", k)
    report["backends"] = {"linear": {kk: v for kk, v in base.items() if kk != "_top"}}
    report["backends"]["linear"]["recall_vs_linear"] = 1.0
    for be in ("kdtree", "kmeans"):
        try:
            r = evaluate_config(searcher, sub, genus, group, config.DEFAULT_WEIGHTS, be, k)
        except RuntimeError as e:
            print(f"[eval] {be}: {e}")
            continue
        overlap = [len(set(r["_top"][q]) & set(base["_top"][q])) / k for q in sub]
        r.pop("_top")
        r["recall_vs_linear"] = float(np.mean(overlap))
        report["backends"][be] = r
    for be, r in report["backends"].items():
        print(f"[eval] backend {be:<7} P@{k}={r['p_at_k']:.3f} recall@{k} so voi linear={r['recall_vs_linear']:.3f} "
              f"{r['ms_per_query']:.2f} ms/truy van", flush=True)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = config.RESULTS_DIR / "evaluation.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval] ghi {out}")
    return report


if __name__ == "__main__":
    run()
