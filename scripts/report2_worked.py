"""Tinh lai tung buoc do tuong dong giua anh truy van va mot anh ket qua (muc 4b cua Bao cao 2).

Doc vector truy van tu results/query_examples/<stem>.json va vector anh CSDL tu SQLite, tinh lai
khoang cach tung bo theo dung cong thuc cua treedb.index (L1, L2 tren z-score, cosine), tach dong gop theo
tung khoi, roi doi chieu voi gia tri he thong da luu. Lech > 1e-3 -> loi (bao cao khong duoc in so sai).
"""
from __future__ import annotations

import numpy as np

from treedb import db
from treedb.index import FeatureSpace

TOL = 1e-3


def _query_vec(js: dict, s: str, layout: list[tuple[str, int]]) -> np.ndarray:
    blocks = js["query"]["features"][s]["blocks"]
    return np.concatenate([np.asarray(blocks[name], np.float32) for name, _ in layout])


def _block_slices(layout):
    start = 0
    for name, dim in layout:
        yield name, slice(start, start + dim)
        start += dim


def worked_example(js: dict, rank: int = 1) -> dict:
    """Tra ve {set: {...}} voi chi tiet tinh toan cho ket qua hang `rank` cua truy van da luu trong js."""
    res = js["results"][rank - 1]
    conn = db.connect()
    space = FeatureSpace(conn, js["sets"], js["weights"])
    out: dict = {"result": res, "sets": {}}
    total = 0.0
    for s in js["sets"]:
        layout = space.layout[s]
        q = space.prep(s, _query_vec(js, s, layout))
        x = space.prep(s, db.get_feature(conn, res["image_id"], s))
        metric = space.metric[s]
        info: dict = {"metric": metric, "blocks": []}
        if metric == "l1":
            diff = np.abs(q - x)
            raw = float(diff.sum())
            info["blocks"] = [(n, float(diff[sl].sum())) for n, sl in _block_slices(layout)]
            fg = dict(_block_slices(layout))["hsv_fg"]
            top = np.argsort(-diff[fg])[:8]
            info["bins"] = [(int(i), float(q[fg][i]), float(x[fg][i]), float(diff[fg][i])) for i in top]
        elif metric == "l2":
            sq = (q - x) ** 2
            raw = float(np.sqrt(sq.sum()))
            info["blocks"] = [(n, float(sq[sl].sum())) for n, sl in _block_slices(layout)]
        else:
            nq, nx = float(np.linalg.norm(q)), float(np.linalg.norm(x))
            dot = float(q @ x)
            raw = 1.0 - dot / (nq * nx + 1e-8)
            info.update(dot=dot, nq=nq, nx=nx)
        saved = res["per_set"][s]
        if abs(raw - saved["raw"]) > TOL * max(1.0, abs(saved["raw"])):
            raise RuntimeError(f"Tinh lai {s} = {raw:.5f} khac gia tri he thong {saved['raw']:.5f}")
        scale = space.scale(s)
        info.update(raw=raw, scale=scale, norm=raw / scale, weight=space.weights[s],
                    contrib=space.weights[s] * raw / scale)
        total += info["contrib"]
        out["sets"][s] = info
    conn.close()
    if abs(total - res["distance"]) > TOL:
        raise RuntimeError(f"D tinh lai {total:.5f} khac {res['distance']:.5f}")
    out["D"], out["sim"] = total, 1.0 / (1.0 + total)
    return out
