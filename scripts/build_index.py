"""Buoc 3b: xay dung chi muc kd-tree (PCA) va phan cum K-means tren CSDL dac trung.

    python scripts/build_index.py [--k 8] [--pca 32]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treedb import config, db  # noqa: E402
from treedb.index import FeatureSpace, build_indexes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.KMEANS_K)
    ap.add_argument("--pca", type=int, default=config.KDTREE_PCA_DIM)
    args = ap.parse_args()
    conn = db.connect()
    space = FeatureSpace(conn)
    print(f"[space] bo dac trung: {space.sets}; trong so: {space.weights}; n = {len(space.ids)}")
    meta = build_indexes(conn, space, k=args.k, pca_dim=args.pca)
    conn.commit()
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "index_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                        encoding="utf-8")


if __name__ == "__main__":
    main()
