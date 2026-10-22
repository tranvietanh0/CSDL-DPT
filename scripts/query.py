"""Buoc 4: tim 5 anh giong nhat voi mot anh cay moi.

    python scripts/query.py --image path/to/tree.jpg [--top-k 5] [--backend linear|kdtree|rtree|quadtree|kmeans]
                            [--genus Quercus,Betula] [--group broadleaf] [--license "CC BY-SA 4.0"]
                            [--weights color=0.3,shape=0.25,texture=0.2,deep=0.25] [--json out.json] [--fig out.png]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treedb import config  # noqa: E402
from treedb.search import BACKENDS, Searcher, explain, to_jsonable  # noqa: E402


def parse_weights(s: str | None) -> dict | None:
    if not s:
        return None
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--backend", default="linear", choices=list(BACKENDS))
    ap.add_argument("--genus", default=None, help="loc ket qua theo chi (danh sach, cach nhau dau phay)")
    ap.add_argument("--group", default=None, help="loc theo nhom hinh thai (broadleaf / conifer)")
    ap.add_argument("--license", default=None, help="loc theo giay phep")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()

    searcher = Searcher(weights=parse_weights(args.weights))
    filters = {f: set(v.split(",")) for f, v in (("genus", args.genus), ("group_label", args.group),
                                                 ("license", args.license)) if v}
    res = searcher.query(args.image, top_k=args.top_k, backend=args.backend, filters=filters or None)
    print(explain(res))
    if args.json:
        Path(args.json).write_text(json.dumps(to_jsonable(res), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[json] {args.json}")
    if args.fig:
        from treedb.visualize import plot_query_result
        plot_query_result(res, args.fig)
        print(f"[fig] {args.fig}")


if __name__ == "__main__":
    main()
