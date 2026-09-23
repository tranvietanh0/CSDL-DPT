"""Buoc 5: sinh toan bo hinh minh hoa + vi du truy van cho bao cao (docs/figures, results/query_examples).

    python scripts/make_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treedb import config, db, visualize  # noqa: E402
from treedb.search import Searcher, explain, to_jsonable  # noqa: E402

FIG = config.FIG_DIR
QEX = config.RESULTS_DIR / "query_examples"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    QEX.mkdir(parents=True, exist_ok=True)
    conn = db.connect()
    imgs = db.list_images(conn)
    print("[fig] dataset samples ->", visualize.plot_dataset_samples(FIG / "01_dataset_samples.png"))
    print("[fig] dataset stats   ->", visualize.plot_dataset_stats(conn, FIG / "02_dataset_stats.png"))
    seg_names = [imgs[i]["filename"] for i in range(0, len(imgs), max(1, len(imgs) // 5))][:5]
    print("[fig] segmentation    ->", visualize.plot_segmentation_examples(FIG / "03_segmentation.png", seg_names))
    print("[fig] block diagram   ->", visualize.plot_block_diagram(FIG / "04_block_diagram.png"))
    print("[fig] feature dims    ->", visualize.plot_feature_dims(conn, FIG / "05_feature_sets.png"))
    print("[fig] clusters        ->", visualize.plot_clusters(conn, FIG / "06_clusters.png"))
    rep_path = config.RESULTS_DIR / "evaluation.json"
    if rep_path.exists():
        print("[fig] evaluation      ->", visualize.plot_evaluation(visualize.load_report(rep_path),
                                                                    FIG / "07_evaluation.png"))

    # Vi du truy van voi cac anh thu (khong nam trong CSDL)
    searcher = Searcher(conn)
    queries = list(csv.DictReader(open(config.QUERY_MANIFEST, encoding="utf-8")))
    examples = []
    for i, q in enumerate(queries[:4], start=1):
        path = config.QUERY_DIR / q["filename"]
        backend = ["linear", "linear", "kdtree", "kmeans"][i - 1]
        res = searcher.query(path, top_k=config.TOP_K, backend=backend)
        stem = f"query{i:02d}_{backend}"
        visualize.plot_query_result(res, FIG / f"1{i}_{stem}_result.png")
        if i <= 2:
            visualize.plot_query_features(res, FIG / f"1{i}_{stem}_features.png")
        js = to_jsonable(res)
        js["query"]["genus"] = q["genus"]
        js["query"]["group"] = q["group"]
        (QEX / f"{stem}.json").write_text(json.dumps(js, indent=2, ensure_ascii=False), encoding="utf-8")
        (QEX / f"{stem}.txt").write_text(explain(res), encoding="utf-8")
        examples.append({"stem": stem, "query": q["filename"], "genus": q["genus"], "backend": backend,
                         "top": [(r["filename"], r["genus"], round(r["similarity"], 4)) for r in res["results"]]})
        print(f"[query] {stem}: " + ", ".join(f"{f}({g or '?'}:{s})" for f, g, s in examples[-1]["top"]))
    (QEX / "index.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
