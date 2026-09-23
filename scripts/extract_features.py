"""Buoc 2-3: trich rut dac trung cho toan bo anh trong data/images va luu vao CSDL SQLite.

    python scripts/extract_features.py [--workers N] [--no-deep] [--limit N] [--no-gate]

Cong kiem soat chat luong (quality gate) sau phan doan: anh chi duoc dua vao CSDL khi
  0.04 <= ti le vung cay <= 0.90  va  tam vung cay lech khoi tam anh <= 30% (moi truc)
-> dam bao "moi anh gom 1 cay hoan chinh nam giua anh". Anh bi loai ghi o results/excluded_images.csv.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treedb import config, db  # noqa: E402
from treedb.features import EXTRACTOR_VERSION, describe, extract_all, load_rgb  # noqa: E402
from treedb.index import fit_stats  # noqa: E402
from treedb.segment import foreground_mask  # noqa: E402

HANDCRAFTED = ["color", "shape", "texture"]
GATE_COVERAGE = (0.04, 0.90)
GATE_OFFSET = 0.30


def _work(filename: str):
    rgb = load_rgb(config.IMG_DIR / filename)
    mask = foreground_mask(rgb)
    feats = extract_all(rgb, mask, sets=HANDCRAFTED)
    region = feats["shape"][0][:10]
    return (filename, float(mask.mean()), (float(region[8]), float(region[9])),
            {s: v for s, (v, _) in feats.items()}, {s: lay for s, (_, lay) in feats.items()})


def passes_gate(cov: float, off: tuple[float, float]) -> bool:
    return GATE_COVERAGE[0] <= cov <= GATE_COVERAGE[1] and abs(off[0]) <= GATE_OFFSET and abs(off[1]) <= GATE_OFFSET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(config.MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    by_name = {r["filename"]: r for r in rows}
    filenames = list(by_name)

    t0 = time.perf_counter()
    extracted: dict[str, tuple] = {}
    layouts: dict = {}
    done = 0
    with mp.Pool(args.workers) as pool:
        for filename, cov, off, feats, lays in pool.imap(_work, filenames, chunksize=4):
            extracted[filename] = (cov, off, feats)
            layouts.update(lays)
            done += 1
            if done % 100 == 0 or done == len(filenames):
                el = time.perf_counter() - t0
                print(f"[extract] {done}/{len(filenames)}  {el:.0f}s  ({el / done * 1000:.0f} ms/anh)", flush=True)

    kept, excluded = [], []
    for fn in filenames:
        cov, off, _ = extracted[fn]
        (kept if args.no_gate or passes_gate(cov, off) else excluded).append((fn, cov, off))
    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(config.RESULTS_DIR / "excluded_images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "fg_coverage", "dx", "dy", "reason"])
        for fn, cov, off in excluded:
            reason = "coverage" if not (GATE_COVERAGE[0] <= cov <= GATE_COVERAGE[1]) else "off-center"
            w.writerow([fn, f"{cov:.4f}", f"{off[0]:.3f}", f"{off[1]:.3f}", reason])
    print(f"[gate] giu {len(kept)} anh, loai {len(excluded)} anh (results/excluded_images.csv)", flush=True)
    if len(kept) < config.MIN_IMAGES:
        print(f"[WARN] sau cong kiem soat con {len(kept)} < {config.MIN_IMAGES} anh", flush=True)

    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    conn = db.connect()
    id_of = {}
    for fn, cov, _ in kept:
        rec = dict(by_name[fn])
        rec["path"] = str(config.IMG_DIR / fn)
        rec["group_label"] = rec.pop("group", "")
        rec["fg_coverage"] = cov
        id_of[fn] = db.upsert_image(conn, rec)
    conn.commit()
    print(f"[db] {len(kept)} anh trong bang images -> {config.DB_PATH}", flush=True)

    for s in HANDCRAFTED:
        rows_s = [(id_of[fn], s, extracted[fn][2][s]) for fn, _, _ in kept]
        db.upsert_feature_set(conn, s, rows_s[0][2].size, describe(s)["description"], EXTRACTOR_VERSION,
                              describe(s)["metric"], layouts[s])
        db.put_features_bulk(conn, rows_s)
    conn.commit()

    if not args.no_deep:
        from treedb.features import deep
        if deep.is_available():
            t1 = time.perf_counter()
            batch = 32
            rows_deep, layout = [], None
            names = [fn for fn, _, _ in kept]
            for i in range(0, len(names), batch):
                chunk = names[i:i + batch]
                rgbs = [load_rgb(config.IMG_DIR / n) for n in chunk]
                V = deep.extract_batch(rgbs)
                if layout is None:
                    layout = deep.extract(rgbs[0])[1]
                rows_deep += [(id_of[n], "deep", V[j]) for j, n in enumerate(chunk)]
            db.upsert_feature_set(conn, "deep", rows_deep[0][2].size, describe("deep")["description"],
                                  EXTRACTOR_VERSION, describe("deep")["metric"], layout)
            db.put_features_bulk(conn, rows_deep)
            conn.commit()
            print(f"[deep] {len(rows_deep)} embedding, {time.perf_counter() - t1:.0f}s", flush=True)
        else:
            print("[deep] khong kha dung (bo qua)", flush=True)

    for fs in db.list_feature_sets(conn):
        s = fs["set_name"]
        ids, X = db.get_feature_matrix(conn, s)
        st = fit_stats(X, fs["metric"])
        for k, v in st.items():
            db.put_stat(conn, s, k, v)
        print(f"[stats] {s}: dim={X.shape[1]} n={len(ids)} scale={st['scale'][0]:.4f}", flush=True)
    conn.commit()
    print("[summary]", db.summary(conn), flush=True)


if __name__ == "__main__":
    main()
