"""He CSDL quan tri dac trung anh (SQLite) - mo hinh lai (slide bai 1, tr.18; bai 6):

  images         : sieu du lieu quan he cua tung anh (ten file, kich thuoc, nguon, tac gia, giay phep, chi/nhom)
  feature_sets   : mo ta tung bo dac trung (so chieu, do do khoang cach, bo cuc cac khoi)
  features       : vector dac trung (BLOB float32) cua (anh, bo dac trung)
  feature_stats  : thong ke chuan hoa (mean, std, scale) cua tung bo
  indexes        : cau truc chi muc da chieu (PCA + kd-tree, K-means) dang pickle
  image_clusters : anh thuoc cum nao (truy van theo phan cum)
  query_log      : nhat ky truy van
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY, filename TEXT UNIQUE NOT NULL, path TEXT NOT NULL,
    width INT, height INT, format TEXT, file_size INT, sha1 TEXT,
    source_title TEXT, source_url TEXT, author TEXT, license TEXT,
    genus TEXT, group_label TEXT, orig_width INT, orig_height INT, fg_coverage REAL, ingested_at TEXT);
CREATE TABLE IF NOT EXISTS feature_sets (
    set_name TEXT PRIMARY KEY, dim INT NOT NULL, description TEXT, extractor_version TEXT,
    metric TEXT, layout_json TEXT);
CREATE TABLE IF NOT EXISTS features (
    image_id INT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    set_name TEXT NOT NULL REFERENCES feature_sets(set_name),
    dim INT NOT NULL, vector BLOB NOT NULL, PRIMARY KEY (image_id, set_name));
CREATE TABLE IF NOT EXISTS feature_stats (
    set_name TEXT NOT NULL, key TEXT NOT NULL, value BLOB NOT NULL, PRIMARY KEY (set_name, key));
CREATE TABLE IF NOT EXISTS indexes (
    name TEXT PRIMARY KEY, kind TEXT, meta_json TEXT, payload BLOB, built_at TEXT);
CREATE TABLE IF NOT EXISTS image_clusters (
    index_name TEXT NOT NULL, image_id INT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    cluster_id INT NOT NULL, PRIMARY KEY (index_name, image_id));
CREATE TABLE IF NOT EXISTS query_log (
    query_id INTEGER PRIMARY KEY, query_path TEXT, backend TEXT, weights_json TEXT, top_k INT,
    results_json TEXT, elapsed_ms REAL, created_at TEXT);
CREATE INDEX IF NOT EXISTS idx_features_set ON features(set_name);
CREATE INDEX IF NOT EXISTS idx_clusters ON image_clusters(index_name, cluster_id);
CREATE INDEX IF NOT EXISTS idx_images_genus ON images(genus);
"""

IMAGE_COLS = ["image_id", "filename", "path", "width", "height", "format", "file_size", "sha1", "source_title",
              "source_url", "author", "license", "genus", "group_label", "orig_width", "orig_height",
              "fg_coverage", "ingested_at"]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_blob(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr, dtype=np.float32).tobytes()


def _from_blob(b: bytes, dim: int | None = None) -> np.ndarray:
    a = np.frombuffer(b, dtype=np.float32)
    return a.reshape(-1, dim) if dim and a.size != dim else a


def connect(path: Path | str = config.DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)  # cho phep Streamlit dung o nhieu luong
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


# ---- images ---------------------------------------------------------------------------
def upsert_image(conn, row: dict) -> int:
    rec = {k: row.get(k) for k in IMAGE_COLS}
    if rec["group_label"] is None and "group" in row:
        rec["group_label"] = row["group"]
    rec["ingested_at"] = rec["ingested_at"] or _now()
    cols = [c for c in IMAGE_COLS if rec[c] is not None or c == "ingested_at"]
    upd = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in ("image_id", "filename"))
    conn.execute(f"INSERT INTO images ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
                 f"ON CONFLICT(filename) DO UPDATE SET {upd}", [rec[c] for c in cols])
    return int(conn.execute("SELECT image_id FROM images WHERE filename=?", (rec["filename"],)).fetchone()[0])


def get_image(conn, image_id: int) -> dict | None:
    r = conn.execute("SELECT * FROM images WHERE image_id=?", (int(image_id),)).fetchone()
    return dict(r) if r else None


def get_images(conn, ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    q = f"SELECT * FROM images WHERE image_id IN ({', '.join('?' * len(ids))})"
    return {int(r["image_id"]): dict(r) for r in conn.execute(q, [int(i) for i in ids])}


def list_images(conn) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM images ORDER BY image_id")]


def count_images(conn) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM images").fetchone()[0])


# ---- feature sets / features ----------------------------------------------------------
def upsert_feature_set(conn, set_name: str, dim: int, description: str = "", extractor_version: str = "",
                       metric: str = "l2", layout=None) -> None:
    conn.execute("INSERT INTO feature_sets (set_name, dim, description, extractor_version, metric, layout_json) "
                 "VALUES (?,?,?,?,?,?) ON CONFLICT(set_name) DO UPDATE SET dim=excluded.dim, "
                 "description=excluded.description, extractor_version=excluded.extractor_version, "
                 "metric=excluded.metric, layout_json=excluded.layout_json",
                 (set_name, int(dim), description, extractor_version, metric,
                  json.dumps([[n, int(d)] for n, d in (layout or [])])))


def list_feature_sets(conn) -> list[dict]:
    out = []
    for r in conn.execute("SELECT * FROM feature_sets ORDER BY rowid"):
        d = dict(r)
        d["layout"] = [(n, int(k)) for n, k in json.loads(d.pop("layout_json") or "[]")]
        out.append(d)
    return out


def put_feature(conn, image_id: int, set_name: str, vec: np.ndarray) -> None:
    put_features_bulk(conn, [(image_id, set_name, vec)])


def put_features_bulk(conn, rows: list[tuple[int, str, np.ndarray]]) -> None:
    with conn:
        conn.executemany("INSERT INTO features (image_id, set_name, dim, vector) VALUES (?,?,?,?) "
                         "ON CONFLICT(image_id, set_name) DO UPDATE SET dim=excluded.dim, vector=excluded.vector",
                         [(int(i), s, int(np.asarray(v).size), _to_blob(v)) for i, s, v in rows])


def get_feature(conn, image_id: int, set_name: str) -> np.ndarray | None:
    r = conn.execute("SELECT vector FROM features WHERE image_id=? AND set_name=?",
                     (int(image_id), set_name)).fetchone()
    return _from_blob(r[0]) if r else None


def get_feature_matrix(conn, set_name: str) -> tuple[np.ndarray, np.ndarray]:
    rows = conn.execute("SELECT image_id, dim, vector FROM features WHERE set_name=? ORDER BY image_id",
                        (set_name,)).fetchall()
    if not rows:
        return np.empty(0, np.int64), np.empty((0, 0), np.float32)
    ids = np.array([int(r[0]) for r in rows], np.int64)
    X = np.stack([_from_blob(r[2]) for r in rows]).astype(np.float32)
    return ids, X


# ---- stats / indexes / clusters -------------------------------------------------------
def put_stat(conn, set_name: str, key: str, arr: np.ndarray) -> None:
    conn.execute("INSERT INTO feature_stats (set_name, key, value) VALUES (?,?,?) "
                 "ON CONFLICT(set_name, key) DO UPDATE SET value=excluded.value", (set_name, key, _to_blob(arr)))


def get_stat(conn, set_name: str, key: str) -> np.ndarray | None:
    r = conn.execute("SELECT value FROM feature_stats WHERE set_name=? AND key=?", (set_name, key)).fetchone()
    return _from_blob(r[0]) if r else None


def get_stats(conn, set_name: str) -> dict[str, np.ndarray]:
    return {r[0]: _from_blob(r[1]) for r in
            conn.execute("SELECT key, value FROM feature_stats WHERE set_name=?", (set_name,))}


def put_index(conn, name: str, kind: str, payload: bytes, meta: dict) -> None:
    with conn:
        conn.execute("INSERT INTO indexes (name, kind, meta_json, payload, built_at) VALUES (?,?,?,?,?) "
                     "ON CONFLICT(name) DO UPDATE SET kind=excluded.kind, meta_json=excluded.meta_json, "
                     "payload=excluded.payload, built_at=excluded.built_at",
                     (name, kind, json.dumps(meta, ensure_ascii=False), payload, _now()))


def get_index(conn, name: str) -> tuple[bytes, dict] | None:
    r = conn.execute("SELECT payload, meta_json FROM indexes WHERE name=?", (name,)).fetchone()
    return (r[0], json.loads(r[1] or "{}")) if r else None


def set_clusters(conn, index_name: str, assignments: list[tuple[int, int]]) -> None:
    with conn:
        conn.execute("DELETE FROM image_clusters WHERE index_name=?", (index_name,))
        conn.executemany("INSERT INTO image_clusters (index_name, image_id, cluster_id) VALUES (?,?,?)",
                         [(index_name, int(i), int(c)) for i, c in assignments])


def get_cluster_members(conn, index_name: str, cluster_id: int) -> list[int]:
    return [int(r[0]) for r in conn.execute(
        "SELECT image_id FROM image_clusters WHERE index_name=? AND cluster_id=? ORDER BY image_id",
        (index_name, int(cluster_id)))]


def get_cluster_of(conn, index_name: str, image_id: int) -> int | None:
    r = conn.execute("SELECT cluster_id FROM image_clusters WHERE index_name=? AND image_id=?",
                     (index_name, int(image_id))).fetchone()
    return int(r[0]) if r else None


# ---- nhat ky / tong hop ---------------------------------------------------------------
def log_query(conn, query_path: str, backend: str, weights: dict, top_k: int, results: list,
              elapsed_ms: float) -> int:
    with conn:
        cur = conn.execute("INSERT INTO query_log (query_path, backend, weights_json, top_k, results_json, "
                           "elapsed_ms, created_at) VALUES (?,?,?,?,?,?,?)",
                           (str(query_path), backend, json.dumps(weights), int(top_k),
                            json.dumps(results, ensure_ascii=False), float(elapsed_ms), _now()))
    return int(cur.lastrowid)


def summary(conn) -> dict:
    path = conn.execute("PRAGMA database_list").fetchone()[2]
    return {
        "db_path": path,
        "db_size_bytes": Path(path).stat().st_size if path and Path(path).exists() else 0,
        "images": count_images(conn),
        "feature_sets": {fs["set_name"]: {"dim": fs["dim"], "metric": fs["metric"], "count": int(conn.execute(
            "SELECT COUNT(*) FROM features WHERE set_name=?", (fs["set_name"],)).fetchone()[0])}
            for fs in list_feature_sets(conn)},
        "indexes": [r[0] for r in conn.execute("SELECT name FROM indexes")],
        "clusters": {r[0]: int(r[1]) for r in conn.execute(
            "SELECT index_name, COUNT(DISTINCT cluster_id) FROM image_clusters GROUP BY index_name")},
        "queries_logged": int(conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]),
    }
