"""Chuan hoa anh tho ve cung kich thuoc (IMAGE_SIZE x IMAGE_SIZE), cay nam giua anh.

- Xoay theo EXIF, chuyen RGB.
- Thu nho sao cho canh ngan = IMAGE_SIZE, sau do cat giua (center crop) thanh hinh vuong.
- Loai anh trung lap gan (perceptual hash / DCT, khoang cach Hamming <= 6).
- Loai cac anh da bi danh dau khi kiem duyet thu cong (data/manual_exclude.csv).
- Ghi data/manifest.csv (moi dong = 1 anh trong CSDL).
"""
from __future__ import annotations

import csv
import hashlib

import numpy as np
from PIL import Image, ImageOps

from . import config


def normalize_image(img: Image.Image, size: int = config.IMAGE_SIZE) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - size) // 2, (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def perceptual_hash(img: Image.Image) -> int:
    """pHash 63 bit: DCT 8x8 (bo DC) cua anh xam 32x32, so voi trung vi - ben voi thay doi nho."""
    import cv2
    g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float32)
    d = cv2.dct(g)[:8, :8].flatten()[1:]
    return int("".join("1" if x > np.median(d) else "0" for x in d), 2)


DUP_THRESHOLD = 6  # khoang cach Hamming toi da de coi la anh trung lap


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _split_holdout(rows: list[dict], n_hold: int = config.HOLDOUT_QUERIES) -> tuple[list[dict], list[dict]]:
    """Tach n_hold anh CO NHAN chi (rai deu) ra lam anh truy van thu (khong dua vao CSDL)."""
    labeled = [i for i, r in enumerate(rows) if r["genus"]]
    if len(rows) - n_hold < config.MIN_IMAGES or len(labeled) < n_hold:
        return rows, []
    pick = {labeled[int(k)] for k in np.linspace(0, len(labeled) - 1, n_hold)}
    config.QUERY_DIR.mkdir(parents=True, exist_ok=True)
    kept, held = [], []
    for i, r in enumerate(rows):
        if i in pick:
            qname = f"query_{len(held) + 1:02d}.jpg"
            (config.IMG_DIR / r["filename"]).replace(config.QUERY_DIR / qname)
            held.append({**r, "filename": qname, "image_id": len(held) + 1})
        else:
            new_id = len(kept) + 1
            new_name = f"tree_{new_id:04d}.jpg"
            if new_name != r["filename"]:
                (config.IMG_DIR / r["filename"]).replace(config.IMG_DIR / new_name)
            kept.append({**r, "filename": new_name, "image_id": new_id})
    return kept, held


def load_manual_exclusions() -> set[str]:
    """Danh sach anh bi loai sau khi kiem duyet thu cong bang mat (data/manual_exclude.csv, khoa = tieu de nguon)."""
    if not config.MANUAL_EXCLUDE.exists():
        return set()
    with open(config.MANUAL_EXCLUDE, encoding="utf-8") as f:
        return {r["source_title"] for r in csv.DictReader(f)}


def run(min_images: int = config.MIN_IMAGES) -> int:
    config.IMG_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RAW_MANIFEST, encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))
    excluded = load_manual_exclusions()
    hashes: list[int] = []
    out_rows = []
    dup = bad = manual = 0
    for r in raw_rows:
        if r["commons_title"] in excluded:
            manual += 1
            continue
        src = config.RAW_DIR / r["raw_file"]
        try:
            img = Image.open(src)
            img = normalize_image(img)
        except Exception:  # noqa: BLE001
            bad += 1
            continue
        h = perceptual_hash(img)
        if any(hamming(h, o) <= DUP_THRESHOLD for o in hashes):
            dup += 1
            continue
        hashes.append(h)
        image_id = len(out_rows) + 1
        fname = f"tree_{image_id:04d}.jpg"
        dest = config.IMG_DIR / fname
        img.save(dest, "JPEG", quality=config.JPEG_QUALITY, optimize=True)
        sha1 = hashlib.sha1(dest.read_bytes()).hexdigest()
        out_rows.append({
            "image_id": image_id, "filename": fname, "width": config.IMAGE_SIZE, "height": config.IMAGE_SIZE,
            "format": "JPEG", "file_size": dest.stat().st_size, "sha1": sha1,
            "raw_id": r["raw_id"], "source_title": r["commons_title"], "source_url": r["page_url"],
            "author": r["author"], "license": r["license"], "genus": r["genus"], "group": r["group"],
            "orig_width": r["orig_width"], "orig_height": r["orig_height"],
        })
    out_rows, query_rows = _split_holdout(out_rows)
    with open(config.MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    with open(config.QUERY_MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(query_rows[0].keys()))
        w.writeheader()
        w.writerows(query_rows)
    print(f"[preprocess] {len(out_rows)} anh CSDL ({config.IMAGE_SIZE}x{config.IMAGE_SIZE}) + "
          f"{len(query_rows)} anh truy van thu (data/queries); loai thu cong: {manual}; trung lap: {dup}; loi: {bad}",
          flush=True)
    if len(out_rows) < min_images:
        print(f"[WARN] chua du {min_images} anh!", flush=True)
    return len(out_rows)


if __name__ == "__main__":
    run()
