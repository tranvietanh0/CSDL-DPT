"""Bo dac trung 2 - HINH DANG cua tan/than cay (slide bai 11, tr.19-25).

Tinh tren mask nhi phan cua vat the (cay). Tong 113 chieu:
  region      10 : dien tich tuong doi, ti le khung (cao/rong), extent, solidity, tam sai (eccentricity),
                   truc chinh / truc phu (chuan hoa), huong, do lech tam (dx, dy)      -> slide tr.20
  hu           7 : 7 moment bat bien Hu (log)                                            -> slide tr.25
  grid        64 : ma tran luoi 8x8 tren hinh chu nhat co ban, o = 1 neu > 15% dien tich la vat the -> slide tr.21
  fourier     16 : 16 he so Fourier (bien do, chuan hoa) cua duong bao ngoai              -> slide tr.25
  profile     16 : ti le be rong tan cay theo 16 dai ngang tu dinh xuong goc (hinh dang tan / than)
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import regionprops

from ..segment import mask_bbox

GRID_N = 8
GRID_THRESHOLD = 0.15
FOURIER_N = 16
PROFILE_BANDS = 16
CONTOUR_SAMPLES = 128


def region_measures(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    props = regionprops(mask.astype(np.uint8))
    if not props:
        return np.zeros(10, np.float32)
    p = max(props, key=lambda r: r.area)
    top, left, bottom, right = p.bbox
    bh, bw = bottom - top, right - left
    cy, cx = p.centroid
    major = p.axis_major_length / max(H, W)
    minor = p.axis_minor_length / max(H, W)
    return np.array([
        p.area / (H * W),                    # dien tich tuong doi
        bh / max(bw, 1),                     # ti le cao / rong cua hinh chu nhat co ban
        p.extent,                            # area / bbox area
        p.solidity,                          # area / convex hull area
        p.eccentricity,                      # tam sai
        major, minor,                        # truc chinh, truc phu
        abs(p.orientation) / (np.pi / 2),    # huong truc chinh (0 = dung, 1 = ngang)
        cx / W - 0.5, cy / H - 0.5,          # do lech tam so voi tam anh
    ], np.float32)


def hu_moments(mask: np.ndarray) -> np.ndarray:
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    hu = cv2.HuMoments(m).flatten()
    return (-np.sign(hu) * np.log10(np.abs(hu) + 1e-12)).astype(np.float32)


def grid_descriptor(mask: np.ndarray, n: int = GRID_N, thr: float = GRID_THRESHOLD) -> np.ndarray:
    top, left, bottom, right = mask_bbox(mask)
    crop = mask[top:bottom, left:right].astype(np.float32)
    cover = cv2.resize(crop, (n, n), interpolation=cv2.INTER_AREA)  # ti le dien tich vat the trong moi o
    return (cover > thr).astype(np.float32).flatten()


def fourier_descriptor(mask: np.ndarray, n: int = FOURIER_N) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(n, np.float32)
    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(c) < 8:
        return np.zeros(n, np.float32)
    idx = np.linspace(0, len(c) - 1, CONTOUR_SAMPLES).astype(int)
    z = c[idx, 0] + 1j * c[idx, 1]
    F = np.fft.fft(z)
    mag = np.abs(F)
    if mag[1] < 1e-6:
        return np.zeros(n, np.float32)
    desc = mag[2:2 + n] / mag[1]          # bo DC (tinh tien) va chia F1 (ti le); bien do -> bat bien xoay
    return desc.astype(np.float32)


def width_profile(mask: np.ndarray, bands: int = PROFILE_BANDS) -> np.ndarray:
    top, left, bottom, right = mask_bbox(mask)
    crop = mask[top:bottom, left:right]
    bh = crop.shape[0]
    out = np.zeros(bands, np.float32)
    for i in range(bands):
        band = crop[i * bh // bands:max((i + 1) * bh // bands, i * bh // bands + 1)]
        out[i] = band.any(axis=0).mean() if band.size else 0.0  # ti le be rong bi chiem trong dai
    return out


def extract(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[tuple[str, int]]]:
    blocks = [
        ("region", region_measures(mask)),
        ("hu", hu_moments(mask)),
        ("grid", grid_descriptor(mask)),
        ("fourier", fourier_descriptor(mask)),
        ("profile", width_profile(mask)),
    ]
    vec = np.concatenate([b for _, b in blocks]).astype(np.float32)
    layout = [(n, int(b.size)) for n, b in blocks]
    return vec, layout
