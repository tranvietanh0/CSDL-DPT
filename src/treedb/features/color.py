"""Bo dac trung 1 - MAU SAC (slide bai 11, tr.9-18).

Anh trong CSDL da duoc tach nen (cay tren nen trang), nen moi khoi chi tinh tren cac pixel cay
(mask = pixel cay); nen trang khong mang thong tin nen khong co bieu do toan anh / bieu do nen.
Cac khoi (block) trong vector mau, tong 225 chieu:
  hsv_fg       72  : bieu do tan suat mau HSV cua cay, luong tu hoa H=8, S=3, V=3 muc (n = 8*3*3 o mau)
  hsv_grid    144  : bieu do mau cua cay theo 9 vung cua hop bao (luoi 3x3), moi vung 4*2*2 = 16 o
                     -> quan he khong gian: ngon / than, ria trai / phai tan
  moments       9  : trung binh / do lech chuan / do lech (skewness) cua H,S,V tren vung cay
Tat ca bieu do duoc chuan hoa L1 (tong = 1) de khong phu thuoc so diem anh.
Khong gian HSV duoc chon thay RGB vi it phu thuoc thiet bi va anh sang (slide tr.18).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..segment import mask_bbox

HSV_BINS = (8, 3, 3)
GRID = 3
GRID_BINS = (4, 2, 2)
HSV_RANGES = [0, 180, 0, 256, 0, 256]


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def hsv_hist(hsv: np.ndarray, mask: np.ndarray | None = None, bins=HSV_BINS) -> np.ndarray:
    m = None if mask is None else mask.astype(np.uint8)
    h = cv2.calcHist([hsv], [0, 1, 2], m, list(bins), HSV_RANGES).flatten().astype(np.float32)
    s = h.sum()
    return h / s if s > 0 else h


def grid_hist(hsv: np.ndarray, mask: np.ndarray, grid: int = GRID, bins=GRID_BINS) -> np.ndarray:
    """Luoi 3x3 tren hop bao cua cay; o khong co pixel cay -> vector 0."""
    t, l, b, r = mask_bbox(mask)
    hsv, mask = hsv[t:b, l:r], mask[t:b, l:r]
    H, W = hsv.shape[:2]
    out = []
    for i in range(grid):
        for j in range(grid):
            ys, xs = slice(i * H // grid, (i + 1) * H // grid), slice(j * W // grid, (j + 1) * W // grid)
            out.append(hsv_hist(np.ascontiguousarray(hsv[ys, xs]), mask[ys, xs], bins))
    return np.concatenate(out)


def color_moments(hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    px = hsv[mask].astype(np.float32) if mask.any() else hsv.reshape(-1, 3).astype(np.float32)
    px = px / np.array([180.0, 255.0, 255.0], np.float32)
    mean = px.mean(0)
    std = px.std(0)
    skew = ((px - mean) ** 3).mean(0) / (std ** 3 + 1e-6)
    return np.concatenate([mean, std, np.tanh(skew)]).astype(np.float32)


def extract(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[tuple[str, int]]]:
    hsv = to_hsv(rgb)
    blocks = [
        ("hsv_fg", hsv_hist(hsv, mask)),
        ("hsv_grid", grid_hist(hsv, mask)),
        ("moments", color_moments(hsv, mask)),
    ]
    vec = np.concatenate([b for _, b in blocks]).astype(np.float32)
    layout = [(n, int(b.size)) for n, b in blocks]
    return vec, layout
