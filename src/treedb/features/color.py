"""Bo dac trung 1 - MAU SAC (slide bai 11, tr.9-18).

Cac khoi (block) trong vector mau, tong 369 chieu:
  hsv_global   72  : bieu do tan suat mau HSV luong tu hoa H=8, S=3, V=3 muc  (n = 8*3*3 o mau)
  hsv_grid    144  : bieu do mau theo 9 vung (luoi 3x3), moi vung 4*2*2 = 16 o  -> quan he khong gian
  hsv_fg       72  : bieu do mau rieng cua VAT THE (tan cay)                       -> tranh hieu ung che mat
  hsv_bg       72  : bieu do mau rieng cua HINH NEN (troi, dat)
  moments       9  : trung binh / do lech chuan / do lech (skewness) cua H,S,V tren vung cay
Tat ca bieu do duoc chuan hoa L1 (tong = 1) de khong phu thuoc so diem anh.
Khong gian HSV duoc chon thay RGB vi it phu thuoc thiet bi va anh sang (slide tr.18).
"""
from __future__ import annotations

import cv2
import numpy as np

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


def grid_hist(hsv: np.ndarray, grid: int = GRID, bins=GRID_BINS) -> np.ndarray:
    H, W = hsv.shape[:2]
    out = []
    for i in range(grid):
        for j in range(grid):
            cell = hsv[i * H // grid:(i + 1) * H // grid, j * W // grid:(j + 1) * W // grid]
            out.append(hsv_hist(cell, None, bins))
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
        ("hsv_global", hsv_hist(hsv)),
        ("hsv_grid", grid_hist(hsv)),
        ("hsv_fg", hsv_hist(hsv, mask)),
        ("hsv_bg", hsv_hist(hsv, ~mask)),
        ("moments", color_moments(hsv, mask)),
    ]
    vec = np.concatenate([b for _, b in blocks]).astype(np.float32)
    layout = [(n, int(b.size)) for n, b in blocks]
    return vec, layout
