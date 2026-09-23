"""Bo dac trung 3 - KET CAU / BO CUC (slide bai 11, tr.26-27; bai 2, tr.20-21).

Tinh tren anh xam, chu yeu trong vung cay. Tong 85 chieu:
  glcm        10 : ma tran dong xuat hien muc xam (GLCM), 5 thuoc tinh x 2 khoang cach (trung binh 4 huong)
                   contrast, dissimilarity, homogeneity, energy, correlation
  lbp         28 : bieu do Local Binary Pattern (P=8,R=1 -> 10 o; P=16,R=2 -> 18 o) - ket cau tan la
  edge         9 : bieu do huong canh (8 huong, trong so bien do) + mat do canh   - "do sac cua cac net"
  tamura       3 : do tho (coarseness), do tuong phan (contrast), tinh dinh huong (directionality)
  dct         35 : 35 he so DCT tan so thap (khoi 6x6, bo DC) cua anh xam 64x64   - bo cuc tong the (mien nen)
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from ..segment import mask_bbox

GLCM_LEVELS = 32
GLCM_DISTANCES = (1, 3)
GLCM_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
EDGE_BINS = 8
DCT_SIZE = 64
DCT_BLOCK = 6


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _crop(gray: np.ndarray, mask: np.ndarray):
    t, l, b, r = mask_bbox(mask)
    return gray[t:b, l:r], mask[t:b, l:r]


def glcm_features(gray_crop: np.ndarray) -> np.ndarray:
    q = (gray_crop.astype(np.int32) * GLCM_LEVELS // 256).astype(np.uint8)
    if q.shape[0] < 4 or q.shape[1] < 4:
        return np.zeros(len(GLCM_PROPS) * len(GLCM_DISTANCES), np.float32)
    g = graycomatrix(q, distances=list(GLCM_DISTANCES), angles=list(GLCM_ANGLES), levels=GLCM_LEVELS,
                     symmetric=True, normed=True)
    out = []
    for prop in GLCM_PROPS:
        v = graycoprops(g, prop).mean(axis=1)        # trung binh theo 4 huong -> bat bien xoay
        if prop == "contrast":
            v = v / (GLCM_LEVELS - 1) ** 2
        elif prop == "dissimilarity":
            v = v / (GLCM_LEVELS - 1)
        out.append(v)
    return np.concatenate(out).astype(np.float32)


def lbp_hist(gray_crop: np.ndarray, mask_crop: np.ndarray) -> np.ndarray:
    out = []
    for P, R in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(gray_crop, P, R, method="uniform")
        vals = lbp[mask_crop] if mask_crop.any() else lbp.flatten()
        h, _ = np.histogram(vals, bins=P + 2, range=(0, P + 2))
        h = h.astype(np.float32)
        out.append(h / max(h.sum(), 1))
    return np.concatenate(out)


def edge_hist(gray_crop: np.ndarray, mask_crop: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray_crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_crop, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi           # 0..pi (huong canh khong co dau)
    sel = mask_crop if mask_crop.any() else np.ones_like(mask_crop, bool)
    h, _ = np.histogram(ang[sel], bins=EDGE_BINS, range=(0, np.pi), weights=mag[sel])
    h = h.astype(np.float32)
    h = h / max(h.sum(), 1e-6)
    density = float((mag[sel] > 60).mean()) if sel.any() else 0.0
    return np.concatenate([h, [density]]).astype(np.float32)


def tamura(gray_crop: np.ndarray, mask_crop: np.ndarray) -> np.ndarray:
    g = gray_crop.astype(np.float32)
    sel = mask_crop if mask_crop.any() else np.ones_like(mask_crop, bool)
    # coarseness: kich thuoc cua so toi uu (1..5) tai do chenh lech trung binh lon nhat
    best = np.zeros_like(g)
    best_k = np.ones_like(g)
    for k in range(1, 6):
        s = 2 ** k
        if s * 2 >= min(g.shape):
            break
        avg = cv2.blur(g, (s, s))
        dh = np.abs(np.roll(avg, s, axis=1) - np.roll(avg, -s, axis=1))
        dv = np.abs(np.roll(avg, s, axis=0) - np.roll(avg, -s, axis=0))
        e = np.maximum(dh, dv)
        upd = e > best
        best[upd] = e[upd]
        best_k[upd] = s
    coarseness = float(np.log2(best_k[sel]).mean() / 5.0)
    # contrast: sigma / kurtosis^(1/4)
    px = g[sel]
    mu, sd = px.mean(), px.std()
    kurt = ((px - mu) ** 4).mean() / (sd ** 4 + 1e-6) if sd > 0 else 0.0
    contrast = float(sd / (kurt ** 0.25 + 1e-6) / 128.0) if kurt > 0 else 0.0
    # directionality: do "nhon" cua bieu do huong gradient (1 - entropy chuan hoa)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)[sel]
    ang = ((np.arctan2(gy, gx) + np.pi) % np.pi)[sel]
    strong = mag > 30
    if strong.sum() > 10:
        h, _ = np.histogram(ang[strong], bins=16, range=(0, np.pi))
        p = h / h.sum()
        ent = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(16)
        directionality = float(1 - ent)
    else:
        directionality = 0.0
    return np.array([coarseness, contrast, directionality], np.float32)


def dct_features(gray: np.ndarray) -> np.ndarray:
    g = cv2.resize(gray, (DCT_SIZE, DCT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)
    blk = d[:DCT_BLOCK, :DCT_BLOCK].flatten()[1:]        # bo he so DC (do sang trung binh)
    return (blk / (255.0 * DCT_SIZE / 4)).astype(np.float32)


def extract(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[tuple[str, int]]]:
    gray = to_gray(rgb)
    gc, mc = _crop(gray, mask)
    blocks = [
        ("glcm", glcm_features(gc)),
        ("lbp", lbp_hist(gc, mc)),
        ("edge", edge_hist(gc, mc)),
        ("tamura", tamura(gc, mc)),
        ("dct", dct_features(gray)),
    ]
    vec = np.concatenate([b for _, b in blocks]).astype(np.float32)
    layout = [(n, int(b.size)) for n, b in blocks]
    return vec, layout
