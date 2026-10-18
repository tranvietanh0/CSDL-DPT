"""Tach vung cay (foreground) khoi nen (sky / dat) - buoc phan doan truoc khi trich rut dac trung.

Y tuong (slide bai 11, tr.17): tach rieng "vat the" va "hinh nen" de xay dung bieu do mau rieng
va trich rut dac trung hinh dang cua vat the.

Thuat toan:
  1. Phan doan ngu nghia bang SegFormer-b0 (ADE20K, lop "tree") - module segformer.py. Mo hinh hoc san biet
     "cay" la gi nen tach duoc ca tan la cung mau voi co nen, cay trui la, anh den trang, tan cham mep anh.
  2. Du phong khi khong co model (hoac model khong thay cay): GrabCut khoi tao bang hinh chu nhat trung tam.
  3. (Chi voi GrabCut) loai bo bau troi (pixel sang, bao hoa thap, hoac xanh lam) khoi foreground.
  4. Loc hinh thai (dong / mo), giu thanh phan lien thong lon va gan tam anh nhat (bo cac cay o hau canh),
     lap lo trong (khe troi giua cac canh) -> mat na "hinh bao" cua cay.
  5. Mat na chu nhat mac dinh chi la phuong an cuoi; ten phuong an duoc tra ve de thong ke.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from . import segformer

WORK_SIZE = 256          # kich thuoc lam viec cua GrabCut (tang toc)
GRABCUT_ITERS = 5
MIN_COVERAGE = 0.02      # mask nho hon -> phuong an tiep theo


def _default_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), bool)
    m[int(h * 0.08):int(h * 0.92), int(w * 0.2):int(w * 0.8)] = True
    return m


def _sky(small: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sky_white = (ss < 40) & (vv > 170)
    sky_blue = (hh > 90) & (hh < 130) & (ss > 40) & (vv > 120)
    return sky_white | sky_blue


def _grabcut(bgr: np.ndarray, mask: np.ndarray, rect, mode: int) -> np.ndarray:
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, GRABCUT_ITERS, mode)
    except cv2.error:
        return np.zeros(mask.shape, np.uint8)
    return np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)


def _clean(fg: np.ndarray, sky: np.ndarray) -> np.ndarray:
    fg = fg.copy()
    fg[sky] = 0
    sh, sw = fg.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)

    n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        cx, cy = sw / 2, sh / 2
        best, best_score = 0, -1.0
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            d = np.hypot(cent[i][0] - cx, cent[i][1] - cy) / max(sw, sh)
            score = area * (1 - 0.8 * d)
            if score > best_score:
                best, best_score = i, score
        fg = (lab == best).astype(np.uint8)
    return ndimage.binary_fill_holes(fg.astype(bool))


def foreground_mask(rgb: np.ndarray, return_method: bool = False):
    """rgb: uint8 HxWx3  ->  mask bool HxW (True = cay); return_method=True -> (mask, ten phuong an)."""
    h, w = rgb.shape[:2]
    scale = WORK_SIZE / max(h, w)
    small = cv2.resize(rgb, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    sky = _sky(small)
    rect = (int(sw * 0.08), int(sh * 0.04), int(sw * 0.84), int(sh * 0.92))

    fg, method = None, ""
    if segformer.is_available():
        tree = cv2.resize(segformer.tree_mask(rgb).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
        fg, method = _clean(tree, np.zeros_like(sky)), "segformer"   # model da tach troi, khong loc mau
    if fg is None or fg.mean() < MIN_COVERAGE:
        fg = _clean(_grabcut(bgr, np.zeros((sh, sw), np.uint8), rect, cv2.GC_INIT_WITH_RECT), sky)
        method = "grabcut"
    if fg.mean() < MIN_COVERAGE:
        mask, method = _default_mask(h, w), "default_rect"
    else:
        mask = cv2.resize(fg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return (mask, method) if return_method else mask


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(top, left, bottom, right) - bottom/right exclusive."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        h, w = mask.shape
        return 0, 0, h, w
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def overlay(rgb: np.ndarray, mask: np.ndarray, color=(255, 40, 40), alpha=0.45) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    out[mask] = out[mask] * (1 - alpha) + np.array(color, np.float32) * alpha
    return out.astype(np.uint8)
