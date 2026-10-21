"""Trich rut dac trung: mot diem vao chung cho moi bo dac trung.

Moi bo dac trung la mot module co ham  extract(rgb, mask) -> (vector float32, layout)
  layout = [(ten_khoi, so_chieu), ...]  dung de giai thich ket qua trung gian.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from .. import config
from ..preprocess import normalize_image
from ..segment import foreground_mask
from . import color, shape, texture

EXTRACTOR_VERSION = "1.0.0"

SET_INFO = {
    "color": {"module": color, "metric": config.SET_METRIC["color"],
              "description": "Mau sac: bieu do HSV toan cuc (8x3x3), theo luoi 3x3, rieng vat the / nen, moment mau"},
    "shape": {"module": shape, "metric": config.SET_METRIC["shape"],
              "description": "Hinh dang tan cay: do do vung, moment Hu, luoi 8x8 (nguong 15%), Fourier, profile be rong"},
    "texture": {"module": texture, "metric": config.SET_METRIC["texture"],
                "description": "Ket cau / bo cuc: GLCM, LBP, bieu do huong canh, Tamura, he so DCT tan so thap"},
}


def _deep_module():
    try:
        from . import deep  # noqa: WPS433 - tuy chon, can onnxruntime + model
        return deep if deep.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def available_sets(include_deep: bool = True) -> list[str]:
    sets = ["color", "shape", "texture"]
    if include_deep and _deep_module() is not None:
        sets.append("deep")
    return sets


def load_rgb(path: str | Path) -> np.ndarray:
    """Doc anh bat ky, chuan hoa giong het anh trong CSDL (IMAGE_SIZE x IMAGE_SIZE, RGB)."""
    img = Image.open(path)
    return np.asarray(normalize_image(img), dtype=np.uint8)


def extract_all(rgb: np.ndarray, mask: np.ndarray | None = None, sets: list[str] | None = None,
                with_timing: bool = False):
    """Tra ve {set_name: (vector, layout)}; mask duoc tinh neu chua co."""
    if mask is None:
        mask = foreground_mask(rgb)
    sets = sets or available_sets()
    out, timing = {}, {}
    for name in sets:
        t0 = time.perf_counter()
        if name == "deep":
            mod = _deep_module()
            if mod is None:
                continue
            out[name] = mod.extract(rgb, mask)
        else:
            out[name] = SET_INFO[name]["module"].extract(rgb, mask)
        timing[name] = (time.perf_counter() - t0) * 1000
    return (out, timing) if with_timing else out


def describe(set_name: str) -> dict:
    if set_name == "deep":
        return {"metric": config.SET_METRIC["deep"],
                "description": "Embedding sau MobileNetV2 (ImageNet) - dac trung ngu nghia bac cao"}
    return {"metric": SET_INFO[set_name]["metric"], "description": SET_INFO[set_name]["description"]}


def split_blocks(vec: np.ndarray, layout: list[tuple[str, int]]) -> dict[str, np.ndarray]:
    out, i = {}, 0
    for name, d in layout:
        out[name] = vec[i:i + d]
        i += d
    return out
