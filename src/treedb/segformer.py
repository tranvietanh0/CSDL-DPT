"""Phan doan ngu nghia "cay" bang SegFormer-b0 huan luyen tren ADE20K (150 lop, lop 4 = tree).

Model ONNX (~15 MB) tai tu HuggingFace va cache trong config.MODEL_DIR; chay bang ONNX Runtime tren CPU.
Neu khong tai / nap duoc model, is_available() = False va segment.py dung GrabCut (phuong an du phong).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from . import config

MODEL_URL = "https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model.onnx"
MODEL_NAME = "segformer-b0-ade.onnx"
MIN_MODEL_BYTES = 1 << 20
INPUT_SIZE = 512
TREE_CLASS = 4
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

_SESSION = None
_ERROR: str | None = None


def _model_path() -> Path:
    return Path(config.MODEL_DIR) / MODEL_NAME


def _load():
    global _SESSION, _ERROR
    if _SESSION is not None or _ERROR is not None:
        return _SESSION
    path = _model_path()
    try:
        if not (path.is_file() and path.stat().st_size > MIN_MODEL_BYTES):
            path.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(MODEL_URL, headers={"User-Agent": config.USER_AGENT})
            print(f"[segformer] dang tai model: {MODEL_URL}", flush=True)
            with urllib.request.urlopen(req, timeout=300) as r:
                path.write_bytes(r.read())
        import onnxruntime as ort
        _SESSION = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001 - thieu mang / onnxruntime -> dung GrabCut, bao ly do
        _ERROR = f"{type(e).__name__}: {e}"
        print(f"[segformer] khong kha dung, dung GrabCut. Ly do: {_ERROR}", flush=True)
    return _SESSION


def is_available() -> bool:
    return _load() is not None


def tree_mask(rgb: np.ndarray) -> np.ndarray:
    """rgb uint8 HxWx3 -> mask bool (do phan giai dau ra cua model, = INPUT_SIZE/4) cua lop tree."""
    import cv2
    sess = _load()
    x = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    return logits.argmax(0) == TREE_CLASS
