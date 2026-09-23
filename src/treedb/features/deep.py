"""Bo dac trung 4 (mo rong) - EMBEDDING SAU (MobileNetV2, ImageNet).

Y tuong (slide bai 2: "dac trung sau", "embedding"; bai 11 tr.28-30):
Thay vi tu thiet ke cac dac trung mau / hinh dang / ket cau, ta dung mot mang tich chap da
duoc huan luyen truoc tren ImageNet (MobileNetV2) de trich xuat VECTOR DAC TRUNG NGU NGHIA
bac cao. Mang da hoc san cac khai niem thi giac (la, canh, van, hinh dang bo phan...) nen
vector dau ra mo ta NOI DUNG cua anh, bo sung cho cac dac trung tham cong o tren.

Gia tri thong tin: day la "dac trung sau" (deep feature) theo dung nghia bai giang - khong
do con nguoi tu dat ra, ma do mang tu hoc. Nho vay no bat duoc nhung khac biet tinh te
(loai la, kieu tan, cach phan bo canh) ma histogram mau hay GLCM de bo sot.

Hai bien the (tu dong nhan dang theo so chieu dau ra thuc te cua model):
  mobilenetv2_gap     1280 : dau ra lop GlobalAveragePool - VECTOR EMBEDDING ngu nghia.
                             Day la lop gop dac trung cuoi cung TRUOC lop phan loai, nen
                             no khong bi rang buoc vao 1000 lop ImageNet => tong quat hon.
  mobilenetv2_logits  1000 : du phong - dau ra lop cuoi (logits 1000 lop ImageNet) khi
                             khong the tao duoc model "feat" (thieu goi onnx).

Model ONNX (~14 MB) tai tu ONNX Model Zoo va cache trong config.MODEL_DIR:
  mobilenetv2-12.onnx        model goc       (input Nx3x224x224 -> output Nx1000)
  mobilenetv2-12-feat.onnx   model da sua    (them tensor GlobalAveragePool vao graph.output)
Model feat duoc tao bang cach them chinh tensor cua node GlobalAveragePool vao danh sach
graph.output, giu nguyen trong so => khong can huan luyen lai, chi doi "cua ra".

Tien xu ly (theo chuan ImageNet): resize INTER_AREA ve 224x224 -> chia 255 -> tru mean
[0.485, 0.456, 0.406] -> chia std [0.229, 0.224, 0.225].
Vector ket qua duoc chuan hoa L2 => dung do tuong dong cosine (config.SET_METRIC["deep"]).

Bo dac trung nay la TUY CHON - he thong van chay du khi khong co no:
  - features.available_sets() tu dong bo qua "deep" khi chua san sang;
  - deep.is_available() tra ve False, con deep.extract() nem RuntimeError co thong bao ro.
"""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .. import config

MODEL_URL = ("https://github.com/onnx/models/raw/main/validated/vision/"
             "classification/mobilenet/model/mobilenetv2-12.onnx")
MODEL_NAME = "mobilenetv2-12.onnx"
FEAT_NAME = "mobilenetv2-12-feat.onnx"
MIN_MODEL_BYTES = 1 << 20  # duoi 1 MB coi nhu file tai do dang / hong
DOWNLOAD_TIMEOUT = 120  # giay
DOWNLOAD_CHUNK = 1 << 20

INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
EPS = 1e-12

GAP_OP = "GlobalAveragePool"

# mode -> (ten khoi trong layout, so chieu)
BLOCKS = {
    "gap": ("mobilenetv2_gap", 1280),
    "logits": ("mobilenetv2_logits", 1000),
}

_SESSION = None  # onnxruntime.InferenceSession, khoi tao muon (lazy)
_MODE = None  # "gap" | "logits"
_STATE = None  # None = chua thu; True/False = ket qua da cache
_ERROR = ""  # ly do that bai, dung cho thong bao RuntimeError


def ensure_model() -> Path:
    """Dam bao model da co trong config.MODEL_DIR; tra ve file dung de chay.

    Tai mobilenetv2-12.onnx neu chua co, roi thu tao ban "feat" co lop GlobalAveragePool.
    Tra ve duong dan ban feat neu tao duoc, nguoc lai tra ve ban goc (logits 1000 chieu).
    """
    src = _model_path()
    if not _is_complete(src):
        _download(src)
    feat = _feat_path()
    if _is_complete(feat):
        return feat
    return _build_feat(src) or src


def is_available() -> bool:
    """True neu tao duoc session (model da tai + onnxruntime hoat dong). Khong bao gio nem loi."""
    try:
        return _load() is not None
    except Exception:  # noqa: BLE001 - ham nay cam ket khong nem loi
        return False


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """Anh RGB uint8 HxWx3 -> float32 NCHW (1, 3, 224, 224) chuan ImageNet."""
    img = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(x, (2, 0, 1))[None].astype(np.float32)


def extract_batch(rgbs: list[np.ndarray]) -> np.ndarray:
    """Chay model tren ca lo anh; tra ve float32 (N, D) da chuan hoa L2 tung dong."""
    state = _load()
    if state is None:
        raise RuntimeError(_unavailable_message())
    sess, mode = state
    dim = BLOCKS[mode][1]
    if not rgbs:
        return np.zeros((0, dim), np.float32)
    x = np.concatenate([preprocess(rgb) for rgb in rgbs], axis=0)
    if _batch_is_fixed_one(sess):
        # model chi nhan batch = 1 -> chay tung anh roi ghep lai
        y = np.concatenate([_run(sess, x[i:i + 1]) for i in range(x.shape[0])], axis=0)
    else:
        y = _run(sess, x)
    y = y.reshape(y.shape[0], -1).astype(np.float32)  # GAP: (N,1280,1,1) -> (N,1280)
    norms = np.linalg.norm(y, axis=1, keepdims=True)
    return (y / np.maximum(norms, EPS)).astype(np.float32)


def extract(rgb: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Vector embedding cua mot anh + layout.

    Giong chu ky cac bo dac trung khac (color/shape/texture), nhung `mask` bi BO QUA:
    mang da hoc duoc cach tap trung vao vat the chinh, va embedding chi co y nghia khi
    tinh tren TOAN BO anh (cat theo mask se lam lech phan phoi so voi luc huan luyen).
    """
    state = _load()
    if state is None:
        raise RuntimeError(_unavailable_message())
    block, dim = BLOCKS[state[1]]
    vec = extract_batch([rgb])[0]
    return vec.astype(np.float32), [(block, dim)]


# ---------------------------------------------------------------- duong dan / tai model

def _model_path() -> Path:
    return Path(config.MODEL_DIR) / MODEL_NAME


def _feat_path() -> Path:
    return Path(config.MODEL_DIR) / FEAT_NAME


def _is_complete(path: Path) -> bool:
    """File ton tai va lon hon nguong toi thieu (tranh file tai do dang)."""
    try:
        return path.is_file() and path.stat().st_size > MIN_MODEL_BYTES
    except OSError:
        return False


def _download(dest: Path) -> None:
    """Tai model ve dest theo tung khoi, ghi ra file tam roi thay the (atomic)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": config.USER_AGENT})
    print(f"[deep] dang tai model: {MODEL_URL}", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(DOWNLOAD_CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
        if not _is_complete(tmp):
            raise OSError(f"file tai ve qua nho: {tmp} ({tmp.stat().st_size} bytes)")
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"[deep] da tai {dest} ({dest.stat().st_size} bytes)", flush=True)


# ---------------------------------------------------------------- tao model "feat" (co GAP)

def _build_feat(src: Path) -> Path | None:
    """Them tensor GlobalAveragePool vao graph.output cua model goc va luu thanh *_feat.onnx.

    Khong huan luyen lai, khong doi trong so - chi mo them mot "cua ra" o giua mang.
    Tra ve duong dan model feat neu tao duoc, None neu thieu goi onnx hoac loi.
    """
    feat = _feat_path()
    try:
        import onnx  # noqa: WPS433 - tuy chon, chi can khi tao model feat
    except ImportError:
        if _pip_install_onnx():
            try:
                import onnx  # noqa: WPS433
            except ImportError:
                print("[deep] khong import duoc goi onnx -> dung logits 1000 chieu", flush=True)
                return None
        else:
            print("[deep] thieu goi onnx -> dung logits 1000 chieu", flush=True)
            return None
    try:
        model = onnx.load(str(src))
        out_name = _add_gap_output(model.graph)
        if out_name is None:
            return None
        model.graph.output.append(_gap_value_info(onnx, model, out_name))
        onnx.checker.check_model(model)
        feat.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(feat))
    except Exception as exc:  # noqa: BLE001 - moi loi o day deu chi khien ta lui ve logits
        print(f"[deep] khong tao duoc model feat ({exc}) -> dung logits 1000 chieu", flush=True)
        return None
    print(f"[deep] da tao model feat: {feat} (dau ra {out_name} = lop GAP)", flush=True)
    return feat


def _add_gap_output(graph) -> str | None:
    """Chuyen graph.output sang tensor cua lop GlobalAveragePool. None neu khong tim thay."""
    for node in graph.node:
        if node.op_type != GAP_OP:
            continue
        if len(node.output) >= 1 and node.output[0]:
            del graph.output[:]  # bo dau ra cu (logits), chi giu lai "cua ra" GAP
            return node.output[0]
    print(f"[deep] model khong co node {GAP_OP}", flush=True)
    return None


def _gap_value_info(onnx, model, out_name: str):
    """ValueInfo float cho tensor GAP; suy shape bang onnx.shape_inference neu co the."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        for value in list(inferred.graph.value_info) + list(inferred.graph.output):
            if value.name == out_name and value.type.HasField("tensor_type"):
                info = onnx.ValueInfoProto()
                info.CopyFrom(value)
                info.type.ClearField("denotation")
                return info
    except Exception:  # noqa: BLE001 - khong suy duoc thi dung shape toi thieu
        pass
    dims = [1, BLOCKS["gap"][1]]
    return onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, dims)


def _pip_install_onnx() -> bool:
    """Thu `pip install onnx` dung dung trinh Python dang chay. True neu lenh chay thanh cong."""
    try:
        proc = subprocess.run([sys.executable, "-m", "pip", "install", "onnx"],
                              capture_output=True, text=True, timeout=600)
    except Exception as exc:  # noqa: BLE001
        print(f"[deep] khong chay duoc pip install onnx ({exc})", flush=True)
        return False
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
        print(f"[deep] pip install onnx that bai: {tail[0]}", flush=True)
        return False
    print("[deep] da cai goi onnx", flush=True)
    return True


# ---------------------------------------------------------------- session (lazy, cache)

def _load():
    """Tao (va cache) InferenceSession; tra ve (session, mode) hoac None neu khong san sang."""
    global _SESSION, _MODE, _STATE, _ERROR  # noqa: PLW0603 - cache o muc module theo yeu cau
    if _STATE is not None:
        return (_SESSION, _MODE) if _STATE else None
    try:
        import onnxruntime as ort  # noqa: WPS433

        path = ensure_model()
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        output = sess.get_outputs()[0]
        dim = _output_dim(output.shape)
        if dim == BLOCKS["gap"][1]:
            _MODE = "gap"
        elif dim == BLOCKS["logits"][1]:
            _MODE = "logits"
        else:
            _STATE, _ERROR = False, f"so chieu dau ra khong ho tro: {dim} ({output.name})"
            return None
        _SESSION, _STATE, _ERROR = sess, True, ""
        print(f"[deep] san sang: {path.name} -> {output.name} ({_MODE}, {dim} chieu)", flush=True)
        return sess, _MODE
    except Exception as exc:  # noqa: BLE001 - moi loi -> bo qua bo dac trung nay, khong lam sap app
        _STATE, _SESSION, _ERROR = False, None, f"{type(exc).__name__}: {exc}"
        return None


def _output_dim(shape) -> int:
    """So chieu DAC TRUNG cua dau ra: logits la truc cuoi, GAP (N,C,1,1) la truc kenh C."""
    dims = [d for d in (shape or []) if isinstance(d, int) and d > 1]
    return int(dims[-1]) if dims else 0


def _run(sess, x: np.ndarray) -> np.ndarray:
    name = sess.get_inputs()[0].name
    return np.asarray(sess.run(None, {name: x})[0])


def _batch_is_fixed_one(sess) -> bool:
    """True neu truc batch cua input bi ghim = 1 (theo dim_value); batch dong -> False."""
    shape = sess.get_inputs()[0].shape
    if not shape:
        return False
    dim = shape[0]
    return isinstance(dim, int) and dim == 1


def _unavailable_message() -> str:
    return ("Bo dac trung 'deep' chua san sang (thieu onnxruntime hoac khong tai duoc model "
            f"MobileNetV2 vao {config.MODEL_DIR}). Ly do: {_ERROR or 'chua ro'}. "
            "He thong van chay duoc voi color/shape/texture.")
