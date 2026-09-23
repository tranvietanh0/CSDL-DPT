"""Cau hinh dung chung cho toan bo he thong."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "_cache"
IMG_DIR = DATA_DIR / "images"
RAW_MANIFEST = DATA_DIR / "raw_manifest.csv"
MANIFEST = DATA_DIR / "manifest.csv"
QUERY_DIR = DATA_DIR / "queries"
QUERY_MANIFEST = DATA_DIR / "queries.csv"
MANUAL_EXCLUDE = DATA_DIR / "manual_exclude.csv"
DB_PATH = ROOT / "db" / "tree_features.sqlite"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "docs" / "figures"

# Kich thuoc chuan hoa cua moi anh trong CSDL (vuong, cay nam giua anh)
IMAGE_SIZE = 512
JPEG_QUALITY = 92

# So anh toi thieu theo de bai
MIN_IMAGES = 500
# So anh co nhan duoc tach rieng lam anh truy van thu (khong nam trong CSDL)
HOLDOUT_QUERIES = 10

# Trong so mac dinh khi ket hop cac bo dac trung (tong = 1)
DEFAULT_WEIGHTS = {"color": 0.30, "shape": 0.25, "texture": 0.20, "deep": 0.25}

# Do do khoang cach mac dinh cho tung bo dac trung
SET_METRIC = {"color": "l1", "shape": "l2", "texture": "l2", "deep": "cosine"}

TOP_K = 5
KMEANS_K = 8
KMEANS_MIN_CLUSTERS = 2   # so cum gan nhat toi thieu duoc quet khi truy van theo phan cum
KDTREE_PCA_DIM = 32
CANDIDATE_POOL = 60  # so ung vien lay tu chi muc (kd-tree / k-means) truoc khi xep hang chinh xac

USER_AGENT = "TreeCBIR/1.0 (student project PTIT; contact manhnd@the1studio.org)"
