"""Buoc 4c: danh gia he thong (leave-one-out tren anh co nhan chi).   python scripts/run_evaluation.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treedb import evaluate  # noqa: E402

if __name__ == "__main__":
    evaluate.run()
