"""Buoc 1: suu tam + chuan hoa bo du lieu anh cay.   python scripts/build_dataset.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from treedb import collect, preprocess  # noqa: E402

if __name__ == "__main__":
    collect.run()
    preprocess.run()
