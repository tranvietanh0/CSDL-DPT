"""Phan hoi lien quan (relevance feedback) - "qua trinh truy van can co phan hoi" (slide bai 11, tr.7).

Nguoi dung danh dau mot so anh ket qua la LIEN QUAN / KHONG LIEN QUAN; vector truy van cua tung bo dac
trung duoc dich chuyen theo cong thuc Rocchio:

    q' = alpha * q + beta * trung_binh(anh lien quan) - gamma * trung_binh(anh khong lien quan)

roi truy van lai. Bieu do tan suat (bo do bang L1) duoc cat am ve 0 va chuan hoa lai tung khoi de van la
phan bo hop le.
"""
from __future__ import annotations

import numpy as np

ALPHA, BETA, GAMMA = 1.0, 0.75, 0.25


def rocchio(q: np.ndarray, relevant: np.ndarray | None, nonrelevant: np.ndarray | None,
            alpha: float = ALPHA, beta: float = BETA, gamma: float = GAMMA) -> np.ndarray:
    """q (d,), relevant / nonrelevant (n, d) hoac None -> vector truy van moi (d,)."""
    out = alpha * q.astype(np.float64)
    if relevant is not None and len(relevant):
        out += beta * relevant.mean(axis=0)
    if nonrelevant is not None and len(nonrelevant):
        out -= gamma * nonrelevant.mean(axis=0)
    return out.astype(np.float32)


def renormalize_histograms(v: np.ndarray, layout: list[tuple[str, int]], hist_blocks: set[str]) -> np.ndarray:
    """Cat am va chuan hoa L1 lai cac khoi bieu do (giu nguyen cac khoi khac)."""
    v = v.copy()
    i = 0
    for name, d in layout:
        if name in hist_blocks:
            blk = np.clip(v[i:i + d], 0, None)
            s = blk.sum()
            v[i:i + d] = blk / s if s > 0 else blk
        i += d
    return v


def apply_feedback(space, feats: dict[str, np.ndarray], relevant_ids: list[int], nonrelevant_ids: list[int],
                   hist_blocks: set[str]) -> dict[str, np.ndarray]:
    """Tra ve dac trung truy van moi cho moi bo, dung vector da luu trong CSDL cua cac anh duoc danh dau."""
    rel = [space.pos[int(i)] for i in relevant_ids if int(i) in space.pos]
    non = [space.pos[int(i)] for i in nonrelevant_ids if int(i) in space.pos]
    out = {}
    for s, q in feats.items():
        X = space.X[s]
        v = rocchio(q, X[rel] if rel else None, X[non] if non else None)
        if space.metric[s] == "l1":
            v = renormalize_histograms(v, space.layout[s], hist_blocks)
        out[s] = v
    return out
