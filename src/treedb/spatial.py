"""Cay R va cay tu phan vung diem (PR quadtree) - cau truc du lieu da chieu (slide bai 7, tr.41-60).

Ca hai deu cai dat bang numpy, dung tren toa do PCA cua khong gian nhung (giong cay k-d trong index.py)
va tra ve n lang gieng gan nhat lam tap ung vien truoc khi xep hang chinh xac.

- Cay R: moi nut luu hinh chu nhat bao nho nhat (MBR) cua cac con; la chua toi da LEAF_SIZE diem.
  Xay dung theo phuong phap nap lo Sort-Tile-Recursive (STR): sap xep theo tung truc, cat thanh cac "lat"
  roi gom thanh nut, lap lai cho den goc.
- Cay tu phan PR: chia doi hinh vuong theo ca 2 truc thanh 4 o con (NW, NE, SW, SE) khi o chua qua
  QUAD_CAPACITY diem; diem nam o la.
- Tim kiem k-NN "tot nhat truoc" (best-first): hang doi uu tien theo khoang cach nho nhat tu diem truy van
  den MBR / o vuong; mot nut chi duoc mo khi no co the chua diem gan hon diem thu k hien tai.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

LEAF_SIZE = 16       # so diem toi da trong mot la / so con toi da cua mot nut cay R
QUAD_CAPACITY = 8    # so diem toi da trong mot o cua cay tu phan truoc khi chia
QUAD_MAX_DEPTH = 16


def _mindist(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Khoang cach Euclid nho nhat tu q den hinh hop [lo, hi] (0 neu q nam trong hop)."""
    d = np.maximum(lo - q, 0) + np.maximum(q - hi, 0)
    return float(np.sqrt((d * d).sum()))


def _knn(root, q: np.ndarray, Z: np.ndarray, k: int, children, box, points) -> tuple[np.ndarray, np.ndarray, int]:
    """Best-first k-NN dung chung cho ca hai cay. Tra ve (chi so diem, khoang cach, so nut da mo)."""
    heap = [(0.0, 0, root)]
    best: list[tuple[float, int]] = []     # max-heap (-d, idx)
    tie, visited = 1, 0
    while heap:
        dmin, _, node = heapq.heappop(heap)
        if len(best) == k and dmin > -best[0][0]:
            break
        visited += 1
        pts = points(node)
        if pts is not None:
            if len(pts):
                d = np.sqrt(((Z[pts] - q) ** 2).sum(axis=1))
                for di, pi in zip(d, pts):
                    if len(best) < k:
                        heapq.heappush(best, (-float(di), int(pi)))
                    elif di < -best[0][0]:
                        heapq.heapreplace(best, (-float(di), int(pi)))
            continue
        for ch in children(node):
            lo, hi = box(ch)
            heapq.heappush(heap, (_mindist(q, lo, hi), tie, ch))
            tie += 1
    best.sort(key=lambda t: -t[0])
    return np.array([i for _, i in best], int), np.array([-d for d, _ in best]), visited


class RTree:
    """Cay R nap lo STR. Nut = (lo, hi, con) voi con la mang chi so diem (la) hoac danh sach nut."""

    def __init__(self, Z: np.ndarray, leaf_size: int = LEAF_SIZE):
        self.Z = np.asarray(Z, np.float64)
        self.leaf_size = leaf_size
        leaves = [self._node(idx, None) for idx in self._str_groups(np.arange(len(self.Z)), self.Z)]
        level, self.height = leaves, 1
        while len(level) > 1:
            centers = np.array([(n[0] + n[1]) / 2 for n in level])
            groups = self._str_groups(np.arange(len(level)), centers)
            level = [self._node(None, [level[i] for i in g]) for g in groups]
            self.height += 1
        self.root = level[0]
        self.n_leaves = len(leaves)

    def _str_groups(self, idx: np.ndarray, P: np.ndarray, axis: int = 0) -> list[np.ndarray]:
        """Sort-Tile-Recursive: chia idx thanh cac nhom <= leaf_size, gan nhau trong khong gian."""
        if len(idx) <= self.leaf_size:
            return [idx]
        d = P.shape[1]
        n_groups = math.ceil(len(idx) / self.leaf_size)
        slabs = max(1, math.ceil(n_groups ** (1 / max(d - axis, 1))))
        order = idx[np.argsort(P[idx, axis], kind="stable")]
        if axis == d - 1 or slabs == 1:
            return [order[i:i + self.leaf_size] for i in range(0, len(order), self.leaf_size)]
        per_slab = math.ceil(len(order) / slabs)
        out = []
        for i in range(0, len(order), per_slab):
            out += self._str_groups(order[i:i + per_slab], P, axis + 1)
        return out

    def _node(self, pts, kids):
        if pts is not None:
            return (self.Z[pts].min(axis=0), self.Z[pts].max(axis=0), pts)
        return (np.min([k[0] for k in kids], axis=0), np.max([k[1] for k in kids], axis=0), kids)

    def knn(self, q: np.ndarray, k: int):
        return _knn(self.root, np.asarray(q, np.float64), self.Z, min(k, len(self.Z)),
                    children=lambda n: n[2], box=lambda n: (n[0], n[1]),
                    points=lambda n: n[2] if isinstance(n[2], np.ndarray) else None)


class QuadTree:
    """Cay tu phan vung diem (PR quadtree) tren 2 truc dau tien. Nut = [lo, hi, diem | None, con]."""

    def __init__(self, Z2: np.ndarray, capacity: int = QUAD_CAPACITY):
        self.Z = np.asarray(Z2, np.float64)[:, :2]
        self.capacity = capacity
        lo, hi = self.Z.min(axis=0), self.Z.max(axis=0)
        side = float((hi - lo).max()) or 1.0
        self.depth = 0
        self.n_leaves = 0
        self.root = self._build(np.arange(len(self.Z)), lo, lo + side, 0)

    def _build(self, idx: np.ndarray, lo: np.ndarray, hi: np.ndarray, depth: int):
        self.depth = max(self.depth, depth)
        if len(idx) <= self.capacity or depth >= QUAD_MAX_DEPTH:
            self.n_leaves += 1
            return [lo, hi, idx, []]
        mid = (lo + hi) / 2
        P = self.Z[idx]
        kids = []
        for right in (False, True):
            for top in (False, True):
                m = ((P[:, 0] >= mid[0]) == right) & ((P[:, 1] >= mid[1]) == top)
                clo = np.array([mid[0] if right else lo[0], mid[1] if top else lo[1]])
                chi = np.array([hi[0] if right else mid[0], hi[1] if top else mid[1]])
                kids.append(self._build(idx[m], clo, chi, depth + 1))
        return [lo, hi, None, kids]

    def knn(self, q: np.ndarray, k: int):
        return _knn(self.root, np.asarray(q, np.float64)[:2], self.Z, min(k, len(self.Z)),
                    children=lambda n: n[3], box=lambda n: (n[0], n[1]), points=lambda n: n[2])
