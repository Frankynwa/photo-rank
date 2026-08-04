"""Layer 2: 相似聚类 — 感知哈希 + BK-Tree + Union-Find"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image

from config import HAMMING_THRESHOLD
from core.models import SimilarityCluster as SimilarityClusterModel

logger = logging.getLogger(__name__)

# BK-Tree 小数据集 fallback 阈值
# 当照片数 < 此值时，Python 级别的暴力搜索因更低的常数开销反而更快
_BK_TREE_FALLBACK_SIZE = 500


@dataclass
class SimilarityCluster:
    cluster_id: int
    representative: str
    members: list[str]
    member_scores: dict[str, float]


class _BKTree:
    """BK-Tree：用于汉明距离空间的高效近邻搜索

    插入: O(log n) 均摊
    范围查询 (distance ≤ threshold): O(log n) 均摊
    """

    __slots__ = ("_hash", "_children")

    def __init__(self, hash_val: imagehash.ImageHash):
        self._hash = hash_val
        # dict[int, _BKTree] — 距离 -> 子节点
        self._children: dict[int, _BKTree] = {}

    @property
    def hash(self) -> imagehash.ImageHash:
        return self._hash

    def insert(self, other: imagehash.ImageHash) -> None:
        dist = int(self._hash - other)
        if dist == 0:
            # 哈希完全相同，挂在 distance=0 子树
            if 0 in self._children:
                self._children[0].insert(other)
            else:
                self._children[0] = _BKTree(other)
            return
        if dist in self._children:
            self._children[dist].insert(other)
        else:
            self._children[dist] = _BKTree(other)

    def query(self, target: imagehash.ImageHash, threshold: int) -> list[imagehash.ImageHash]:
        """返回树中所有与 target 距离 ≤ threshold 的哈希列表"""
        results: list[imagehash.ImageHash] = []
        self._query_recursive(target, threshold, results)
        return results

    def _query_recursive(
        self,
        target: imagehash.ImageHash,
        threshold: int,
        results: list[imagehash.ImageHash],
    ) -> None:
        dist = int(self._hash - target)
        if dist <= threshold:
            results.append(self._hash)
        # BK-Tree 三角不等式剪枝：只需搜索 [dist-threshold, dist+threshold] 范围内的子树
        lo, hi = dist - threshold, dist + threshold
        for d, child in self._children.items():
            if lo <= d <= hi:
                child._query_recursive(target, threshold, results)


def _brute_force_pairs(
    valid: list[str],
    hashes: dict[str, imagehash.ImageHash],
    threshold: int,
    uf: _UnionFind,
) -> None:
    """暴力 O(n²) 两两比较（小数据集 fallback）"""
    m = len(valid)
    for i in range(m):
        for j in range(i + 1, m):
            if hashes[valid[i]] - hashes[valid[j]] <= threshold:
                uf.union(i, j)


def _bktree_pairs(
    valid: list[str],
    hashes: dict[str, imagehash.ImageHash],
    threshold: int,
    uf: _UnionFind,
) -> None:
    """使用增量 BK-Tree 进行范围查询，避免 O(n²) 全量比较

    核心思路：逐张插入，每张照片插入前先在已有树中查询邻居，
    天然避免重复配对，无需 seen_pairs 集合。
    """
    tree = _BKTree(hashes[valid[0]])
    # 增量维护：哈希值 -> 已插入索引列表（直接用 ImageHash 对象作键）
    hash_to_indices: dict[imagehash.ImageHash, list[int]] = {}
    h0 = hashes[valid[0]]
    hash_to_indices[h0] = [0]

    for i in range(1, len(valid)):
        h = hashes[valid[i]]
        # 在已插入的 i 张照片中查找距离 ≤ threshold 的邻居
        for nh in tree.query(h, threshold):
            for j in hash_to_indices.get(nh, []):
                uf.union(j, i)
        # 查询完毕后再插入，避免自匹配
        tree.insert(h)
        if h in hash_to_indices:
            hash_to_indices[h].append(i)
        else:
            hash_to_indices[h] = [i]


class _UnionFind:
    """并查集：O(α(n)) 查找 + 合并"""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def compute_phash(image_path: str | Path) -> imagehash.ImageHash | None:
    """计算感知哈希"""
    try:
        img = Image.open(image_path).convert("RGB")
        return imagehash.phash(img)
    except Exception as e:
        logger.warning(f"哈希计算失败 {Path(image_path).name}: {e}")
        return None


def cluster_photos(
    image_paths: list[str | Path],
    sharpness_scores: dict[str, float],
    threshold: int = HAMMING_THRESHOLD,
) -> list[SimilarityClusterModel]:
    """将相似照片聚类（Union-Find + 感知哈希）"""
    paths = [str(p) for p in image_paths]

    # 计算哈希（保留失败的照片不丢失）
    hashes = {}
    failed = []
    for p in paths:
        h = compute_phash(p)
        if h is not None:
            hashes[p] = h
        else:
            failed.append(p)

    # Union-Find 聚类
    valid = list(hashes.keys())
    m = len(valid)
    uf = _UnionFind(m)

    if m < _BK_TREE_FALLBACK_SIZE:
        # 小数据集：暴力搜索开销更低
        _brute_force_pairs(valid, hashes, threshold, uf)
    else:
        # 大数据集：BK-Tree 范围查询 O(n log n)
        logger.info(f"照片数量 {m} ≥ {_BK_TREE_FALLBACK_SIZE}，启用 BK-Tree 索引")
        _bktree_pairs(valid, hashes, threshold, uf)

    # 分组
    groups = {}
    for i, p in enumerate(valid):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(p)

    # 构建结果：每组按清晰度取前 2 张作为代表
    # 同场景连拍的第 2 张也有机会进入审美层（L3）参与竞争，
    # 最终由融合分 + 多样性惩罚决定去留，避免"只留清晰但构图差"的一张。
    clusters: list[SimilarityClusterModel] = []
    for root, members in groups.items():
        ranked = sorted(members, key=lambda p: sharpness_scores.get(p, 0), reverse=True)
        best = ranked[0]
        representatives = ranked[:2]
        clusters.append(SimilarityClusterModel(
            cluster_id=len(clusters),
            representative=best,
            representatives=representatives,
            filename=Path(best).name,
            members=members,
            member_count=len(members),
            member_scores={p: sharpness_scores.get(p, 0) for p in members},
        ))

    # 失败的照片各成一个独立簇
    for p in failed:
        clusters.append(SimilarityClusterModel(
            cluster_id=len(clusters),
            representative=p,
            representatives=[p],
            filename=Path(p).name,
            members=[p],
            member_count=1,
            member_scores={p: sharpness_scores.get(p, 0)},
        ))
        logger.debug(f"哈希失败，独立簇: {Path(p).name}")

    return clusters
