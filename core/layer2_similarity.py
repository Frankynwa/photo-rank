"""Layer 2: 相似聚类 — 感知哈希 + BK-Tree + Union-Find"""
from __future__ import annotations

import logging
import re
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

# 文件名中的拍摄日期模式：MVIMG_20260707_064715.jpg → 20260707
# 仅匹配照片命名（视频帧无日期，不受影响）
_PHOTO_DATE_RE = re.compile(r"(20\d{6})_")


def photo_date_key(image_path: str | Path) -> str | None:
    """从文件名提取拍摄日期（YYYYMMDD）；无法解析返回 None（不参与同日期约束）"""
    m = _PHOTO_DATE_RE.search(Path(image_path).name)
    return m.group(1) if m else None


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
    same_date: bool = False,
    date_keys: dict[str, str | None] | None = None,
) -> None:
    """暴力 O(n²) 两两比较（小数据集 fallback）

    same_date=True 时仅同日期照片配对（照片榜防跨日期误合并）；
    视频榜帧无日期（date_keys 全 None）不受影响。
    """
    m = len(valid)
    for i in range(m):
        for j in range(i + 1, m):
            if same_date and date_keys and date_keys.get(valid[i]) != date_keys.get(valid[j]):
                continue  # 跨日期不视为同场景连拍
            if hashes[valid[i]] - hashes[valid[j]] <= threshold:
                uf.union(i, j)


def _bktree_pairs(
    valid: list[str],
    hashes: dict[str, imagehash.ImageHash],
    threshold: int,
    uf: _UnionFind,
    same_date: bool = False,
    date_keys: dict[str, str | None] | None = None,
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
                if same_date and date_keys and date_keys.get(valid[j]) != date_keys.get(valid[i]):
                    continue  # 跨日期不视为同场景连拍
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
    """计算感知哈希（复用解码缓存，HEIC/超大图不重复解原图）"""
    try:
        from core.image_cache import get_cached_path
        img = Image.open(get_cached_path(image_path)).convert("RGB")
        return imagehash.phash(img)
    except Exception as e:
        logger.warning(f"哈希计算失败 {Path(image_path).name}: {e}")
        return None


def cluster_photos(
    image_paths: list[str | Path],
    sharpness_scores: dict[str, float],
    threshold: int = HAMMING_THRESHOLD,
    precomputed_hashes: dict[str, imagehash.ImageHash] | None = None,
    same_date: bool = False,
) -> list[SimilarityClusterModel]:
    """将相似照片聚类（Union-Find + 感知哈希）

    precomputed_hashes: path -> ImageHash（来自 L1 流水阶段），传入后跳过
    哈希计算阶段直接聚类；为 None 或缺失时回退到内部串行计算。
    same_date: 仅同日期照片可合并（照片榜开启，防跨日期不同场景误合并）；
               视频榜帧无日期信息，保持 False 不启用。
    """
    paths = [str(p) for p in image_paths]
    date_keys = {p: photo_date_key(p) for p in paths} if same_date else None

    # 计算哈希（优先复用流水阶段结果，缺失的照片单独计入 failed）
    hashes = {}
    failed = []
    if precomputed_hashes:
        for p in paths:
            h = precomputed_hashes.get(p)
            if h is not None:
                hashes[p] = h
            else:
                failed.append(p)
    else:
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
        _brute_force_pairs(valid, hashes, threshold, uf, same_date=same_date, date_keys=date_keys)
    else:
        # 大数据集：BK-Tree 范围查询 O(n log n)
        logger.info(f"照片数量 {m} ≥ {_BK_TREE_FALLBACK_SIZE}，启用 BK-Tree 索引")
        _bktree_pairs(valid, hashes, threshold, uf, same_date=same_date, date_keys=date_keys)

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
