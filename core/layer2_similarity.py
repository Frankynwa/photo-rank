"""Layer 2: 相似聚类 — 感知哈希 + Union-Find"""
import logging
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
import imagehash
from config import LAPLACIAN_THRESHOLD

logger = logging.getLogger(__name__)

# 汉明距离阈值（与 config.py SIMILARITY_THRESHOLD 的语义对应）
HAMMING_THRESHOLD = 10


@dataclass
class SimilarityCluster:
    cluster_id: int
    representative: str
    members: list[str]
    member_scores: dict[str, float]


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
) -> list[dict]:
    """将相似照片聚类（Union-Find + 感知哈希）"""
    paths = [str(p) for p in image_paths]
    n = len(paths)

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

    for i in range(m):
        for j in range(i + 1, m):
            distance = hashes[valid[i]] - hashes[valid[j]]
            if distance <= threshold:
                uf.union(i, j)

    # 分组
    groups = {}
    for i, p in enumerate(valid):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(p)

    # 构建结果
    clusters = []
    for root, members in groups.items():
        best = max(members, key=lambda p: sharpness_scores.get(p, 0))
        clusters.append({
            "cluster_id": len(clusters),
            "representative": best,
            "filename": Path(best).name,
            "members": members,
            "member_count": len(members),
            "member_scores": {p: sharpness_scores.get(p, 0) for p in members},
        })

    # 失败的照片各成一个独立簇
    for p in failed:
        clusters.append({
            "cluster_id": len(clusters),
            "representative": p,
            "filename": Path(p).name,
            "members": [p],
            "member_count": 1,
            "member_scores": {p: sharpness_scores.get(p, 0)},
        })
        logger.debug(f"哈希失败，独立簇: {Path(p).name}")

    return clusters
