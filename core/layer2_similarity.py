"""Layer 2: 相似聚类 — 感知哈希（零显存占用）"""
import hashlib
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
import imagehash


@dataclass
class SimilarityCluster:
    """相似照片聚类结果"""
    cluster_id: int
    representative: str      # 代表照片（清晰度最高的）
    members: list[str]       # 所有成员
    member_scores: dict[str, float]  # 每张照片的清晰度分


def compute_phash(image_path: str | Path) -> imagehash.ImageHash:
    """计算感知哈希"""
    try:
        img = Image.open(image_path).convert("RGB")
        return imagehash.phash(img)
    except Exception as e:
        print(f"哈希计算失败 {image_path}: {e}")
        return None


def cluster_photos(
    image_paths: list[str | Path],
    sharpness_scores: dict[str, float],
    threshold: int = 10,  # 汉明距离阈值（越小越严格）
) -> list[dict]:
    """将相似照片聚类（基于感知哈希）"""
    # 计算所有照片的哈希
    hashes = {}
    for path in image_paths:
        h = compute_phash(path)
        if h is not None:
            hashes[str(path)] = h
    
    # 聚类
    paths = list(hashes.keys())
    n = len(paths)
    visited = set()
    clusters = []
    
    for i in range(n):
        if paths[i] in visited:
            continue
        
        cluster_members = [paths[i]]
        visited.add(paths[i])
        
        for j in range(i + 1, n):
            if paths[j] in visited:
                continue
            
            # 汉明距离
            distance = hashes[paths[i]] - hashes[paths[j]]
            if distance <= threshold:
                cluster_members.append(paths[j])
                visited.add(paths[j])
        
        # 选择代表照片（清晰度最高的）
        best_path = max(
            cluster_members,
            key=lambda p: sharpness_scores.get(p, 0)
        )
        
        clusters.append({
            "cluster_id": len(clusters),
            "representative": best_path,
            "filename": Path(best_path).name,
            "members": cluster_members,
            "member_count": len(cluster_members),
            "member_scores": {p: sharpness_scores.get(p, 0) for p in cluster_members},
        })
    
    return clusters
