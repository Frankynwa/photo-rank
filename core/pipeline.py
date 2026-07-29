"""Pipeline — 整合所有层的完整流水线（RTX 5070 CUDA）"""
import gc
import torch
import asyncio
from pathlib import Path
from config import PHOTOS_DIR, OUTPUT_DIR, CATEGORIES, PLATFORMS, TOP_K_PER_CATEGORY, FINAL_OUTPUT_COUNT


def run_pipeline(photos_dir: str | Path = None, platform: str = "xiaohongshu") -> dict:
    """运行完整流水线"""
    from core.layer1_objective import batch_analyze
    from core.layer2_similarity import cluster_photos
    from core.layer3_aesthetic import score_photos
    from core.layer4_face import analyze_faces
    
    photos_dir = Path(photos_dir or PHOTOS_DIR)
    output_dir = OUTPUT_DIR / platform
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有照片
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.heic", "*.HEIC"]
    photos = []
    for ext in extensions:
        photos.extend(photos_dir.glob(ext))
    
    if not photos:
        print(f"未找到照片: {photos_dir}")
        return {"error": "未找到照片"}
    
    print(f"=" * 60)
    print(f"PhotoRank 旅行照片 AI 智能筛选")
    print(f"=" * 60)
    print(f"照片目录: {photos_dir}")
    print(f"目标平台: {PLATFORMS[platform]['name']}")
    print(f"照片数量: {len(photos)}")
    print()
    
    # ─────────────────────────────────────────────────────────
    # Layer 1: 客观指标（CPU）
    # ─────────────────────────────────────────────────────────
    print("[Layer 1] 客观指标分析（清晰度/曝光/噪声）...")
    l1_results = batch_analyze(photos)
    kept = [r for r in l1_results if not r["rejected"]]
    rejected = [r for r in l1_results if r["rejected"]]
    
    print(f"  ✅ 保留: {len(kept)} 张")
    print(f"  ❌ 淘汰: {len(rejected)} 张")
    for r in rejected:
        print(f"    - {r['filename']}: {r['reject_reason']}")
    print()
    
    # ─────────────────────────────────────────────────────────
    # Layer 2: 相似聚类（pHash，CPU）
    # ─────────────────────────────────────────────────────────
    print("[Layer 2] 相似照片聚类（pHash）...")
    sharpness_scores = {r["path"]: r["sharpness"] for r in kept}
    clusters = cluster_photos([r["path"] for r in kept], sharpness_scores)
    
    representatives = []
    for c in clusters:
        representatives.append(c["representative"])
    
    merged_count = sum(c["member_count"] - 1 for c in clusters if c["member_count"] > 1)
    print(f"  聚类结果: {len(clusters)} 组")
    print(f"  合并相似: {merged_count} 张")
    print(f"  保留代表: {len(representatives)} 张")
    print()
    
    # ─────────────────────────────────────────────────────────
    # Layer 3: 审美评分（NIMA-VGG16，RTX 5070 CUDA）
    # ─────────────────────────────────────────────────────────
    print("[Layer 3] 审美评分（NIMA-VGG16，RTX 5070）...")
    aes_results = score_photos(representatives)
    aes_results.sort(key=lambda x: x["overall_score"], reverse=True)
    
    print(f"  Top 10:")
    for i, r in enumerate(aes_results[:10]):
        print(f"    {i+1}. {r['filename']}: aesthetic={r['aesthetic_score']}, technical={r['technical_score']}, overall={r['overall_score']}")
    print()
    
    # ─────────────────────────────────────────────────────────
    # Layer 4: 人脸质量（MediaPipe，CPU）
    # ─────────────────────────────────────────────────────────
    print("[Layer 4] 人脸质量分析（MediaPipe）...")
    face_results = analyze_faces([r["path"] for r in aes_results])
    
    face_photos = [r for r in face_results if r["has_face"]]
    blink_photos = [r for r in face_results if r["blink_detected"]]
    
    print(f"  含人脸: {len(face_photos)} 张")
    print(f"  检测到闭眼: {len(blink_photos)} 张")
    print()
    
    # 调整排序：闭眼扣分
    face_scores = {r["path"]: r["overall_face_score"] for r in face_results}
    for r in aes_results:
        face_score = face_scores.get(r["path"], 0)
        r["final_score"] = r["overall_score"] + face_score * 0.1
    
    aes_results.sort(key=lambda x: x["final_score"], reverse=True)
    
    # ─────────────────────────────────────────────────────────
    # Layer 5: 深度分析（Qwen-VL API，仅 Top N）
    # ─────────────────────────────────────────────────────────
    print("[Layer 5] 深度分析（Qwen-VL API）...")
    from core.layer5_analysis import deep_analyze
    
    top_n = min(FINAL_OUTPUT_COUNT, len(aes_results))
    top_photos = aes_results[:top_n]
    
    # 异步运行深度分析
    async def run_deep_analysis():
        results = []
        for i, r in enumerate(top_photos):
            print(f"  深度分析进度: {i + 1}/{top_n} - {r['filename']}")
            try:
                analysis = await deep_analyze(r["path"], platform)
                results.append(analysis)
            except Exception as e:
                print(f"    分析失败: {e}")
                results.append({
                    "path": r["path"],
                    "filename": r["filename"],
                    "classification": "未知",
                    "composition": "分析失败",
                    "improvement": "分析失败",
                    "emotion": {"primary_emotion": "未知", "emotion_intensity": 5},
                    "copywriting": "分析失败",
                    "platform": platform,
                })
        return results
    
    deep_results = asyncio.run(run_deep_analysis())
    print()
    
    # ─────────────────────────────────────────────────────────
    # 保存结果
    # ─────────────────────────────────────────────────────────
    print("[输出] 保存结果...")
    
    # 合并评分和深度分析
    import json
    final_results = []
    for i, r in enumerate(top_photos):
        deep = deep_results[i] if i < len(deep_results) else {}
        final_results.append({
            **r,
            "deep_analysis": deep,
        })
    
    # 保存完整结果
    ranking_file = output_dir / "ranking.json"
    with open(ranking_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"  排名文件: {ranking_file}")
    
    # 复制照片
    import shutil
    for i, r in enumerate(top_photos):
        src = Path(r["path"])
        dst = output_dir / f"{i+1:02d}_{src.name}"
        shutil.copy2(src, dst)
    
    print(f"  Top {top_n} 照片已复制到: {output_dir}")
    print()
    
    # ─────────────────────────────────────────────────────────
    # 汇总
    # ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("筛选完成！")
    print("=" * 60)
    print(f"输入: {len(photos)} 张照片")
    print(f"Layer 1 淘汰: {len(rejected)} 张（模糊/过曝/欠曝）")
    print(f"Layer 2 合并: {merged_count} 张（相似照片）")
    print(f"Layer 3 排序: {len(aes_results)} 张（审美评分）")
    print(f"Layer 4 调整: {len(blink_photos)} 张（闭眼扣分）")
    print(f"Layer 5 分析: {len(deep_results)} 张（深度分析）")
    print(f"输出: Top {top_n} 张 → {output_dir}")
    print()
    print("Top 5 最终排名:")
    for i, r in enumerate(top_photos[:5]):
        print(f"  {i+1}. {r['filename']}: final_score={r['final_score']:.4f}")
    
    return {
        "total": len(photos),
        "rejected": len(rejected),
        "merged": merged_count,
        "ranked": len(aes_results),
        "output_count": top_n,
        "output_dir": str(output_dir),
        "top_photos": top_photos,
        "deep_results": deep_results,
    }


if __name__ == "__main__":
    result = run_pipeline()
