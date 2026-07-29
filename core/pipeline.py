"""Pipeline — 整合所有层的完整流水线（RTX 5070 CUDA）"""
import asyncio
import json
import shutil
import logging
from pathlib import Path
from config import PHOTOS_DIR, OUTPUT_DIR, FINAL_OUTPUT_COUNT

logger = logging.getLogger(__name__)


def _emit(progress_callback, event: str, **data):
    """安全调用进度回调"""
    if progress_callback:
        try:
            progress_callback(event, data)
        except Exception:
            pass


async def run_deep_analysis_async(
    top_photos: list[dict], platform: str, progress_callback=None
) -> list[dict]:
    """异步深度分析（Layer 5 并行）"""
    from core.layer5_analysis import deep_analyze

    tasks = [deep_analyze(r["path"], platform) for r in top_photos]
    total = len(tasks)
    results = []

    for i, coro in enumerate(tasks):
        _emit(progress_callback, "layer_progress", layer=5, layer_name="深度分析",
              current=i + 1, total=total,
              message=f"分析中: {Path(top_photos[i]['filename']).name}")
        try:
            result = await coro
            results.append(result)
        except Exception as e:
            logger.error(f"深度分析失败 {top_photos[i]['filename']}: {e}")
            results.append({
                "path": top_photos[i]["path"],
                "filename": top_photos[i]["filename"],
                "classification": "分析失败", "composition": "分析失败",
                "improvement": "分析失败",
                "emotion": {"primary_emotion": "未知", "emotion_intensity": 5},
                "copywriting": "分析失败", "platform": platform,
            })

    return results


def run_pipeline(
    photos_dir: str | Path = None,
    platform: str = "xiaohongshu",
    progress_callback=None,
) -> dict:
    """运行完整流水线（支持进度回调）"""
    from core.layer1_objective import batch_analyze
    from core.layer2_similarity import cluster_photos
    from core.layer3_aesthetic import score_photos
    from core.layer4_face import analyze_faces

    photos_dir = Path(photos_dir or PHOTOS_DIR)

    # 收集所有照片
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.heic", "*.HEIC", "*.JPG", "*.PNG"]
    photos = []
    for ext in extensions:
        photos.extend(photos_dir.glob(ext))

    if not photos:
        logger.warning(f"未找到照片: {photos_dir}")
        return {"error": "未找到照片", "total": 0}

    # 去重：同名文件不去重
    photos = sorted(set(photos), key=lambda p: p.name)

    logger.info(f"照片目录: {photos_dir} | 数量: {len(photos)} | 平台: {platform}")
    _emit(progress_callback, "layer_start", layer=0, layer_name="开始", total=len(photos))

    skipped_layers = []

    # ── Layer 1: 客观指标 ──
    _emit(progress_callback, "layer_start", layer=1, layer_name="客观指标")
    try:
        l1_results = batch_analyze(photos)
        kept = [r for r in l1_results if not r["rejected"]]
        rejected = [r for r in l1_results if r["rejected"]]
        logger.info(f"[L1] 保留: {len(kept)}, 淘汰: {len(rejected)}")
        for r in rejected:
            logger.debug(f"  淘汰: {r['filename']} ({r['reject_reason']})")
    except Exception as e:
        logger.error(f"[L1] 失败，跳过: {e}")
        kept = [{"path": str(p), "filename": Path(p).name, "rejected": False,
                 "sharpness": 0, "reject_reason": ""} for p in photos]
        rejected = []
        skipped_layers.append("Layer 1")
    _emit(progress_callback, "layer_end", layer=1, kept=len(kept), rejected=len(rejected))

    # ── Layer 2: 相似聚类 ──
    _emit(progress_callback, "layer_start", layer=2, layer_name="相似聚类")
    try:
        sharpness_scores = {r["path"]: r.get("sharpness", 0) for r in kept}
        clusters = cluster_photos([r["path"] for r in kept], sharpness_scores)
        representatives = [c["representative"] for c in clusters]
        merged = sum(c["member_count"] - 1 for c in clusters if c["member_count"] > 1)
        logger.info(f"[L2] 聚类: {len(clusters)} 组, 合并: {merged}")
    except Exception as e:
        logger.error(f"[L2] 失败，跳过: {e}")
        representatives = [r["path"] for r in kept]
        merged = 0
        skipped_layers.append("Layer 2")
    _emit(progress_callback, "layer_end", layer=2, groups=len(representatives), merged=merged)

    # ── Layer 3: 审美评分 ──
    _emit(progress_callback, "layer_start", layer=3, layer_name="审美评分")
    try:
        aes_results = score_photos(representatives)
        aes_results.sort(key=lambda x: x["overall_score"], reverse=True)
    except Exception as e:
        logger.error(f"[L3] 失败，跳过: {e}")
        aes_results = [{"path": p, "filename": Path(p).name, "aesthetic_score": 0,
                        "technical_score": 0, "overall_score": 0, "final_score": 0}
                       for p in representatives]
        skipped_layers.append("Layer 3")
    _emit(progress_callback, "layer_end", layer=3, scored=len(aes_results))

    # ── Layer 4: 人脸质量 ──
    _emit(progress_callback, "layer_start", layer=4, layer_name="人脸质量")
    try:
        face_results = analyze_faces([r["path"] for r in aes_results])
        face_scores = {r["path"]: r.get("overall_face_score", 0) for r in face_results}
        face_count = sum(1 for r in face_results if r.get("has_face"))
        blink_count = sum(1 for r in face_results if r.get("blink_detected"))
        logger.info(f"[L4] 人脸: {face_count}, 闭眼: {blink_count}")
    except Exception as e:
        logger.error(f"[L4] 失败，跳过: {e}")
        face_scores = {}
        skipped_layers.append("Layer 4")
    _emit(progress_callback, "layer_end", layer=4, faces=len(face_scores))

    # 综合排序
    for r in aes_results:
        r["final_score"] = r["overall_score"] + face_scores.get(r["path"], 0) * 0.1
    aes_results.sort(key=lambda x: x["final_score"], reverse=True)

    # ── Layer 5: 深度分析 ──
    top_n = min(FINAL_OUTPUT_COUNT, len(aes_results))
    top_photos = aes_results[:top_n]

    _emit(progress_callback, "layer_start", layer=5, layer_name="深度分析", total=top_n)
    try:
        deep_results = asyncio.run(
            run_deep_analysis_async(top_photos, platform, progress_callback)
        )
    except Exception as e:
        logger.error(f"[L5] 失败，跳过: {e}")
        deep_results = [{"path": r["path"], "filename": r["filename"],
                         "classification": "跳过", "composition": str(e),
                         "improvement": str(e), "emotion": {}, "copywriting": "跳过",
                         "platform": platform} for r in top_photos]
        skipped_layers.append("Layer 5")
    _emit(progress_callback, "layer_end", layer=5, done=len(deep_results))

    # ── 保存结果 ──
    output_dir = OUTPUT_DIR / platform
    output_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧输出
    for old in output_dir.glob("*"):
        if old.is_file():
            old.unlink()

    # 合并结果并保存
    final_results = []
    for i, r in enumerate(top_photos):
        deep = deep_results[i] if i < len(deep_results) else {}
        final_results.append({**r, "deep_analysis": deep})

    ranking_file = output_dir / "ranking.json"
    with open(ranking_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    # 复制照片
    for i, r in enumerate(top_photos):
        src = Path(r["path"])
        dst = output_dir / f"{i+1:02d}_{src.name}"
        shutil.copy2(src, dst)

    logger.info(f"输出: Top {top_n} → {output_dir}")

    # ── 汇总 ──
    summary = {
        "total": len(photos),
        "rejected": len(rejected),
        "merged": merged,
        "ranked": len(aes_results),
        "output_count": top_n,
        "output_dir": str(output_dir),
        "skipped_layers": skipped_layers,
    }

    logger.info(f"完成: {summary}")
    return summary
