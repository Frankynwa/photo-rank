"""Pipeline — 整合所有层的完整流水线（支持 GPU/CPU 自动切换 + 断点续传）"""
import asyncio
import json
import logging
import shutil
from pathlib import Path

import imagehash

from config import (
    DIVERSITY_DISTANCE_THRESHOLD,
    DIVERSITY_PENALTY,
    FINAL_OUTPUT_COUNT,
    OUTPUT_DIR,
    PHOTOS_DIR,
    PLATFORMS,
)
from core.models import (
    FinalResult,
    Layer1Result,
    Layer3Result,
    Layer5Result,
    PipelineResult,
    SimilarityCluster,
)

logger = logging.getLogger(__name__)


def _emit(progress_callback, event: str, **data):
    """安全调用进度回调"""
    if progress_callback:
        try:
            progress_callback(event, data)
        except Exception:
            pass


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint.json"


def _save_checkpoint(output_dir: Path, data: dict) -> None:
    """保存断点 checkpoint"""
    output_dir.mkdir(parents=True, exist_ok=True)
    cp = _checkpoint_path(output_dir)
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"[Checkpoint] 已保存: layer={data.get('completed_layer')}")


def _load_checkpoint(output_dir: Path) -> dict | None:
    """加载断点 checkpoint，不存在则返回 None"""
    cp = _checkpoint_path(output_dir)
    if cp.exists():
        try:
            with open(cp, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"[Checkpoint] 已加载: layer={data.get('completed_layer')}")
            return data
        except Exception as e:
            logger.warning(f"[Checkpoint] 加载失败，将重新开始: {e}")
    return None


def _remove_checkpoint(output_dir: Path) -> None:
    """删除断点 checkpoint"""
    cp = _checkpoint_path(output_dir)
    if cp.exists():
        cp.unlink()
        logger.info("[Checkpoint] 已删除")


async def run_deep_analysis_async(
    top_photos: list[Layer3Result], platform: str, progress_callback=None
) -> list[Layer5Result]:
    """异步深度分析（Layer 5 并行）"""
    from core.layer5_analysis import deep_analyze

    tasks = [deep_analyze(r.path, platform) for r in top_photos]
    total = len(tasks)
    results: list[Layer5Result] = []

    for i, coro in enumerate(tasks):
        _emit(progress_callback, "layer_progress", layer=5, layer_name="深度分析",
              current=i + 1, total=total,
              message=f"分析中: {top_photos[i].filename}")
        try:
            result = await coro
            results.append(result)
        except Exception as e:
            logger.error(f"深度分析失败 {top_photos[i].filename}: {e}")
            results.append(Layer5Result(
                path=top_photos[i].path,
                filename=top_photos[i].filename,
                classification="分析失败", composition="分析失败",
                improvement="分析失败", copywriting="分析失败",
                platform=platform,
            ))

    return results


def run_pipeline(
    photos_dir: str | Path = None,
    platform: str = "xiaohongshu",
    progress_callback=None,
    force: bool = False,
) -> dict:
    """运行完整流水线（支持进度回调 + 断点续传）

    Args:
        photos_dir: 照片目录
        platform: 目标平台
        progress_callback: 进度回调函数
        force: 为 True 时忽略 checkpoint 从头开始
    """
    from core.layer1_objective import batch_analyze
    from core.layer2_similarity import cluster_photos
    from core.layer3_aesthetic import score_photos
    from core.layer4_face import analyze_faces

    photos_dir = Path(photos_dir or PHOTOS_DIR)
    output_dir = OUTPUT_DIR / platform

    # 收集所有照片
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.heic", "*.HEIC", "*.JPG", "*.PNG"]
    photos = []
    for ext in extensions:
        photos.extend(photos_dir.glob(ext))

    if not photos:
        logger.warning(f"未找到照片: {photos_dir}")
        return {"error": "未找到照片", "total": 0}

    photos = sorted(set(photos), key=lambda p: p.name)

    logger.info(f"照片目录: {photos_dir} | 数量: {len(photos)} | 平台: {platform}")

    # ── 检查 checkpoint ──
    checkpoint = None if force else _load_checkpoint(output_dir)
    start_layer = 1
    if checkpoint:
        start_layer = checkpoint["completed_layer"] + 1
        logger.info(f"从 Layer {start_layer} 继续（跳过已完成 Layer {checkpoint['completed_layer']}）")

    # 仅在全新启动时清理输出目录
    if not checkpoint:
        output_dir.mkdir(parents=True, exist_ok=True)
        for old in output_dir.glob("*"):
            if old.is_file() and old.name != "checkpoint.json":
                old.unlink()

    _emit(progress_callback, "layer_start", layer=0, layer_name="开始", total=len(photos))

    skipped_layers: list[str] = []

    # ── Layer 1: 客观指标 ──
    if checkpoint and checkpoint["completed_layer"] >= 1:
        kept = [Layer1Result(**r) for r in checkpoint["kept"]]
        rejected = [Layer1Result(**r) for r in checkpoint["rejected"]]
        skipped_layers = checkpoint.get("skipped_layers", [])
        logger.info(f"[L1] 从 checkpoint 恢复: 保留 {len(kept)}, 淘汰 {len(rejected)}")
    else:
        _emit(progress_callback, "layer_start", layer=1, layer_name="客观指标")
        try:
            l1_results = batch_analyze(photos)
            kept = [r for r in l1_results if not r.rejected]
            rejected = [r for r in l1_results if r.rejected]
            logger.info(f"[L1] 保留: {len(kept)}, 淘汰: {len(rejected)}")
            for r in rejected:
                logger.debug(f"  淘汰: {r.filename} ({r.reject_reason})")
        except Exception as e:
            logger.error(f"[L1] 失败，跳过: {e}")
            kept = [Layer1Result(path=str(p), filename=p.name) for p in photos]
            rejected = []
            skipped_layers.append("Layer 1")
        _emit(progress_callback, "layer_end", layer=1, kept=len(kept), rejected=len(rejected))
        _save_checkpoint(output_dir, {
            "completed_layer": 1,
            "kept": [r.model_dump() for r in kept],
            "rejected": [r.model_dump() for r in rejected],
            "skipped_layers": skipped_layers,
        })

    # 建立 Layer 1 客观分映射（0-1 范围）
    l1_scores = {r.path: r.overall for r in kept}

    # ── Layer 2: 相似聚类 ──
    if checkpoint and checkpoint["completed_layer"] >= 2:
        clusters = [SimilarityCluster(**c) for c in checkpoint["clusters"]]
        representatives = [c.representative for c in clusters]
        merged = sum(c.member_count - 1 for c in clusters if c.member_count > 1)
        skipped_layers = checkpoint.get("skipped_layers", skipped_layers)
        logger.info(f"[L2] 从 checkpoint 恢复: {len(clusters)} 组")
    else:
        _emit(progress_callback, "layer_start", layer=2, layer_name="相似聚类")
        try:
            sharpness_scores = {r.path: r.sharpness for r in kept}
            clusters = cluster_photos([r.path for r in kept], sharpness_scores)
            representatives = [c.representative for c in clusters]
            merged = sum(c.member_count - 1 for c in clusters if c.member_count > 1)
            logger.info(f"[L2] 聚类: {len(clusters)} 组, 合并: {merged}")
        except Exception as e:
            logger.error(f"[L2] 失败，跳过: {e}")
            representatives = [r.path for r in kept]
            merged = 0
            skipped_layers.append("Layer 2")
        _emit(progress_callback, "layer_end", layer=2, groups=len(representatives), merged=merged)
        _save_checkpoint(output_dir, {
            "completed_layer": 2,
            "clusters": [c.model_dump() for c in clusters],
            "skipped_layers": skipped_layers,
        })

    # ── Layer 3: 审美评分 ──
    if checkpoint and checkpoint["completed_layer"] >= 3:
        aes_results = [Layer3Result(**r) for r in checkpoint["aes_results"]]
        skipped_layers = checkpoint.get("skipped_layers", skipped_layers)
        logger.info(f"[L3] 从 checkpoint 恢复: {len(aes_results)} 张")
    else:
        _emit(progress_callback, "layer_start", layer=3, layer_name="审美评分")
        try:
            aes_results = score_photos(representatives)
            aes_results.sort(key=lambda x: x.overall_score, reverse=True)
        except Exception as e:
            logger.error(f"[L3] 失败，跳过: {e}")
            aes_results = [Layer3Result(path=p, filename=Path(p).name) for p in representatives]
            skipped_layers.append("Layer 3")
        _emit(progress_callback, "layer_end", layer=3, scored=len(aes_results))
        _save_checkpoint(output_dir, {
            "completed_layer": 3,
            "aes_results": [r.model_dump() for r in aes_results],
            "skipped_layers": skipped_layers,
        })

    # ── Layer 4: 人脸质量 ──
    if checkpoint and checkpoint["completed_layer"] >= 4:
        face_scores = checkpoint["face_scores"]
        skipped_layers = checkpoint.get("skipped_layers", skipped_layers)
        logger.info(f"[L4] 从 checkpoint 恢复: {len(face_scores)} 张")
    else:
        _emit(progress_callback, "layer_start", layer=4, layer_name="人脸质量")
        try:
            face_results = analyze_faces([r.path for r in aes_results])
            face_scores = {r.path: r.overall_face_score for r in face_results}
            face_count = sum(1 for r in face_results if r.has_face)
            blink_count = sum(1 for r in face_results if r.blink_detected)
            logger.info(f"[L4] 人脸: {face_count}, 闭眼: {blink_count}")
        except Exception as e:
            logger.error(f"[L4] 失败，跳过: {e}")
            face_scores = {}
            skipped_layers.append("Layer 4")
        _emit(progress_callback, "layer_end", layer=4, faces=len(face_scores))
        _save_checkpoint(output_dir, {
            "completed_layer": 4,
            "face_scores": face_scores,
            "skipped_layers": skipped_layers,
        })

    # 综合排序：客观分×0.2 + 审美分×0.6 + 人脸分×0.2
    # Layer 1 overall: 0-1 范围
    # Layer 3 overall_score: 0-10 范围（需归一化到 0-1）
    # Layer 4 overall_face_score: 0-1 范围
    aes_results.sort(
        key=lambda x: (
            l1_scores.get(x.path, 0.5) * 0.2 +  # 客观分（默认 0.5）
            (x.overall_score / 10.0) * 0.6 +     # 审美分（归一化到 0-1）
            face_scores.get(x.path, 0) * 0.2     # 人脸分
        ),
        reverse=True,
    )

    # ── 多样性惩罚：对相似照片扣分，避免同场景占据多个名额 ──

    def _compute_final_score(r) -> float:
        """计算综合得分（与排序逻辑一致）"""
        return (
            l1_scores.get(r.path, 0.5) * 0.2
            + (r.overall_score / 10.0) * 0.6
            + face_scores.get(r.path, 0) * 0.2
        )

    # 为所有照片计算 pHash
    from core.layer2_similarity import compute_phash
    hash_cache: dict[str, imagehash.ImageHash | None] = {}
    for r in aes_results:
        hash_cache[r.path] = compute_phash(r.path)

    # 计算原始综合分
    score_map: dict[str, float] = {r.path: _compute_final_score(r) for r in aes_results}

    # 找出所有汉明距离 < 阈值的相似对，对每对中排名靠后的扣分
    penalized: set[str] = set()
    for i in range(len(aes_results)):
        hi = hash_cache[aes_results[i].path]
        if hi is None:
            continue
        for j in range(i + 1, len(aes_results)):
            hj = hash_cache[aes_results[j].path]
            if hj is None:
                continue
            distance = abs(int(hi - hj))
            if distance < DIVERSITY_DISTANCE_THRESHOLD:
                penalize_path = aes_results[j].path  # j 排名靠后
                if penalize_path not in penalized:
                    score_map[penalize_path] *= (1.0 - DIVERSITY_PENALTY)
                    penalized.add(penalize_path)
                    logger.info(
                        f"[多样性] 对 {Path(penalize_path).name} 施加 "
                        f"{DIVERSITY_PENALTY*100:.0f}% 扣分"
                        f"（与 {Path(aes_results[i].path).name} 相似，距离={distance}）"
                    )

    # 按扣分后的分数重新排序
    aes_results.sort(key=lambda x: score_map[x.path], reverse=True)

    # ── LLM 终审：从 Top 20 中选出最适合平台的 Top 12 ──
    top_n = min(FINAL_OUTPUT_COUNT, len(aes_results))
    CANDIDATE_COUNT = 20  # 候选数量
    candidates = aes_results[:min(CANDIDATE_COUNT, len(aes_results))]

    # 获取平台权重
    platform_config = PLATFORMS.get(platform, PLATFORMS["xiaohongshu"])
    weights = platform_config.get("weights", {
        "composition": 0.35, "color": 0.30, "lighting": 0.20, "face": 0.15
    })

    # 计算加权分数
    def calc_weighted_score(r):
        return (
            r.composition_score * weights.get("composition", 0.35)
            + r.color_score * weights.get("color", 0.30)
            + r.lighting_score * weights.get("lighting", 0.20)
            + face_scores.get(r.path, 0) * 10 * weights.get("face", 0.15)  # 人脸分 0-1 → 0-10
        )

    # 按加权分数排序
    candidates.sort(key=calc_weighted_score, reverse=True)
    top_photos = candidates[:top_n]

    logger.info(f"[终审] 平台: {platform_config['name']}, 候选: {len(candidates)}, 入选: {len(top_photos)}")
    logger.info(
        f"[终审] 权重: 构图{weights.get('composition', 0.35)} "
        f"色彩{weights.get('color', 0.30)} "
        f"光影{weights.get('lighting', 0.20)} "
        f"人脸{weights.get('face', 0.15)}"
    )

    # ── Layer 5: 深度分析 ──
    if checkpoint and checkpoint["completed_layer"] >= 5:
        deep_results = [Layer5Result(**r) for r in checkpoint["deep_results"]]
        skipped_layers = checkpoint.get("skipped_layers", skipped_layers)
        logger.info(f"[L5] 从 checkpoint 恢复: {len(deep_results)} 张")
    else:
        _emit(progress_callback, "layer_start", layer=5, layer_name="深度分析", total=top_n)
        try:
            deep_results = asyncio.run(
                run_deep_analysis_async(top_photos, platform, progress_callback)
            )
        except Exception as e:
            logger.error(f"[L5] 失败，跳过: {e}")
            deep_results = [Layer5Result(
                path=r.path, filename=r.filename,
                classification="跳过", composition=str(e),
                improvement=str(e), copywriting="跳过",
                platform=platform,
            ) for r in top_photos]
            skipped_layers.append("Layer 5")
        _emit(progress_callback, "layer_end", layer=5, done=len(deep_results))
        _save_checkpoint(output_dir, {
            "completed_layer": 5,
            "deep_results": [r.model_dump() for r in deep_results],
            "skipped_layers": skipped_layers,
        })

    # ── 保存结果 ──
    output_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧输出（保留 checkpoint 在最终清理时删除）
    for old in output_dir.glob("*"):
        if old.is_file() and old.name != "checkpoint.json":
            old.unlink()

    # 合并结果并保存
    final_results: list[FinalResult] = []
    for i, r in enumerate(top_photos):
        deep = deep_results[i] if i < len(deep_results) else Layer5Result(
            path=r.path, filename=r.filename,
        )
        final_results.append(FinalResult(
            path=r.path,
            filename=r.filename,
            aesthetic_score=r.aesthetic_score,
            overall_score=r.overall_score,
            final_score=score_map[r.path] * 10,  # 含多样性惩罚，放大到 0-10 范围便于展示
            deep_analysis=deep,
        ))

    ranking_file = output_dir / "ranking.json"
    with open(ranking_file, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in final_results], f, ensure_ascii=False, indent=2)

    # 复制照片
    for i, r in enumerate(top_photos):
        src = Path(r.path)
        dst = output_dir / f"{i+1:02d}_{src.name}"
        shutil.copy2(src, dst)

    logger.info(f"输出: Top {top_n} → {output_dir}")

    # ── 删除 checkpoint（所有层完成） ──
    _remove_checkpoint(output_dir)

    # ── 汇总 ──
    summary = PipelineResult(
        total=len(photos),
        rejected=len(rejected),
        merged=merged,
        ranked=len(aes_results),
        output_count=top_n,
        output_dir=str(output_dir),
        skipped_layers=skipped_layers,
        results=final_results,
    )

    logger.info(f"完成: total={summary.total}, rejected={summary.rejected}, "
                f"merged={summary.merged}, ranked={summary.ranked}, output={summary.output_count}")
    return summary.model_dump()
