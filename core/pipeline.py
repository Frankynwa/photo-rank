"""Pipeline — 整合所有层的完整流水线（支持 GPU/CPU 自动切换 + 断点续传）"""
import json
import logging
import shutil
from pathlib import Path

import imagehash

from config import (
    DIVERSITY_DISTANCE_THRESHOLD,
    DIVERSITY_PENALTY,
    FINAL_OUTPUT_COUNT,
    FUSION_WEIGHTS,
    OUTPUT_DIR,
    PHOTOS_DIR,
    PLATFORMS,
    VIDEO_FRAMES_DIR,
    VIDEO_OUTPUT_COUNT,
)
from core.models import (
    FinalResult,
    Layer1Result,
    Layer3Result,
    Layer4Result,
    PipelineResult,
    SimilarityCluster,
)

logger = logging.getLogger(__name__)


def _effective_weights(has_face: bool, has_vlm: bool = False) -> dict[str, float]:
    """每张照片的融合权重（总和恒为 1.0）：

    1. VLM 已评分（人脸因素已被 VLM 基于 L4 注入信息消化）→ face 权重归零重分配，
       避免人脸被双重计数；
    2. VLM 未评分（降级路径）且照片无人脸 → face 权重重分配，
       无脸照片（风景/动物等）不因缺少人脸维度而系统性吃亏。
    """
    w = dict(FUSION_WEIGHTS)
    if has_vlm and "face" in w:
        extra = w.pop("face")
        total = sum(w.values())
        if total > 0:
            for k in w:
                w[k] += extra * (w[k] / total)
    if not has_face and "face" in w:
        extra = w.pop("face")
        total = sum(w.values())
        if total > 0:
            for k in w:
                w[k] += extra * (w[k] / total)
    return w


def _emit(progress_callback, event: str, **data):
    """安全调用进度回调"""
    if progress_callback:
        try:
            progress_callback(event, data)
        except Exception:
            pass


def _compute_vlm_dim_score(r, weights: dict, dim_w_sum: float) -> float:
    """平台权重加权后的 VLM 构图/色彩/光影分（0-10）"""
    if dim_w_sum <= 0:
        return r.overall_score
    return (
        r.composition_score * weights.get("composition", 0.35)
        + r.color_score * weights.get("color", 0.30)
        + r.lighting_score * weights.get("lighting", 0.20)
    ) / dim_w_sum


def _vlm_effective(r, weights: dict, dim_w_sum: float) -> float | None:
    """VLM 有效分（与融合 q_vlm 公式一致，见 _compute_final_score）；VLM 未评分返回 None"""
    if r.vlm_overall_score <= 0:
        return None
    return 0.7 * _compute_vlm_dim_score(r, weights, dim_w_sum) + 0.3 * r.vlm_overall_score


def _vlm_topiq_divergence(r, weights: dict, dim_w_sum: float) -> float | None:
    """VLM 有效分与 TOPIQ 分之差（离群诊断，不改分）"""
    q = _vlm_effective(r, weights, dim_w_sum)
    if q is None:
        return None
    return round(abs(q - r.overall_score), 4)


def _compute_final_score(r, weights: dict, dim_w_sum: float, face_scores: dict, l1_scores: dict) -> float:
    """融合综合分（0-10）：VLM×w_vlm + TOPIQ×w_topiq + 人脸×w_face + 客观×w_obj

    - VLM 已评分时人脸因素由 VLM 基于 L4 注入信息消化，face 权重归零（见 _effective_weights）
    - VLM 未评分（降级路径）时保留 face 权重；无脸照片权重重分配
    """
    has_face = face_scores.get(r.path, 0.0) > 0
    has_vlm = r.vlm_overall_score > 0
    w = _effective_weights(has_face, has_vlm)
    topiq10 = r.overall_score  # TOPIQ 批次归一化后 0-10
    if has_vlm:
        q_vlm = 0.7 * _compute_vlm_dim_score(r, weights, dim_w_sum) + 0.3 * r.vlm_overall_score
    else:
        q_vlm = topiq10  # VLM 未评分，回退到客观审美
    return (
        w["vlm"] * q_vlm
        + w["topiq"] * topiq10
        + w.get("face", 0.0) * (face_scores.get(r.path, 0.0) * 10)
        + w["objective"] * (l1_scores.get(r.path, 0.5) * 10)
    )


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


def _save_layer_checkpoint(output_dir: Path, layer: int, extra: dict) -> None:
    """保存指定层的 checkpoint，自动合并前面各层的数据

    修复历史 bug：原实现每层只存自己的数据，中间层数据被覆盖，
    中断后无法从任意层恢复。累积式保存保证 checkpoint 始终包含
    所有已完成层的数据（kept/rejected/clusters/aes_results/face_scores/…）。
    """
    base = _load_checkpoint(output_dir) or {}
    skipped = list(base.get("skipped_layers", []))
    for s in extra.get("skipped_layers", []):
        if s not in skipped:
            skipped.append(s)
    data = {**base, "version": 2, "completed_layer": layer, **extra, "skipped_layers": skipped}
    _save_checkpoint(output_dir, data)


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
    from core.layer1_objective import batch_analyze_with_phash
    from core.layer2_similarity import cluster_photos
    from core.layer3_aesthetic import apply_vlm_dimensions, score_topiq
    from core.layer4_face import analyze_faces, to_face_prompt_summary

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

    platform_config = PLATFORMS.get(platform, PLATFORMS["xiaohongshu"])

    logger.info(f"照片目录: {photos_dir} | 数量: {len(photos)} | 平台: {platform}")

    # ── 检查 checkpoint ──
    # 版本化：v2 的 completed_layer 为步骤编号（1=L1, 2=L2, 3=L4, 4=L3，L4 前置）。
    # 旧版 v1 checkpoint（completed_layer 为层号，且 L3 的 VLM 评分未注入人脸信息）：
    # 仅复用 L1/L2 数据，L4/L3 强制重跑。
    checkpoint = None if force else _load_checkpoint(output_dir)
    if checkpoint:
        if checkpoint.get("version") == 2:
            logger.info(
                f"从步骤 {checkpoint['completed_layer'] + 1} 继续"
                f"（跳过已完成 {checkpoint['completed_layer']} 步）"
            )
        else:
            completed = checkpoint.get("completed_layer", 0)
            if completed >= 2:
                checkpoint["completed_layer"] = 2
                logger.info("旧版 checkpoint：复用 L1/L2，L4/L3 重跑（VLM 需注入人脸信息）")
            elif completed >= 1:
                checkpoint["completed_layer"] = 1
                logger.info("旧版 checkpoint：复用 L1，L2 起重跑")
            else:
                checkpoint = None

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
        kept = [Layer1Result(**r) for r in checkpoint.get("kept", [])]
        rejected = [Layer1Result(**r) for r in checkpoint.get("rejected", [])]
        skipped_layers = checkpoint.get("skipped_layers", [])
        # 重建流水阶段算好的 pHash（序列化为十六进制字符串存储）
        phash_map = {
            p: (imagehash.ImageHash(v) if isinstance(v, str) else v)
            for p, v in checkpoint.get("phash_map", {}).items()
        }
        logger.info(f"[L1] 从 checkpoint 恢复: 保留 {len(kept)}, 淘汰 {len(rejected)}")
    else:
        _emit(progress_callback, "layer_start", layer=1, layer_name="客观指标")
        try:
            # 逐张流水：分析完即算 pHash（不等整批），哈希供 L2/融合复用
            l1_results, phash_map = batch_analyze_with_phash(photos)
            kept = [r for r in l1_results if not r.rejected]
            rejected = [r for r in l1_results if r.rejected]
            logger.info(f"[L1] 保留: {len(kept)}, 淘汰: {len(rejected)}")
            for r in rejected:
                logger.debug(f"  淘汰: {r.filename} ({r.reject_reason})")
        except Exception as e:
            logger.error(f"[L1] 失败，跳过: {e}")
            kept = [Layer1Result(path=str(p), filename=p.name) for p in photos]
            rejected = []
            phash_map = {}
            skipped_layers.append("Layer 1")
        _emit(progress_callback, "layer_end", layer=1, kept=len(kept), rejected=len(rejected))
        _save_layer_checkpoint(output_dir, 1, {
            "kept": [r.model_dump() for r in kept],
            "rejected": [r.model_dump() for r in rejected],
            "phash_map": {p: str(h) for p, h in phash_map.items()},
            "skipped_layers": skipped_layers,
        })

    # 建立 Layer 1 客观分映射（0-1 范围）
    l1_scores = {r.path: r.overall for r in kept}

    # ── Layer 2: 相似聚类 ──
    if checkpoint and checkpoint["completed_layer"] >= 2:
        clusters = [SimilarityCluster(**c) for c in checkpoint.get("clusters", [])]
        representatives = [r for c in clusters for r in (c.representatives or [c.representative])]
        merged = sum(c.member_count - 1 for c in clusters if c.member_count > 1)
        skipped_layers = checkpoint.get("skipped_layers", skipped_layers)
        logger.info(f"[L2] 从 checkpoint 恢复: {len(clusters)} 组")
    else:
        _emit(progress_callback, "layer_start", layer=2, layer_name="相似聚类")
        try:
            sharpness_scores = {r.path: r.sharpness for r in kept}
            # 复用 L1 流水阶段算好的 pHash，跳过重复计算
            clusters = cluster_photos([r.path for r in kept], sharpness_scores, precomputed_hashes=phash_map)
            representatives = [r for c in clusters for r in (c.representatives or [c.representative])]
            merged = sum(c.member_count - 1 for c in clusters if c.member_count > 1)
            logger.info(f"[L2] 聚类: {len(clusters)} 组, 合并: {merged}")
        except Exception as e:
            logger.error(f"[L2] 失败，跳过: {e}")
            representatives = [r.path for r in kept]
            merged = 0
            skipped_layers.append("Layer 2")
        _emit(progress_callback, "layer_end", layer=2, groups=len(representatives), merged=merged)
        _save_layer_checkpoint(output_dir, 2, {
            "clusters": [c.model_dump() for c in clusters],
            "skipped_layers": skipped_layers,
        })

    # ── Layer 4 + Layer 3-TOPIQ：层间并发（人脸分析 ∥ TOPIQ 推理，互不依赖）──
    if checkpoint and checkpoint["completed_layer"] >= 3:
        face_results = [Layer4Result(**r) for r in checkpoint.get("face_results", [])]
        aes_topiq_results = []  # L3 将从 checkpoint 恢复或重跑 TOPIQ
        skipped_layers = checkpoint.get("skipped_layers", skipped_layers)
        logger.info(f"[L4] 从 checkpoint 恢复: {len(face_results)} 张")
    else:
        _emit(progress_callback, "layer_start", layer=3, layer_name="人脸质量")
        from concurrent.futures import ThreadPoolExecutor
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_face = pool.submit(analyze_faces, representatives)
                f_topiq = pool.submit(score_topiq, representatives, "auto")
                try:
                    face_results = f_face.result()
                    face_count = sum(1 for r in face_results if r.has_face)
                    blink_count = sum(1 for r in face_results if r.blink_detected)
                    logger.info(f"[L4] 人脸: {face_count}, 闭眼: {blink_count}")
                except Exception as e:
                    logger.error(f"[L4] 失败，跳过: {e}")
                    face_results = []
                    skipped_layers.append("Layer 4")
                try:
                    aes_topiq_results = f_topiq.result()
                    logger.info(f"[L3-TOPIQ] 完成: {len(aes_topiq_results)} 张")
                except Exception as e:
                    logger.error(f"[L3-TOPIQ] 失败，跳过: {e}")
                    aes_topiq_results = []
                    skipped_layers.append("Layer 3")
        except Exception as e:
            logger.error(f"[L4/L3-TOPIQ] 并发启动失败: {e}")
            face_results = []
            aes_topiq_results = []
        _emit(progress_callback, "layer_end", layer=3, faces=len(face_results))
        _save_layer_checkpoint(output_dir, 3, {
            "face_results": [r.model_dump() for r in face_results],
            "skipped_layers": skipped_layers,
        })

    # 派生 L4 产物：融合分映射 + VLM prompt 注入文案
    face_scores = {r.path: r.overall_face_score for r in face_results}
    # 降级兜底：多人环境场景（≥6 张脸且主脸占比 < 10%）判定人脸为环境元素，不参与加分
    for r in face_results:
        if r.face_count >= 6 and r.face_ratio < 0.10:
            face_scores[r.path] = 0.0
    face_summaries = {r.path: to_face_prompt_summary(r) for r in face_results}
    # 人像硬伤标记：主脸闭眼或主脸明显模糊（对应 prompt 硬伤预检第 9 条）
    # 作为 VLM 综合审美 6 分上限的程序化钳制依据（不依赖 LLM 自觉）
    face_hard_flags = {
        r.path: r.main_face_blink or (r.has_face and r.face_clarity < 0.25)
        for r in face_results
    }

    # ── Layer 3: 审美评分（TOPIQ 已与 L4 并发完成，此处补 VLM 维度分析）──
    if checkpoint and checkpoint["completed_layer"] >= 4:
        aes_results = [Layer3Result(**r) for r in checkpoint.get("aes_results", [])]
        skipped_layers = checkpoint.get("skipped_layers", skipped_layers)
        logger.info(f"[L3] 从 checkpoint 恢复: {len(aes_results)} 张")
    else:
        _emit(progress_callback, "layer_start", layer=4, layer_name="审美评分")
        try:
            if not aes_topiq_results:
                aes_topiq_results = score_topiq(representatives, "auto")
            aes_results = apply_vlm_dimensions(
                aes_topiq_results,
                face_summaries=face_summaries,
                face_hard_flags=face_hard_flags,
            )
            aes_results.sort(key=lambda x: x.overall_score, reverse=True)
        except Exception as e:
            logger.error(f"[L3] 失败，跳过: {e}")
            aes_results = [Layer3Result(path=p, filename=Path(p).name) for p in representatives]
            skipped_layers.append("Layer 3")
        _emit(progress_callback, "layer_end", layer=4, scored=len(aes_results))
        _save_layer_checkpoint(output_dir, 4, {
            "aes_results": [r.model_dump() for r in aes_results],
            "skipped_layers": skipped_layers,
        })

    # ── 融合评分：VLM 多维审美 + TOPIQ + 人脸 + 客观指标统一加权 ──
    # 说明：VLM 评分失败/未配置 API 时，q_vlm 回退为 TOPIQ 分，保证仍可排序。
    weights = platform_config.get("weights", {
        "composition": 0.35, "color": 0.30, "lighting": 0.20, "face": 0.15
    })
    dim_w_sum = (weights.get("composition", 0.35) + weights.get("color", 0.30)
                 + weights.get("lighting", 0.20))

    # 融合分计算复用模块级函数（与 run_video_ranking 同源）

    # 多样性惩罚：复用 L1 流水阶段算好的 pHash（不再重复计算）
    hash_cache: dict[str, imagehash.ImageHash | None] = {
        r.path: phash_map.get(r.path) for r in aes_results
    }
    if not any(hash_cache.values()):
        # phash_map 缺失（旧 checkpoint 恢复路径）：兜底重算，保证多样性惩罚生效
        from core.layer2_similarity import compute_phash
        logger.info("phash_map 缺失，兜底重算 pHash（多样性惩罚）...")
        for r in aes_results:
            hash_cache[r.path] = compute_phash(r.path)

    # 计算融合分
    score_map: dict[str, float] = {r.path: _compute_final_score(r, weights, dim_w_sum, face_scores, l1_scores) for r in aes_results}

    # 多样性惩罚：对相似照片扣分，避免同场景占据多个名额
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

    # 按融合分排序，直接取 Top N（不再分两级，避免候选池边界错杀）
    aes_results.sort(key=lambda x: score_map[x.path], reverse=True)
    top_n = min(FINAL_OUTPUT_COUNT, len(aes_results))
    top_photos = aes_results[:top_n]

    logger.info(f"[终审] 平台: {platform_config['name']}, 候选: {len(aes_results)}, 入选: {len(top_photos)}")
    logger.info(
        f"[终审] 融合权重: VLM{FUSION_WEIGHTS['vlm']} TOPIQ{FUSION_WEIGHTS['topiq']} "
        f"人脸{FUSION_WEIGHTS['face']}（VLM 评分时归零） 客观{FUSION_WEIGHTS['objective']} | "
        f"平台维度权重: 构图{weights.get('composition', 0.35)} "
        f"色彩{weights.get('color', 0.30)} "
        f"光影{weights.get('lighting', 0.20)} "
        f"人脸{weights.get('face', 0.15)}"
    )

    # ── 保存结果 ──
    output_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧输出（保留 checkpoint 在最终清理时删除）
    for old in output_dir.glob("*"):
        if old.is_file() and old.name != "checkpoint.json":
            old.unlink()

    # 合并结果并保存
    final_results: list[FinalResult] = []
    for r in top_photos:
        final_results.append(FinalResult(
            path=r.path,
            filename=r.filename,
            aesthetic_score=r.aesthetic_score,
            overall_score=r.overall_score,
            final_score=score_map[r.path],  # 融合综合分（0-10），含多样性惩罚
            category=r.category,  # L3 VLM 分类标签（风景/人像/夜景等）
        ))

    ranking_file = output_dir / "ranking.json"
    with open(ranking_file, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in final_results], f, ensure_ascii=False, indent=2)

    # 保存全部评分（完整排行，不仅仅是 Top N）
    full_ranking = [
        {
            "path": r.path,
            "filename": r.filename,
            "aesthetic_score": r.aesthetic_score,
            "overall_score": r.overall_score,
            "final_score": score_map[r.path],
            "category": r.category,
            "vlm_overall_score": r.vlm_overall_score,
            "composition_score": r.composition_score,
            "color_score": r.color_score,
            "lighting_score": r.lighting_score,
            # 离群诊断：VLM 与 TOPIQ 严重分歧的照片（TOPIQ 有语义盲区，分歧≠错误）
            "vlm_topiq_divergence": _vlm_topiq_divergence(r, weights, dim_w_sum),
        }
        for r in aes_results
    ]
    full_ranking.sort(key=lambda x: x["final_score"], reverse=True)
    full_file = output_dir / "full_ranking.json"
    with open(full_file, "w", encoding="utf-8") as f:
        json.dump(full_ranking, f, ensure_ascii=False, indent=2)
    logger.info(f"完整排行已保存: {len(full_ranking)} 条 → {full_file}")

    # 复制照片（序号名，供前端展示）
    display_names: dict[str, str] = {}
    for i, r in enumerate(top_photos):
        src = Path(r.path)
        dst = output_dir / f"{i+1:02d}_{src.name}"
        shutil.copy2(src, dst)
        display_names[r.path] = dst.name
    # 榜单 filename 用展示文件名（保证 /api/results/.../{filename} 可访问）
    for fr in final_results:
        fr.filename = display_names.get(fr.path, fr.filename)

    logger.info(f"输出: Top {top_n} → {output_dir}")

    # checkpoint 保留（不再删除——方便后续查阅中间数据）

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


def _load_video_manifest(frames_dir: Path) -> list[dict]:
    """读取视频帧清单 manifest.json；缺失时从文件名兜底解析

    文件名格式：{视频名}_{签名}_t{秒数}.jpg；兜底解析仅保证
    source_video/timestamp 基本可用（签名前缀截断可能不精确）。
    """
    mf = frames_dir / "manifest.json"
    if mf.exists():
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception as e:
            logger.warning(f"manifest.json 读取失败，回退文件名解析: {e}")

    entries: list[dict] = []
    for p in sorted(frames_dir.glob("*.jpg")):
        stem = p.stem
        ts, src = 0.0, stem
        if "_t" in stem:
            head, _, tpart = stem.rpartition("_t")
            try:
                ts = float(tpart)
            except ValueError:
                ts = 0.0
            src = head.rsplit("_", 1)[0]  # 去掉签名段
        entries.append({
            "filename": p.name, "source_video": src, "timestamp": ts,
            "sharpness": 0.0, "overall": 0.5, "phash": "", "duration": 0.0,
        })
    return entries


def _allocate_quotas(
    videos: dict[str, float],
    total_quota: int,
    cap_ratio: float | None = None,
) -> dict[str, int]:
    """按视频时长比例分配视频榜名额（每视频至少 1，总和恰为 total_quota）

    Args:
        videos: {视频名: 时长秒}
        total_quota: 总额度（VIDEO_OUTPUT_COUNT）
        cap_ratio: 可选单视频上限占比（默认 None 不限制；
            注意：视频数 < 4 时 cap 会与比例分配冲突，需自行权衡）
    """
    if not videos:
        return {}
    total_dur = sum(videos.values())
    if total_dur <= 0:
        # 时长缺失退化：均分（余数给前几个）
        n = len(videos)
        base, rem = divmod(total_quota, n)
        return {
            v: base + (1 if i < rem else 0)
            for i, v in enumerate(sorted(videos))
        }

    quotas = {v: max(1, round(total_quota * d / total_dur)) for v, d in videos.items()}
    cap = None
    if cap_ratio:
        cap = max(1, int(total_quota * cap_ratio))
        for v in quotas:
            quotas[v] = min(quotas[v], cap)
    # 超发收敛：按“占比偏差最大者”逐减（偏差大 = 分多了）
    while sum(quotas.values()) > total_quota:
        cands = [v for v, q in quotas.items() if q > 1 and (cap is None or q > 1)]
        if not cands:
            break
        v = max(cands, key=lambda x: quotas[x] / videos[x])
        quotas[v] -= 1
    # 欠发补齐：给未达 cap 的最长视频（默认不设限则给最长视频）
    while sum(quotas.values()) < total_quota:
        if cap is not None:
            cands = [v for v, q in quotas.items() if q < cap]
            if not cands:
                break  # 名额受 cap 限制无法用满（视频数过少时属预期）
            v = max(cands, key=lambda x: videos[x])
        else:
            v = max(videos, key=lambda x: videos[x])
        quotas[v] += 1
    return quotas


def run_video_ranking(
    video_frames_dir: str | Path = None,
    platform: str = "xiaohongshu",
    progress_callback=None,
    max_total: int | None = None,
) -> dict:
    """视频帧独立竞争排名（双榜单中的“视频精选榜”）

    视频帧在抽帧时已过 L1/L2/L4 本地快筛，此处做跨视频聚类（L2）、
    L3 VLM + TOPIQ + 人脸深度评分与融合排名；最终名额按各视频时长
    比例分配（_allocate_quotas），避免长/短视频不公。

    Args:
        video_frames_dir: 视频帧目录（含 manifest.json）
        platform: 目标平台
        progress_callback: 进度回调
        max_total: 视频榜总额度（默认 VIDEO_OUTPUT_COUNT）
    """
    from concurrent.futures import ThreadPoolExecutor

    from core.layer2_similarity import cluster_photos
    from core.layer3_aesthetic import apply_vlm_dimensions, score_topiq
    from core.layer4_face import analyze_faces, to_face_prompt_summary

    frames_dir = Path(video_frames_dir or VIDEO_FRAMES_DIR)
    max_total = max_total or VIDEO_OUTPUT_COUNT
    platform_config = PLATFORMS.get(platform, PLATFORMS["xiaohongshu"])
    output_dir = OUTPUT_DIR / platform

    manifest = _load_video_manifest(frames_dir)
    by_file = {frames_dir / m["filename"]: m for m in manifest if (frames_dir / m["filename"]).exists()}
    if not by_file:
        logger.info(f"[视频榜] 无合格视频帧（{frames_dir}），跳过")
        return {"total": 0, "ranked": 0, "output_count": 0, "results": []}
    logger.info(f"[视频榜] 候选帧: {len(by_file)}（来源视频数: {len({m['source_video'] for m in by_file.values()})}）")

    # L2 聚类（复用快筛阶段算好的 pHash，跨视频帧也合并去重）
    sharpness_scores = {str(p): m["sharpness"] for p, m in by_file.items()}
    precomputed = {
        str(p): imagehash.hex_to_hash(m["phash"])
        for p, m in by_file.items() if m.get("phash")
    }
    try:
        clusters = cluster_photos([str(p) for p in by_file], sharpness_scores, precomputed_hashes=precomputed)
        representatives = [r for c in clusters for r in (c.representatives or [c.representative])]
        merged = sum(c.member_count - 1 for c in clusters if c.member_count > 1)
        logger.info(f"[视频榜-L2] 聚类: {len(clusters)} 组, 合并: {merged}")
    except Exception as e:
        logger.error(f"[视频榜-L2] 失败，降级: {e}")
        representatives = [str(p) for p in by_file]
        merged = 0

    # L4 + L3-TOPIQ 并发
    _emit(progress_callback, "layer_start", layer=1, layer_name="视频榜·人脸/TOPIQ")
    face_results, aes_topiq_results = [], []
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_face = pool.submit(analyze_faces, representatives)
            f_topiq = pool.submit(score_topiq, representatives, "auto")
            face_results = f_face.result()
            aes_topiq_results = f_topiq.result()
    except Exception as e:
        logger.error(f"[视频榜-L4/L3] 失败，跳过: {e}")
        aes_topiq_results = [Layer3Result(path=p, filename=Path(p).name) for p in representatives]
    _emit(progress_callback, "layer_end", layer=1, faces=len(face_results))

    # L3 VLM 维度分析
    face_scores = {r.path: r.overall_face_score for r in face_results}
    face_summaries = {r.path: to_face_prompt_summary(r) for r in face_results}
    face_hard_flags = {
        r.path: r.main_face_blink or (r.has_face and r.face_clarity < 0.25)
        for r in face_results
    }
    try:
        aes_results = apply_vlm_dimensions(
            aes_topiq_results,
            face_summaries=face_summaries,
            face_hard_flags=face_hard_flags,
        )
        aes_results.sort(key=lambda x: x.overall_score, reverse=True)
    except Exception as e:
        logger.error(f"[视频榜-L3] VLM 失败，降级为 TOPIQ 排序: {e}")
        aes_results = aes_topiq_results or [
            Layer3Result(path=p, filename=Path(p).name) for p in representatives
        ]

    # 融合分 + 多样性惩罚（与照片榜同源）
    l1_scores = {str(p): m["overall"] for p, m in by_file.items()}
    weights = platform_config.get("weights", {
        "composition": 0.35, "color": 0.30, "lighting": 0.20, "face": 0.15
    })
    dim_w_sum = (weights.get("composition", 0.35) + weights.get("color", 0.30)
                 + weights.get("lighting", 0.20))
    score_map = {
        r.path: _compute_final_score(r, weights, dim_w_sum, face_scores, l1_scores)
        for r in aes_results
    }

    # 多样性惩罚：同一视频内相似帧扣分
    penalized: set[str] = set()
    for i in range(len(aes_results)):
        hi = precomputed.get(aes_results[i].path)
        if hi is None:
            continue
        for j in range(i + 1, len(aes_results)):
            hj = precomputed.get(aes_results[j].path)
            if hj is None:
                continue
            if abs(int(hi - hj)) < DIVERSITY_DISTANCE_THRESHOLD:
                pj = aes_results[j].path
                if pj not in penalized:
                    score_map[pj] *= (1.0 - DIVERSITY_PENALTY)
                    penalized.add(pj)

    # 按视频时长比例分配名额，组内取配额
    video_dur: dict[str, float] = {}
    frames_by_video: dict[str, list] = {}
    for r in aes_results:
        m = by_file.get(Path(r.path), {})
        src = m.get("source_video", Path(r.path).stem)
        video_dur.setdefault(src, m.get("duration", 0.0))
        frames_by_video.setdefault(src, []).append(r)
    quotas = _allocate_quotas(video_dur, max_total)
    logger.info(f"[视频榜] 名额分配(时长比例): {quotas}")

    selected: list = []
    for src, rs in frames_by_video.items():
        rs.sort(key=lambda x: score_map[x.path], reverse=True)
        selected.extend(rs[:quotas.get(src, 1)])
    selected.sort(key=lambda x: score_map[x.path], reverse=True)

    def _to_result(r) -> FinalResult:
        m = by_file.get(Path(r.path), {})
        return FinalResult(
            path=r.path,
            filename=r.filename,
            aesthetic_score=r.aesthetic_score,
            overall_score=r.overall_score,
            final_score=score_map[r.path],
            category=r.category,
            source_video=m.get("source_video", ""),
            timestamp=m.get("timestamp", 0.0),
        )

    video_results = [_to_result(r) for r in selected]

    # 复制入选帧到 output_dir（序号名，供前端展示；与照片榜命名风格一致）
    output_dir.mkdir(parents=True, exist_ok=True)
    display_names: dict[str, str] = {}
    for i, r in enumerate(selected, 1):
        src = Path(r.path)
        if src.exists():
            dst = output_dir / f"v{i:02d}_{src.name}"
            shutil.copy2(src, dst)
            display_names[r.path] = dst.name

    # 保存 video_ranking.json / video_full_ranking.json（filename 用 output 目录内可展示文件名）
    for vr in video_results:
        vr.filename = display_names.get(vr.path, vr.filename)
    with open(output_dir / "video_ranking.json", "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in video_results], f, ensure_ascii=False, indent=2)

    all_sorted = sorted(aes_results, key=lambda x: score_map[x.path], reverse=True)
    full = [
        {
            **r.model_dump(),
            "final_score": score_map[r.path],
            # 与照片榜一致：VLM 与 TOPIQ 离群诊断（未评分返回 null）
            "vlm_topiq_divergence": _vlm_topiq_divergence(r, weights, dim_w_sum),
            "source_video": by_file.get(Path(r.path), {}).get("source_video", ""),
            "timestamp": by_file.get(Path(r.path), {}).get("timestamp", 0.0),
        }
        for r in all_sorted
    ]
    with open(output_dir / "video_full_ranking.json", "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[视频榜] 完成: 候选{len(aes_results)} → 入选{len(video_results)}"
        f"（名额{sum(quotas.values())}/{max_total}）"
    )
    return {
        "total": len(aes_results),
        "ranked": len(video_results),
        "output_count": len(video_results),
        "quotas": quotas,
        "results": [r.model_dump() for r in video_results],
        "output_dir": str(output_dir),
    }
