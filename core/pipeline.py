"""Pipeline — 整合所有层的完整流水线（支持 GPU/CPU 自动切换 + 断点续传）"""
import json
import logging
import os
import re
import shutil
from bisect import bisect_left as _bisect_left, bisect_right as _bisect_right
from collections import Counter
from datetime import datetime
from pathlib import Path

import imagehash

from config import (
    BUCKET_REJECT_FLOOR,
    BUCKET_REJECT_PERCENTILE,
    BUCKET_SELECT_FLOOR,
    BUCKET_SELECT_PERCENTILE,
    CATEGORY_MIN_RATIO,
    CATEGORY_QUOTA_MIN_SCORE_RATIO,
    CATEGORY_WEIGHT_OVERRIDES,
    DIVERSITY_DISTANCE_THRESHOLD,
    DIVERSITY_PENALTY,
    FINAL_OUTPUT_COUNT,
    FUSION_WEIGHTS,
    OUTPUT_DIR,
    PHOTOS_DIR,
    PLATFORMS,
    QWEN_MODEL,
    SCENE_BUCKET_MAX,
    SCENE_BUCKET_MAX_RELAXED,
    SCENE_BUCKET_MINUTES,
    TOP_SEGMENT_ENSURE_CATEGORY,
    VIDEO_FRAMES_DIR,
    VIDEO_OUTPUT_COUNT,
    VLM_SATURATION_RESAMPLE,
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


def _face_hard_flag(r: Layer4Result) -> str | None:
    """L4 人像硬伤触发源："blink"（主脸闭眼）/ "blur"（主脸占比 >10% 且清晰度 <0.10）/ None

    仅给出客观触发源，是否钳制由 layer3 结合 VLM 主体裁断
    （person_is_subject / 自报硬伤）统一决定，见 _apply_dims_to_result。
    """
    if r.main_face_blink:
        return "blink"
    if r.has_face and r.face_ratio > 0.10 and r.face_clarity < 0.10:
        return "blur"
    return None


def _report_scoring_health(aes_results: list[Layer3Result], label: str) -> None:
    """跑后统计报警：钳制率/低分占比/饱和分/同分簇异常即打 warning

    历史教训：53 张中 29 张被误钳制（55%，pareidolia 误检 + 二次惩罚），
    系统静默通过，直到用户看照片才发现。加统计报警后同类 bug 分钟级可现。
    报警附带具体文件名与规则分布（clamp_source），可直接定位取证。
    """
    scored = [r for r in aes_results if r.vlm_overall_score > 0]
    if not scored:
        return
    clamped = [r for r in scored if r.vlm_clamp]
    low = sum(1 for r in scored if r.vlm_overall_score < 2.0)
    clamp_rate = len(clamped) / len(scored)
    low_rate = low / len(scored)
    logger.info(
        f"[{label}] 评分健康: VLM 评分 {len(scored)} 张，钳制 {len(clamped)} 张（{clamp_rate:.0%}），"
        f"低分(<2) {low} 张（{low_rate:.0%}）"
    )
    if clamp_rate > 0.20:
        by_source = dict(Counter(r.clamp_source or "unknown" for r in clamped))
        examples = ", ".join(r.filename for r in clamped[:3])
        logger.warning(
            f"[{label}] 钳制率 {clamp_rate:.0%} 超过 20%——疑似人脸硬伤判定误触发"
            f"（检测误检/阈值漂移），规则分布 {by_source}，例: {examples}"
        )
    if low_rate > 0.10:
        logger.warning(
            f"[{label}] 低分(<2) 占比 {low_rate:.0%} 超过 10%——疑似二次惩罚或归一化异常"
        )
    # 饱和分体检：VLM 锚点打分/捧分信号（实证：历史数据 11.1% 拿满分）
    sat = sum(1 for r in scored if r.vlm_saturated)
    if sat:
        logger.info(f"[{label}] VLM 饱和分 {sat} 张（{sat / len(scored):.0%}）（综合≥9.8 或三维全≥9）")
    # 同分簇体检：完全同分 ≥5 张 → 疑似锚点打分，头部区分度崩坏
    vec_count = Counter(
        (round(r.composition_score, 1), round(r.color_score, 1),
         round(r.lighting_score, 1), round(r.vlm_overall_score, 2))
        for r in scored
    )
    top_vec, top_n = vec_count.most_common(1)[0]
    if top_n >= 5:
        logger.warning(
            f"[{label}] 同分簇异常: {top_n} 张分数完全一致 {list(top_vec)}——"
            f"疑似 VLM 锚点打分，头部区分度下降"
        )


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


def _weights_for(r, base: dict) -> dict:
    """该照片的维度权重：题材命中时覆盖平台权重，否则原样返回

    题材化权重的现实意义：夜景的光影分、美食的色彩分、人像的人脸分
    才是决定性的；用平台统一权重等于用同一把尺子量不同题材。
    """
    override = CATEGORY_WEIGHT_OVERRIDES.get((r.category or "").strip())
    if not override:
        return base
    merged = dict(base)
    merged.update(override)
    return merged


def _dim_weight_sum(weights: dict) -> float:
    """构图/色彩/光影三项权重之和（VLM 维度加权分的分母）"""
    return (
        weights.get("composition", 0.35)
        + weights.get("color", 0.30)
        + weights.get("lighting", 0.20)
    )


def _vlm_effective(r, weights: dict, dim_w_sum: float | None = None) -> float | None:
    """VLM 有效分（与融合 q_vlm 公式一致，见 _compute_final_score）；VLM 未评分返回 None

    权重按题材覆盖后重算分母——dim_w_sum 参数仅为签名兼容，传入值不再使用。
    """
    if r.vlm_overall_score <= 0:
        return None
    w = _weights_for(r, weights)
    return 0.7 * _compute_vlm_dim_score(r, w, _dim_weight_sum(w)) + 0.3 * r.vlm_overall_score


def _vlm_topiq_divergence(r, weights: dict, dim_w_sum: float | None = None) -> float | None:
    """VLM 有效分与 TOPIQ 分之差（离群诊断，不改分）"""
    q = _vlm_effective(r, weights, dim_w_sum)
    if q is None:
        return None
    return round(abs(q - r.overall_score), 4)


def _compute_final_score(r, weights: dict, dim_w_sum: float | None, face_scores: dict, l1_scores: dict) -> float:
    """融合综合分（0-10）：VLM×w_vlm + TOPIQ×w_topiq + 人脸×w_face + 客观×w_obj

    - VLM 已评分时人脸因素由 VLM 基于 L4 注入信息消化，face 权重归零（见 _effective_weights）
    - VLM 未评分（降级路径）时保留 face 权重；无脸照片权重重分配
    - 维度权重按题材覆盖（CATEGORY_WEIGHT_OVERRIDES），dim_w_sum 参数仅为兼容保留
    """
    weights = _weights_for(r, weights)
    dim_w_sum = _dim_weight_sum(weights)
    has_face = face_scores.get(r.path, 0.0) > 0
    if has_face and r.person_is_subject is False:
        # VLM 裁断人物非主体（含背景路人/误检假脸）：face 权重归零，
        # 防降级路径（VLM 未评分）被假脸的低分人脸分污染
        has_face = False
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


def _expand_rerun_layers(rerun_layers: list[str] | None) -> set[str]:
    """增量重跑依赖传播：上游层重跑 → 依赖它的下游层必须连带重跑

    依赖链：l1(客观) → l2(聚类) → {l4(人脸), topiq, vlm}；l4(人脸) → vlm（VLM prompt 注入人脸信息）。
    - 重跑 l1 → l2/l4/topiq/vlm 全部连带（聚类与评分全部基于新客观筛选）
    - 重跑 l2 → l4/topiq/vlm 连带（代表帧集合变化）
    - 重跑 l4 → vlm 连带（人脸信息变化，VLM 需重新注入）
    - topiq 与 vlm 相互独立：各自可单独重跑
    """
    rerun = set(rerun_layers or [])
    if "l1" in rerun:
        rerun |= {"l2", "l4", "topiq", "vlm"}
    if "l2" in rerun:
        rerun |= {"l4", "topiq", "vlm"}
    if "l4" in rerun:
        rerun |= {"vlm"}
    return rerun


def _save_layer_checkpoint(output_dir: Path, layer: int, extra: dict) -> None:
    """保存指定层的 checkpoint，自动合并前面各层的数据

    修复历史 bug：原实现每层只存自己的数据，中间层数据被覆盖，
    中断后无法从任意层恢复。累积式保存保证 checkpoint 始终包含
    所有已完成层的数据（kept/rejected/clusters/aes_results/face_scores/…）。

    v3 起携带 meta（VLM prompt/模型指纹）：加载时校验，不一致则
    旧 VLM 分作废重评，防新旧 prompt 分数静默混排。
    """
    from core.layer3_aesthetic import vlm_prompt_fingerprint

    base = _load_checkpoint(output_dir) or {}
    skipped = list(base.get("skipped_layers", []))
    for s in extra.get("skipped_layers", []):
        if s not in skipped:
            skipped.append(s)
    data = {
        **base, "version": 3, "completed_layer": layer, **extra,
        "skipped_layers": skipped,
        "meta": {"prompt_hash": vlm_prompt_fingerprint(), "model": QWEN_MODEL},
    }
    _save_checkpoint(output_dir, data)


def run_pipeline(
    photos_dir: str | Path = None,
    platform: str = "xiaohongshu",
    progress_callback=None,
    force: bool = False,
    rerun_layers: list[str] | None = None,
    incremental: bool = False,
    skip_l1: bool = False,
) -> dict:
    """运行完整流水线（支持进度回调 + 断点续传 + 分层增量重跑 + 新增照片增量）

    Args:
        photos_dir: 照片目录
        platform: 目标平台
        progress_callback: 进度回调函数
        force: 为 True 时忽略 checkpoint 从头开始
        rerun_layers: 只重跑指定层（其余从 checkpoint 恢复），如 ["topiq"]。
            可选: l1 / l2 / l4 / topiq / vlm；有依赖的层自动连带重跑
            （重跑 l1 → 全量；重跑 l2 → l4/topiq/vlm；重跑 l4 → vlm）。
            前提：checkpoint 已包含其余层的中间结果。
        incremental: 为 True 时只处理新增照片（不在 checkpoint 中的），
            旧照片的 L1/L2/L4/TOPIQ/VLM 结果全部从 checkpoint 复用；
            L2 重新聚类（本地计算，快），仅对新增代表跑评分，最终全量排名。
            与 rerun_layers 互斥（同时传入时 incremental 优先）。
        skip_l1: 为 True 时跳过 L1 淘汰（模糊/曝光不拦截），全部照片进入后续层
            （仍计算 sharpness/pHash 供聚类与多样性使用）；强制全量重跑。
    """
    from core.layer3_aesthetic import vlm_prompt_fingerprint

    photos_dir = Path(photos_dir or PHOTOS_DIR)
    output_dir = OUTPUT_DIR / platform

    # 收集所有照片
    photos = _scan_photos(photos_dir)
    if not photos:
        logger.warning(f"未找到照片: {photos_dir}")
        return {"error": "未找到照片", "total": 0}

    platform_config = PLATFORMS.get(platform, PLATFORMS["xiaohongshu"])

    logger.info(f"照片目录: {photos_dir} | 数量: {len(photos)} | 平台: {platform}")

    # 顺带清理过期解码缓存（防 cache/images 无限增长；失败不影响流水线）
    try:
        from core.image_cache import clean_image_cache
        clean_image_cache()
    except Exception as e:
        logger.debug(f"解码缓存清理失败（不影响流水线）: {e}")

    # ── 运行模式解析（checkpoint 恢复 / 增量重跑 / 新增照片增量）──
    prompt_hash = vlm_prompt_fingerprint()
    checkpoint, rerun, incremental, incremental_new = _resolve_run_mode(
        output_dir, force, rerun_layers, skip_l1, incremental, photos, prompt_hash,
    )

    # 仅在全新启动时清理输出目录
    if not checkpoint:
        output_dir.mkdir(parents=True, exist_ok=True)
        for old in output_dir.glob("*"):
            if old.is_file() and old.name != "checkpoint.json":
                old.unlink()

    _emit(progress_callback, "layer_start", layer=0, layer_name="开始", total=len(photos))

    skipped_layers: list[str] = []

    # ── Layer 1: 客观指标 ──
    kept, rejected, phash_map, skipped_layers = _stage_l1(
        photos, output_dir, checkpoint, rerun, incremental, incremental_new,
        skip_l1, progress_callback, skipped_layers,
    )

    # 建立 Layer 1 客观分映射（0-1 范围）
    l1_scores = {r.path: r.overall for r in kept}

    # ── Layer 2: 相似聚类 ──
    clusters, representatives, merged, skipped_layers = _stage_l2(
        kept, phash_map, output_dir, checkpoint, rerun, incremental,
        progress_callback, skipped_layers,
    )

    # ── Layer 4 + Layer 3-TOPIQ：层间并发 + VLM 图片级流水 ──
    # L4 分析完一张立即发起该张的 VLM API 调用（与 TOPIQ 推理并行），
    # 消除"等全部 L4 完成才启动 VLM"的空等（VLM 是网络 IO 最慢环节，提前启动收益最大）
    # 增量重跑：L4 / TOPIQ 可从 checkpoint 独立恢复（rerun_layers 指定重跑部分）
    vlm_tasks: list[tuple[str, object, str | None]] = []  # (path, VLM future, hard_flag)
    api_key = os.environ.get("QWEN_API_KEY", "")
    need_l4, need_topiq, scoring_targets = _plan_scoring(checkpoint, rerun, incremental, representatives)
    face_results, aes_topiq_results, skipped_layers = _stage_l4_topiq(
        checkpoint, representatives, scoring_targets, need_l4, need_topiq,
        api_key, output_dir, progress_callback, skipped_layers, vlm_tasks,
    )

    # 增量模式：合并 checkpoint 中旧代表的评分 + 本轮新代表的评分
    face_results, aes_topiq_results = _merge_incremental(
        incremental, checkpoint, representatives, scoring_targets,
        face_results, aes_topiq_results, output_dir, skipped_layers,
    )

    # 派生 L4 产物：融合分映射 + 人像硬伤触发源
    face_scores, face_hard_flags = _derive_face_maps(face_results)

    # ── Layer 3: 审美评分（TOPIQ 已与 L4 并发完成，VLM API 已在 L4 回调阶段逐张发起）──
    aes_results, skipped_layers = _stage_vlm(
        checkpoint, rerun, incremental, scoring_targets, need_topiq,
        aes_topiq_results, representatives, face_results, face_hard_flags,
        vlm_tasks, api_key, output_dir, progress_callback, skipped_layers,
    )

    # ── 融合评分：VLM 多维审美 + TOPIQ + 人脸 + 客观指标统一加权 ──
    score_map, top_photos, top_n, weights, dim_w_sum, select_audit = _fuse_and_rank(
        aes_results, phash_map, face_scores, l1_scores, platform_config,
    )

    # ── 保存结果并汇总 ──
    return _save_outputs(
        output_dir, platform, prompt_hash, photos, rejected, merged,
        top_n, skipped_layers, aes_results, top_photos, score_map, weights, dim_w_sum,
        select_audit,
    )


def _scan_photos(photos_dir: Path):
    """收集照片目录中的候选照片（按文件名排序去重）"""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.heic", "*.HEIC", "*.JPG", "*.PNG"]
    photos: list[Path] = []
    for ext in extensions:
        photos.extend(photos_dir.glob(ext))
    return sorted(set(photos), key=lambda p: p.name)



def _resolve_run_mode(output_dir: Path, force: bool, rerun_layers: list[str] | None,
    skip_l1: bool, incremental: bool, photos: list[Path], prompt_hash: str):
    """解析本轮运行模式：checkpoint 恢复 / 增量重跑 / 新增照片增量 / 跳过 L1

    集中处理模式互斥与依赖传播（上游层重跑→下游连带、VLM 指纹变更→强制重评）。
    返回 (checkpoint, rerun, incremental, incremental_new)。"""
    from core.layer3_aesthetic import vlm_prompt_fingerprint

    # ── 检查 checkpoint ──
    # 版本化：v2 的 completed_layer 为步骤编号（1=L1, 2=L2, 3=L4, 4=L3，L4 前置）。
    # 旧版 v1 checkpoint（completed_layer 为层号，且 L3 的 VLM 评分未注入人脸信息）：
    # 仅复用 L1/L2 数据，L4/L3 强制重跑。
    checkpoint = None if force else _load_checkpoint(output_dir)
    prompt_hash = vlm_prompt_fingerprint()

    # 增量重跑依赖传播：上游层重跑 → 依赖它的下游层必须连带重跑
    rerun = _expand_rerun_layers(rerun_layers)
    if rerun:
        logger.info(f"[增量重跑] 指定: {sorted(rerun_layers or [])} → 实际: {sorted(rerun)}")

    # 跳过 L1 淘汰：全部照片进入后续层，强制全量重跑
    if skip_l1:
        logger.info("[跳过L1] 模糊/曝光不拦截，全部照片进入后续层（强制全量重跑）")
        rerun = _expand_rerun_layers(["l1"])

    # VLM 分新鲜度校验：prompt/模型已变更（或旧 v2 无指纹）时，旧 VLM 分与
    # 新分语义不可比 → 强制全量重评 VLM（TOPIQ/L4 复用），防静默混排
    if checkpoint and "vlm" not in rerun and checkpoint.get("version") in (2, 3):
        cp_hash = (checkpoint.get("meta") or {}).get("prompt_hash")
        if cp_hash != prompt_hash:
            reason = "旧 checkpoint 无指纹(v2)" if cp_hash is None else f"指纹 {cp_hash} → {prompt_hash}"
            logger.warning(
                f"[Checkpoint] VLM 评分配置已变更（{reason}）："
                f"全部旧 VLM 分作废重评（TOPIQ/L4 复用）"
            )
            rerun = rerun | {"vlm"}
            if incremental:
                logger.warning("[增量] 已禁用：VLM 评分配置变更，全部照片需重评 VLM")
                incremental = False

    # 新增照片增量：只处理不在 checkpoint 中的照片，旧照片结果全部复用
    incremental_new: list[Path] = []
    if incremental:
        if rerun:
            logger.warning("[增量] 与 rerun_layers 同时传入，incremental 优先，忽略 rerun_layers")
            rerun = set()
        if checkpoint and checkpoint.get("version") in (2, 3):
            known = {r["path"] for r in checkpoint.get("kept", [])} | {
                r["path"] for r in checkpoint.get("rejected", [])
            }
            incremental_new = [p for p in photos if str(p) not in known]
            if not incremental_new:
                logger.info("[增量] 无新增照片，全部复用 checkpoint 结果")
                incremental = False  # 退化：走正常 checkpoint 恢复路径
            else:
                logger.info(
                    f"[增量] 新增 {len(incremental_new)} 张照片，"
                    f"其余 {len(photos) - len(incremental_new)} 张复用 checkpoint"
                )
        else:
            logger.warning("[增量] 无可用 checkpoint（v2/v3），退化为全量重跑")
            incremental = False
    if checkpoint:
        if checkpoint.get("version") in (2, 3):
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

    return checkpoint, rerun, incremental, incremental_new


def _stage_l1(photos: list[Path], output_dir: Path, checkpoint: dict | None,
    rerun: set, incremental: bool, incremental_new: list[Path],
    skip_l1: bool, progress_callback, skipped_layers: list[str]):
    """Layer 1 客观指标：checkpoint 恢复 / 增量补跑 / 全量重跑

    返回 (kept, rejected, phash_map, skipped_layers)。"""
    from core.layer1_objective import batch_analyze_with_phash

    # ── Layer 1: 客观指标 ──
    if checkpoint and checkpoint["completed_layer"] >= 1 and "l1" not in rerun:
        kept = [Layer1Result(**r) for r in checkpoint.get("kept", [])]
        rejected = [Layer1Result(**r) for r in checkpoint.get("rejected", [])]
        skipped_layers = checkpoint.get("skipped_layers", [])
        # 重建流水阶段算好的 pHash（序列化为十六进制字符串存储）
        phash_map = {
            p: (imagehash.hex_to_hash(v) if isinstance(v, str) else v)
            for p, v in checkpoint.get("phash_map", {}).items()
        }
        if incremental:
            # 增量：只对新增照片跑 L1，旧照片结果从 checkpoint 复用
            if incremental_new:
                new_l1, new_phash = batch_analyze_with_phash(incremental_new)
                new_kept = [r for r in new_l1 if not r.rejected]
                new_rej = [r for r in new_l1 if r.rejected]
                kept += new_kept
                rejected += new_rej
                phash_map.update(new_phash)
                logger.info(
                    f"[L1] 增量: 恢复 {len(kept) - len(new_kept)} 保留/"
                    f"{len(rejected) - len(new_rej)} 淘汰 + 新增 {len(new_kept)} 保留/"
                    f"{len(new_rej)} 淘汰"
                )
                _save_layer_checkpoint(output_dir, 1, {
                    "kept": [r.model_dump() for r in kept],
                    "rejected": [r.model_dump() for r in rejected],
                    "phash_map": {p: str(h) for p, h in phash_map.items()},
                    "skipped_layers": skipped_layers,
                })
            else:
                logger.info(f"[L1] 增量（无新增）: 保留 {len(kept)}, 淘汰 {len(rejected)}")
        else:
            logger.info(f"[L1] 从 checkpoint 恢复: 保留 {len(kept)}, 淘汰 {len(rejected)}")
    else:
        _emit(progress_callback, "layer_start", layer=1, layer_name="客观指标")
        try:
            # 逐张流水：分析完即算 pHash（不等整批），哈希供 L2/融合复用
            l1_results, phash_map = batch_analyze_with_phash(photos)
            if skip_l1:
                # 跳过 L1 淘汰：全部照片保留（含模糊/曝光），sharpness 仍供聚类与多样性使用
                kept = list(l1_results)
                rejected = []
                logger.info(f"[L1] 跳过淘汰: 全部 {len(kept)} 张保留")
            else:
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

    return kept, rejected, phash_map, skipped_layers


def _stage_l2(kept: list[Layer1Result], phash_map: dict, output_dir: Path,
    checkpoint: dict | None, rerun: set, incremental: bool,
    progress_callback, skipped_layers: list[str]):
    """Layer 2 相似聚类：checkpoint 恢复 / 增量重聚类 / 全量聚类

    返回 (clusters, representatives, merged, skipped_layers)。"""
    from core.layer2_similarity import cluster_photos

    # ── Layer 2: 相似聚类 ──
    if checkpoint and checkpoint["completed_layer"] >= 2 and "l2" not in rerun and not incremental:
        clusters = [SimilarityCluster(**c) for c in checkpoint.get("clusters", [])]
        representatives = [r for c in clusters for r in (c.representatives or [c.representative])]
        merged = sum(c.member_count - 1 for c in clusters if c.member_count > 1)
        skipped_layers = checkpoint.get("skipped_layers", skipped_layers)
        logger.info(f"[L2] 从 checkpoint 恢复: {len(clusters)} 组")
    else:
        _emit(progress_callback, "layer_start", layer=2, layer_name="相似聚类")
        try:
            sharpness_scores = {r.path: r.sharpness for r in kept}
            # 代表选取用 L1 综合分（清晰/曝光/噪点加权），而非只看 sharpness：
            # 旧实现使连拍组里"最锐但过曝/高噪点"的照片挤掉更优的一张
            quality_scores = {r.path: r.overall for r in kept}
            # 复用 L1 流水阶段算好的 pHash，跳过重复计算
            # 同日期约束：跨日期 pHash 相近的（海景/天空/建筑同质化）不视为连拍，防误合并
            clusters = cluster_photos(
                [r.path for r in kept], sharpness_scores,
                precomputed_hashes=phash_map, same_date=True,
                quality_scores=quality_scores,
            )
            representatives = [r for c in clusters for r in (c.representatives or [c.representative])]
            merged = sum(c.member_count - 1 for c in clusters if c.member_count > 1)
            logger.info(f"[L2] 聚类: {len(clusters)} 组, 合并: {merged}")
        except Exception as e:
            logger.error(f"[L2] 失败，跳过: {e}")
            clusters = []  # 降级：无聚类，全部保留为独立代表（修复 except 后 clusters 未定义导致 checkpoint 落盘 UnboundLocalError）
            representatives = [r.path for r in kept]
            merged = 0
            skipped_layers.append("Layer 2")
        _emit(progress_callback, "layer_end", layer=2, groups=len(representatives), merged=merged)
        _save_layer_checkpoint(output_dir, 2, {
            "clusters": [c.model_dump() for c in clusters],
            "skipped_layers": skipped_layers,
        })

    return clusters, representatives, merged, skipped_layers


def _plan_scoring(checkpoint: dict | None, rerun: set, incremental: bool, representatives: list[str]):
    """计算 L4/TOPIQ 本轮是否需重跑及评分目标集合

    返回 (need_l4, need_topiq, scoring_targets)。"""
    l4_ckpt_ok = bool(checkpoint and checkpoint["completed_layer"] >= 3 and checkpoint.get("face_results"))
    topiq_ckpt_ok = bool(checkpoint and checkpoint["completed_layer"] >= 4 and checkpoint.get("aes_results"))
    rerun_l4 = "l4" in rerun
    rerun_topiq = "topiq" in rerun
    need_l4 = not (l4_ckpt_ok and not rerun_l4)
    need_topiq = not (topiq_ckpt_ok and not rerun_topiq)

    # 增量模式：只对新增代表评分（旧代表从 checkpoint 恢复）
    scoring_targets: list[str] = representatives
    if incremental:
        old_face_paths = {r["path"] for r in checkpoint.get("face_results", [])}
        scoring_targets = [p for p in representatives if p not in old_face_paths]
        need_l4 = bool(scoring_targets)
        need_topiq = bool(scoring_targets)
        logger.info(
            f"[增量] 代表 {len(representatives)} 张，需评分 {len(scoring_targets)} 张"
            f"（其余从 checkpoint 复用）"
        )

    return need_l4, need_topiq, scoring_targets


def _stage_l4_topiq(checkpoint: dict | None, representatives: list[str], scoring_targets: list[str],
    need_l4: bool, need_topiq: bool, api_key: str, output_dir: Path,
    progress_callback, skipped_layers: list[str], vlm_tasks: list):
    """Layer 4 人脸 + Layer 3-TOPIQ 并发执行（L4 逐张回调预发 VLM 任务）

    返回 (face_results, aes_topiq_results, skipped_layers)。"""
    from core.layer3_aesthetic import score_topiq
    from core.layer4_face import analyze_faces, to_face_prompt_summary

    if not need_l4 and not need_topiq:
        face_results = [Layer4Result(**r) for r in checkpoint["face_results"]]
        aes_topiq_results = [Layer3Result(**r) for r in checkpoint["aes_results"]]
        skipped_layers = checkpoint.get("skipped_layers", skipped_layers)
        logger.info(f"[L4] 从 checkpoint 恢复: {len(face_results)} 张")
        logger.info(f"[L3-TOPIQ] 从 checkpoint 恢复: {len(aes_topiq_results)} 张")
    else:
        _emit(progress_callback, "layer_start", layer=3, layer_name="人脸质量")
        from concurrent.futures import ThreadPoolExecutor

        from core.layer3_aesthetic import _VLM_CONCURRENCY, _run_dimension_analysis
        vlm_pool = None
        try:
            # VLM 独立线程池（8 并发，网络 IO 密集；与 L4/TOPIQ 的 2-worker 池分离）
            vlm_pool = ThreadPoolExecutor(max_workers=_VLM_CONCURRENCY)

            def _on_face_done(r: Layer4Result) -> None:
                """L4 分析完一张 → 立即发起该张 VLM API 调用（图片级流水）"""
                if not api_key:
                    return
                fsum = to_face_prompt_summary(r)
                hflag = _face_hard_flag(r)
                vlm_tasks.append((
                    r.path,
                    vlm_pool.submit(_run_dimension_analysis, r.path, face_summary=fsum),
                    hflag,
                ))

            with ThreadPoolExecutor(max_workers=2) as pool:
                f_face = pool.submit(analyze_faces, scoring_targets, on_photo_done=_on_face_done) if need_l4 else None
                f_topiq = pool.submit(score_topiq, scoring_targets, "auto") if need_topiq else None
                if f_face is not None:
                    try:
                        face_results = f_face.result()
                        face_count = sum(1 for r in face_results if r.has_face)
                        blink_count = sum(1 for r in face_results if r.blink_detected)
                        logger.info(f"[L4] 人脸: {face_count}, 闭眼: {blink_count}")
                    except Exception as e:
                        logger.error(f"[L4] 失败，跳过: {e}")
                        face_results = []
                        skipped_layers.append("Layer 4")
                else:
                    face_results = [Layer4Result(**r) for r in checkpoint["face_results"]]
                    logger.info(f"[L4] 从 checkpoint 恢复: {len(face_results)} 张")
                if f_topiq is not None:
                    try:
                        aes_topiq_results = f_topiq.result()
                        logger.info(f"[L3-TOPIQ] 完成: {len(aes_topiq_results)} 张")
                    except Exception as e:
                        logger.error(f"[L3-TOPIQ] 失败，跳过: {e}")
                        aes_topiq_results = []
                        skipped_layers.append("Layer 3")
                else:
                    aes_topiq_results = [Layer3Result(**r) for r in checkpoint["aes_results"]]
                    logger.info(f"[L3-TOPIQ] 从 checkpoint 恢复: {len(aes_topiq_results)} 张")
        except Exception as e:
            logger.error(f"[L4/L3-TOPIQ] 并发启动失败: {e}")
            face_results = [] if need_l4 else [Layer4Result(**r) for r in checkpoint["face_results"]]
            aes_topiq_results = [] if need_topiq else [Layer3Result(**r) for r in checkpoint["aes_results"]]
        finally:
            if vlm_pool is not None:
                vlm_pool.shutdown(wait=False)  # 已提交任务照常完成，L3 阶段 await
        _emit(progress_callback, "layer_end", layer=3, faces=len(face_results))
        _save_layer_checkpoint(output_dir, 3, {
            "face_results": [r.model_dump() for r in face_results],
            "skipped_layers": skipped_layers,
        })

    return face_results, aes_topiq_results, skipped_layers


def _merge_incremental(incremental: bool, checkpoint: dict | None, representatives: list[str], scoring_targets: list[str],
    face_results: list[Layer4Result], aes_topiq_results: list[Layer3Result],
    output_dir: Path, skipped_layers: list[str]):
    """增量模式合并：checkpoint 旧代表评分 + 本轮新代表评分（TOPIQ 统一重归一化）

    非增量模式原样返回。返回 (face_results, aes_topiq_results)。"""
    # 增量模式：合并 checkpoint 中旧代表的评分 + 本轮新代表的评分
    if incremental:
        old_face = {r["path"]: Layer4Result(**r) for r in checkpoint.get("face_results", [])}
        old_aes = {r["path"]: Layer3Result(**r) for r in checkpoint.get("aes_results", [])}
        rep_set = set(representatives)
        merged_face = [old_face[p] for p in representatives if p in old_face]
        merged_face += [r for r in face_results if r.path in rep_set and r.path not in old_face]
        face_results = merged_face
        merged_aes = [old_aes[p] for p in representatives if p in old_aes]
        merged_aes += [r for r in aes_topiq_results if r.path in rep_set and r.path not in old_aes]
        aes_topiq_results = merged_aes
        # TOPIQ 全量统一重归一化：新/旧分可能来自不同子批次的 p5-p95 参照系，
        # 用原始分（topiq_raw_score）幂等重算统一标尺；旧记录缺 raw（旧 checkpoint）则保持原行为
        if aes_topiq_results and all(
            r.topiq_raw_score > 0 for r in aes_topiq_results if not r.error and r.overall_score > 0
        ):
            from core.layer3_aesthetic import _normalize_batch_scores
            _normalize_batch_scores(aes_topiq_results)
        logger.info(
            f"[增量] 合并评分: 人脸 {len(face_results)} 张（新评 {len(scoring_targets)} 张），"
            f"TOPIQ {len(aes_topiq_results)} 张"
        )
        _save_layer_checkpoint(output_dir, 3, {
            "face_results": [r.model_dump() for r in face_results],
            "skipped_layers": skipped_layers,
        })

    return face_results, aes_topiq_results


def _derive_face_maps(face_results: list[Layer4Result]):
    """派生 L4 产物：融合分映射 + 人像硬伤触发源（VLM 注入文案已在回调中生成）

    返回 (face_scores, face_hard_flags)。"""
    # 派生 L4 产物：融合分映射（VLM prompt 注入文案已在图片级流水回调中逐张生成）
    face_scores = {r.path: r.overall_face_score for r in face_results}
    # 降级兜底：多人环境场景（≥6 张脸且主脸占比 < 10%）判定人脸为环境元素，不参与加分
    for r in face_results:
        if r.face_count >= 6 and r.face_ratio < 0.10:
            face_scores[r.path] = 0.0
    # 人像硬伤触发源："blink"（主脸闭眼）/ "blur"（主脸占比 >10% 且清晰度 <0.10）
    # 最终是否钳制由 _apply_dims_to_result 结合 VLM 主体裁断决定（见 layer3）：
    # 人物非主体（含背景路人/误检假脸）→ 永不钳制；blur 路径需 VLM 自报"人脸模糊"佐证
    # 收紧条件：仅当人脸占比 >10%（主体人像）且清晰度 <0.10（真糊）才触发 blur，
    # 避免远景路人/正常拍摄的人像（人脸小、clarity 天然偏低）被误钳制到 6 分
    # （实测 0.25 阈值致 53 张含人照片几乎全部触发钳制，VLM 好分被归一化压至 ~2.6）
    face_hard_flags = {r.path: _face_hard_flag(r) for r in face_results}

    return face_scores, face_hard_flags


def _reapply_derived_scores(aes_results: list[Layer3Result],
                            face_hard_flags: dict[str, str | None]) -> None:
    """对 checkpoint 恢复的旧评分重算派生态（硬伤封顶 / VLM 归一化 / 钳制回填）

    历史断链：`_stage_vlm` 的恢复分支直接返回旧记录，跳过了 `_apply_dims_to_result`
    中的硬伤封顶与 clamp 回填。旧 checkpoint 的 `person_is_subject` 全为 None、
    `vlm_clamp` 全为 False，即使修复了判定逻辑也不会生效——实测 28 张同时满足
    "脸占比>10% + clarity<10% + VLM 自报人脸模糊" 的照片，0 张被钳制。
    本函数保证任何恢复路径都重算派生态，使逻辑修复能作用于历史数据。
    """
    from core.layer3_aesthetic import (
        _decide_vlm_clamp,
        _normalize_vlm_batch,
        apply_hard_flaw_caps,
        clamp_hard_flaws_after_normalize,
    )

    for r in aes_results:
        if r.error:
            continue
        r.capped_dims = apply_hard_flaw_caps(r)
    # VLM 分数按当前开关重算（vlm_raw_score 已固化则幂等；旧记录无 raw 时以当前值为原始分）
    _normalize_vlm_batch(aes_results)
    # 钳制回填：用 L4 当前触发源 + 已存字段重算，不再只信任 checkpoint 里的旧 vlm_clamp
    for r in aes_results:
        if r.vlm_overall_score <= 0:
            continue
        clamp, source = _decide_vlm_clamp(
            face_hard_flags.get(r.path),
            r.person_is_subject,
            r.hard_flaws,
        )
        r.vlm_clamp = clamp
        r.clamp_source = source
    clamp_hard_flaws_after_normalize(aes_results)


def _stage_vlm(checkpoint: dict | None, rerun: set, incremental: bool, scoring_targets: list[str],
    need_topiq: bool, aes_topiq_results: list[Layer3Result], representatives: list[str],
    face_results: list[Layer4Result], face_hard_flags: dict, vlm_tasks: list,
    api_key: str, output_dir: Path, progress_callback, skipped_layers: list[str]):
    """Layer 3 审美评分：TOPIQ 重跑复用 / checkpoint 恢复 / 全量 VLM 评分（三分支）

    返回 (aes_results, skipped_layers)。"""
    from core.layer3_aesthetic import score_topiq
    from core.layer4_face import to_face_prompt_summary

    # ── Layer 3: 审美评分（TOPIQ 已与 L4 并发完成，VLM API 已在 L4 回调阶段逐张发起）──
    rerun_vlm = "vlm" in rerun
    aes_ckpt_ok = bool(checkpoint and checkpoint["completed_layer"] >= 4 and checkpoint.get("aes_results"))
    if incremental:
        # 增量：有新代表才需要走 VLM 处理（旧代表沿用 checkpoint 分）
        need_vlm = bool(scoring_targets)
    else:
        need_vlm = not (aes_ckpt_ok and not rerun_vlm)

    if not need_vlm and need_topiq:
        # TOPIQ-only 增量重跑：新 TOPIQ 分覆盖到 checkpoint 结果上，VLM 分直接复用
        aes_results = [Layer3Result(**r) for r in checkpoint["aes_results"]]
        new_topiq_by_path = {r.path: r for r in aes_topiq_results if not r.error}
        replaced = 0
        for r in aes_results:
            nr = new_topiq_by_path.get(r.path)
            if nr is None:
                continue  # 新 TOPIQ 失败/缺失 → 保留 checkpoint 旧分
            r.aesthetic_score = nr.aesthetic_score
            r.technical_score = nr.technical_score
            r.overall_score = nr.overall_score
            r.final_score = nr.final_score
            r.device = nr.device
            replaced += 1
        aes_results.sort(key=lambda x: x.overall_score, reverse=True)
        # TOPIQ 全量统一重归一化（同增量合并路径）：新/旧分参照系可能不同，raw 齐备才重算
        if all(r.topiq_raw_score > 0 for r in aes_results if not r.error and r.overall_score > 0):
            from core.layer3_aesthetic import _normalize_batch_scores
            _normalize_batch_scores(aes_results)
            aes_results.sort(key=lambda x: x.overall_score, reverse=True)
        skipped_layers = checkpoint.get("skipped_layers", skipped_layers)
        logger.info(f"[L3] TOPIQ 重跑完成（替换 {replaced}/{len(aes_results)} 张），VLM 分复用 checkpoint")
        _reapply_derived_scores(aes_results, face_hard_flags)
        _report_scoring_health(aes_results, "照片榜(TOPIQ重跑)")
        _save_layer_checkpoint(output_dir, 4, {
            "aes_results": [r.model_dump() for r in aes_results],
            "skipped_layers": skipped_layers,
        })
    elif not need_vlm:
        aes_results = [Layer3Result(**r) for r in checkpoint["aes_results"]]
        skipped_layers = checkpoint.get("skipped_layers", skipped_layers)
        # 恢复路径同样重算派生态 + 健康报警，与 VLM 重评分支对齐（杜绝旧 checkpoint 静默跳过封顶/钳制）
        _reapply_derived_scores(aes_results, face_hard_flags)
        aes_results.sort(key=lambda x: x.overall_score, reverse=True)
        _report_scoring_health(aes_results, "照片榜(checkpoint恢复)")
        logger.info(f"[L3] 从 checkpoint 恢复: {len(aes_results)} 张")
    else:
        _emit(progress_callback, "layer_start", layer=4, layer_name="审美评分")
        try:
            if not aes_topiq_results:
                aes_topiq_results = score_topiq(representatives, "auto")
            if api_key:
                # 图片级流水：等待 VLM 任务完成并写入（任务已在 L4 回调阶段发起）
                if not vlm_tasks:
                    # 续传/降级路径：L4 从 checkpoint 恢复或 L4 失败，VLM 未提前发起 → 补齐逐张发起
                    from concurrent.futures import ThreadPoolExecutor as _TPE

                    from core.layer3_aesthetic import _VLM_CONCURRENCY, _run_dimension_analysis
                    face_by_path = {r.path: r for r in face_results}
                    # 增量模式：兜底也只对新增代表发起（旧代表严禁重跑 VLM）
                    vlm_scope = aes_topiq_results
                    if incremental:
                        new_rep_set = set(scoring_targets)
                        vlm_scope = [r for r in aes_topiq_results if r.path in new_rep_set]
                    with _TPE(max_workers=_VLM_CONCURRENCY) as vp:
                        for r in vlm_scope:
                            if r.error or r.path not in face_by_path:
                                continue
                            fr = face_by_path[r.path]
                            vlm_tasks.append((
                                r.path,
                                vp.submit(
                                    _run_dimension_analysis, r.path,
                                    face_summary=to_face_prompt_summary(fr),
                                ),
                                _face_hard_flag(fr),
                            ))
                from core.layer3_aesthetic import (
                    _apply_dims_to_result,
                    _decide_vlm_clamp,
                    _is_saturated,
                    _merge_dims,
                    _normalize_vlm_batch,
                    _run_dimension_analysis,
                    clamp_hard_flaws_after_normalize,
                )
                topiq_by_path = {r.path: r for r in aes_topiq_results}
                face_by_path = {r.path: r for r in face_results}
                for path, fut, hflag in vlm_tasks:
                    r = topiq_by_path.get(path)
                    if r is None or r.error:
                        continue  # TOPIQ 失败的照片不写 VLM 分
                    try:
                        dims = fut.result()
                        if VLM_SATURATION_RESAMPLE and _is_saturated(dims):
                            # 饱和重采样（默认关）：追加一次评分取均值，缓解锚点打分
                            fr = face_by_path.get(path)
                            fsum = to_face_prompt_summary(fr) if fr is not None else None
                            try:
                                dims = _merge_dims(dims, _run_dimension_analysis(path, face_summary=fsum))
                            except Exception as e:
                                logger.warning(f"饱和重采样失败 {r.filename}: {e}（采用首次评分）")
                    except Exception as e:
                        logger.warning(f"维度分析失败 {r.filename}: {e}")
                        dims = {}
                    _apply_dims_to_result(r, dims, hard_flag=hflag)
                # 全量统一重归一化：增量时旧/新 VLM 分可能来自不同批次的 p5-p95 参照系，
                # 基于 vlm_raw_score（记录内保留的归一化前原始分）幂等重算，消除跨批次不可比
                _normalize_vlm_batch(aes_topiq_results)
                # checkpoint 恢复的旧 VLM 记录无 vlm_clamp 字段（默认 False），
                # 用 L4 触发源 + 已存 person_is_subject/hard_flaws 回填，再统一钳制
                for r in aes_topiq_results:
                    if r.vlm_overall_score > 0 and not r.vlm_clamp:
                        clamp, source = _decide_vlm_clamp(
                            face_hard_flags.get(r.path),
                            r.person_is_subject,
                            r.hard_flaws,
                        )
                        r.vlm_clamp = clamp
                        # 回填路径审计：带 backfill: 前缀区分于当场裁决
                        r.clamp_source = f"backfill:{source}" if clamp else ""
                clamp_hard_flaws_after_normalize(aes_topiq_results)
            else:
                # 无 API key：VLM 维度默认值 5.0（与 apply_vlm_dimensions 行为一致）
                for result in aes_topiq_results:
                    result.composition_score = 5.0
                    result.color_score = 5.0
                    result.lighting_score = 5.0
            aes_results = aes_topiq_results
            aes_results.sort(key=lambda x: x.overall_score, reverse=True)
            _report_scoring_health(aes_results, "照片榜")
        except Exception as e:
            logger.error(f"[L3] 失败，跳过: {e}")
            aes_results = [Layer3Result(path=p, filename=Path(p).name) for p in representatives]
            skipped_layers.append("Layer 3")
        _emit(progress_callback, "layer_end", layer=4, scored=len(aes_results))
        _save_layer_checkpoint(output_dir, 4, {
            "aes_results": [r.model_dump() for r in aes_results],
            "skipped_layers": skipped_layers,
        })

    return aes_results, skipped_layers


def _scene_bucket(path: str) -> tuple[str, int] | None:
    """拍摄场景桶 =（拍摄日期, 日内时间片序号）；无法解析返回 None（不参与配额）

    依据：实测 Top100 中 13 个 30 分钟桶 ≥3 张、最大单桶 11 张，而标准 64-bit pHash
    在 Top100 内只识别出 1 对相似——pHash 无法表达"同场景不同构图"，只能靠拍摄时间约束。
    兼容 IMG_20260713_104947.HEIC / MVIMG_20260713_110423.jpg / 20260719_131006_895.jpg。
    """
    m = _PHOTO_TS_RE.search(Path(path).name)
    if not m:
        return None
    date, hh, mm = m.group(1), int(m.group(2)), int(m.group(3))
    return date, (hh * 60 + mm) // SCENE_BUCKET_MINUTES


_PHOTO_TS_RE = re.compile(r"(20\d{6})[_-](\d{2})(\d{2})")


def _greedy_by_bucket(ranked: list[Layer3Result], k: int, cap: int):
    """按分数降序贪心选取，同一场景桶不超过 cap 张；返回 (选中, 被配额拦下的)"""
    chosen: list[Layer3Result] = []
    used: Counter = Counter()
    blocked: list[Layer3Result] = []
    for r in ranked:
        b = _scene_bucket(r.path)
        if b is not None and used[b] >= cap:
            blocked.append(r)
            continue
        chosen.append(r)
        if b is not None:
            used[b] += 1
        if len(chosen) >= k:
            break
    return chosen, blocked


def _apply_category_quota(chosen: list[Layer3Result], ranked: list[Layer3Result],
                          score_map: dict[str, float], k: int) -> list[Layer3Result]:
    """类别保底：缺口从已选中最低分的非保底类照片置换，不增加总量

    实测人像入榜率 7.2%、动物 3.8%，远低于风景 14.1%，属系统性失衡；
    旅行照里"人"最可能被发出来，故对人像/人文纪实给最低配额。

    两条硬约束，防止"为凑数塞烂片"与"保底绕过多样性"：
    1) 质量地板：候选融合分需 ≥ 置换前入选最低分 × CATEGORY_QUOTA_MIN_SCORE_RATIO；
    2) 场景配额：置换后同桶不得超过 SCENE_BUCKET_MAX（不再借用放宽上限，
       否则保底会把某场景堆到 3 张而审计仍报告 relax_level=0）。
    """
    if not CATEGORY_MIN_RATIO or not chosen:
        return chosen
    # 地板基准在置换前一次算定，避免逐次置换后最低分下滑、地板跟着塌陷
    floor = min(score_map.get(r.path, 0.0) for r in chosen) * CATEGORY_QUOTA_MIN_SCORE_RATIO
    chosen_paths = {r.path for r in chosen}
    bucket_used: Counter = Counter()
    for r in chosen:
        b = _scene_bucket(r.path)
        if b is not None:
            bucket_used[b] += 1

    for cat, ratio in CATEGORY_MIN_RATIO.items():
        need = int(round(k * ratio))
        have = sum(1 for r in chosen if (r.category or "") == cat)
        lack = need - have
        if lack <= 0:
            continue
        pool = [r for r in ranked
                if (r.category or "") == cat and r.path not in chosen_paths
                and score_map.get(r.path, 0.0) >= floor]
        added = 0
        for cand in pool:
            if added >= lack:
                break
            # 从尾部找分数最低的非保底类作为被置换者
            victim_idx = None
            for i in range(len(chosen) - 1, -1, -1):
                if (chosen[i].category or "") in CATEGORY_MIN_RATIO:
                    continue
                victim_idx = i
                break
            if victim_idx is None:
                break
            victim = chosen[victim_idx]
            cb, vb = _scene_bucket(cand.path), _scene_bucket(victim.path)
            # 严格场景配额：同桶置换净值不变，跨桶置换才 +1
            if cb is not None and cb != vb and bucket_used[cb] >= SCENE_BUCKET_MAX:
                continue
            if vb is not None:
                bucket_used[vb] -= 1
            chosen_paths.discard(victim.path)
            chosen_paths.add(cand.path)
            chosen[victim_idx] = cand
            if cb is not None:
                bucket_used[cb] += 1
            added += 1
        if added:
            logger.info(f"[类别保底] {cat} 补 {added} 张（目标 {need}，原有 {have}）")

    chosen.sort(key=lambda x: score_map.get(x.path, 0.0), reverse=True)
    return chosen


def _ensure_top_segment(chosen: list[Layer3Result], ranked: list[Layer3Result],
                        score_map: dict[str, float], seg_size: int):
    """前段保底：榜单前 seg_size 张（平台发帖量，用户实际会发的第一屏）内，
    每个保底类别至少 1 张；返回 (新榜单, 置换次数)

    实测：Top100 人像 10 张，但 Top12 人像 0 张——全文配额对用户第一眼看到的
    结果完全不可见。这里只替换"前段最低分的非保底类照片"，且仍受场景桶严格
    配额与质量地板约束，因此代价可控。
    注意：不重新排序。保底照片分数天然低于前段均值，重排会把它挤出前段，保底即失效。
    """
    if not TOP_SEGMENT_ENSURE_CATEGORY or not CATEGORY_MIN_RATIO:
        return chosen, 0
    if seg_size <= 0 or len(chosen) <= seg_size:
        return chosen, 0

    head = chosen[:seg_size]
    tail = chosen[seg_size:]
    head_paths = {r.path for r in head}
    floor = min(score_map.get(r.path, 0.0) for r in head) * CATEGORY_QUOTA_MIN_SCORE_RATIO
    buckets = Counter(_scene_bucket(r.path) for r in chosen if _scene_bucket(r.path) is not None)
    changes = 0

    for cat in CATEGORY_MIN_RATIO:
        if any((r.category or "") == cat for r in head):
            continue
        # 候选优先取自榜尾（榜内移动不改变桶计数，必不破配额），其次榜外高分候选
        in_list = [r for r in tail if (r.category or "") == cat]
        tail_paths = {r.path for r in tail}
        off_list = [r for r in ranked
                    if (r.category or "") == cat and r.path not in head_paths
                    and r.path not in tail_paths]
        pool = sorted(in_list + off_list,
                      key=lambda r: score_map.get(r.path, 0.0), reverse=True)
        pool = [r for r in pool if score_map.get(r.path, 0.0) >= floor]
        if not pool:
            continue

        # 被置换者：前段中分数最低、且不属于任何保底类别
        victim_idx = None
        for i in range(len(head) - 1, -1, -1):
            if (head[i].category or "") in CATEGORY_MIN_RATIO:
                continue
            victim_idx = i
            break
        if victim_idx is None:
            continue
        victim = head[victim_idx]
        vb = _scene_bucket(victim.path)

        placed = None
        for cand in pool:
            cb = _scene_bucket(cand.path)
            # 榜内候选（在榜尾）移动位置不改变桶计数；同桶置换净值也不变；
            # 只有榜外候选新进榜才占用一个新名额
            already = cand.path in tail_paths
            if cb is not None and not already and cb != vb and buckets[cb] >= SCENE_BUCKET_MAX:
                continue
            placed = cand
            break
        if placed is None:
            continue

        already = placed.path in tail_paths
        cb = _scene_bucket(placed.path)
        tail = [r for r in tail if r.path != placed.path]
        head[victim_idx] = placed
        if already:
            # 榜内置换：placed 从榜尾移入前段，被挤出的 victim 回填榜尾，总量守恒
            tail = sorted(tail + [victim], key=lambda r: score_map.get(r.path, 0.0), reverse=True)
        if vb is not None:
            buckets[vb] -= 1
        if cb is not None and not already:
            buckets[cb] += 1
        changes += 1
        logger.info(f"[前段保底] {cat} 进入前 {seg_size} 张，替换 {victim.filename}")

    return head + tail, changes


def _select_with_quota(ranked: list[Layer3Result], score_map: dict[str, float], k: int,
                       top_segment: int = 0):
    """选片：场景桶硬配额（逐级放宽）→ 类别保底置换 → 前段保底；返回 (top, 审计信息)"""
    chosen: list[Layer3Result] = []
    relax = 0
    for cap in (SCENE_BUCKET_MAX, SCENE_BUCKET_MAX_RELAXED, 10 ** 9):
        chosen, blocked = _greedy_by_bucket(ranked, k, cap)
        if len(chosen) >= k:
            break
        relax += 1
    # 仍不足（k 超过候选总数）：按分数补齐
    if len(chosen) < k:
        have = {r.path for r in chosen}
        for r in ranked:
            if len(chosen) >= k:
                break
            if r.path not in have:
                chosen.append(r)
                have.add(r.path)

    chosen = _apply_category_quota(chosen, ranked, score_map, k)
    chosen, top_changes = _ensure_top_segment(chosen, ranked, score_map, top_segment)

    buckets = Counter(_scene_bucket(r.path) for r in chosen if _scene_bucket(r.path) is not None)
    audit = {
        "scene_bucket_max": SCENE_BUCKET_MAX,
        "relax_level": relax,
        "scene_bucket_count": len(buckets),
        "scene_bucket_peak": max(buckets.values()) if buckets else 0,
        "category_counts": dict(Counter((r.category or "未分类") for r in chosen)),
        "top_segment": top_segment,
        "top_segment_changes": top_changes,
        "top_segment_category_counts": dict(
            Counter((r.category or "未分类") for r in chosen[:top_segment])
        ) if top_segment else {},
    }
    return chosen, audit


def _percentile_of(values: list[float], q: float) -> float:
    """线性插值分位数（与 numpy.percentile 默认算法同构，免引入 numpy 依赖）"""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _assign_buckets(scores: dict[str, float], selected_paths: set[str]) -> dict:
    """三桶分级（精选 / 可用 / 淘汰），返回全部候选的分桶结果与门槛

    门槛 = max(批次分位数, 固定底线)，取严：
    - 精选：final_score >= max(P85, 7.0)
    - 淘汰：final_score <  max(P20, 4.0)
    - 可用：其余

    两道闸都要生效：分位数吸收批次难度（整批都好片时不至于全进精选），
    固定底线保住绝对语义（整批都废片时不会因“相对较好”而进精选）。

    两条现实约束：
    1. 已入选 top_photos 的照片永不标淘汰——类别保底会特意选低分照片冲配额，
       标成淘汰会与“入选”自相矛盾，降级为可用并置 promoted=True 留痕；
    2. 分位数用全候选集而非已选子集算，否则门槛会随配额结果漂移。
    """
    vals = list(scores.values())
    if not vals:
        return {"map": {}, "thresholds": {}, "counts": {}}

    p_select = _percentile_of(vals, BUCKET_SELECT_PERCENTILE)
    p_reject = _percentile_of(vals, BUCKET_REJECT_PERCENTILE)
    # 取 max：两个门槛都是“数字越大越严”
    t_select = max(p_select, BUCKET_SELECT_FLOOR)
    t_reject = max(p_reject, BUCKET_REJECT_FLOOR)
    # 异常保护：淘汰门槛不得反超精选门槛（权重配置异常时强制拉开）
    if t_reject >= t_select:
        t_reject = min(t_reject, t_select - 1e-6)

    ordered = sorted(vals)
    n = len(ordered)
    out: dict[str, dict] = {}
    counts = {"select": 0, "maybe": 0, "reject": 0}
    promoted = 0
    for path, s in scores.items():
        # 百分位：低于该分数的占比（并列取中位位置，避免名次抖动）
        lo = _bisect_left(ordered, s)
        hi = _bisect_right(ordered, s)
        pct = round((lo + hi) / 2 / n * 100, 2)
        if s >= t_select:
            bucket, is_promoted = "select", False
        elif s < t_reject:
            if path in selected_paths:
                bucket, is_promoted = "maybe", True
                promoted += 1
            else:
                bucket, is_promoted = "reject", False
        else:
            bucket, is_promoted = "maybe", False
        counts[bucket] += 1
        out[path] = {"bucket": bucket, "percentile": pct, "promoted": is_promoted}

    return {
        "map": out,
        "thresholds": {
            "select": round(t_select, 4),
            "reject": round(t_reject, 4),
            "select_p85": round(p_select, 4),
            "reject_p20": round(p_reject, 4),
            "select_floor": BUCKET_SELECT_FLOOR,
            "reject_floor": BUCKET_REJECT_FLOOR,
        },
        "counts": {**counts, "promoted": promoted},
    }


def _fuse_and_rank(aes_results: list[Layer3Result], phash_map: dict, face_scores: dict,
    l1_scores: dict, platform_config: dict):
    """融合评分：VLM 多维审美 + TOPIQ + 人脸 + 客观指标统一加权 + 多样性配额选片

    返回 (score_map, top_photos, top_n, weights, dim_w_sum, select_audit)。"""
    # ── 融合评分：VLM 多维审美 + TOPIQ + 人脸 + 客观指标统一加权 ──
    # 说明：VLM 评分失败/未配置 API 时，q_vlm 回退为 TOPIQ 分，保证仍可排序。
    weights = platform_config.get("weights", {
        "composition": 0.35, "color": 0.30, "lighting": 0.20, "face": 0.15
    })
    dim_w_sum = (weights.get("composition", 0.35) + weights.get("color", 0.30)
                 + weights.get("lighting", 0.20))

    # 融合分计算复用模块级函数（与 run_video_ranking 同源）

    # 计算融合分
    score_map: dict[str, float] = {r.path: _compute_final_score(r, weights, dim_w_sum, face_scores, l1_scores) for r in aes_results}

    # 先排序，再在配额约束下选片（必须按此顺序：旧的"排序前扣分"逻辑中
    # "penalize j = 排名靠后" 是伪命题，J 尚未排序；且 DIVERSITY_DISTANCE_THRESHOLD(10)
    # 小于 HAMMING_THRESHOLD(16)，候选对已在 L2 被合并，惩罚循环实际空转）
    aes_results.sort(key=lambda x: score_map[x.path], reverse=True)
    top_n = min(FINAL_OUTPUT_COUNT, len(aes_results))
    # 前段 = 平台发帖量（小红书/朋友圈 12 张）：用户实际只会发/看这么多
    top_segment = int(platform_config.get("output_count", 12))
    top_photos, select_audit = _select_with_quota(
        aes_results, score_map, top_n, top_segment=top_segment)
    logger.info(
        f"[多样性] 场景配额 {select_audit['scene_bucket_max']} 张/桶（放宽级别 "
        f"{select_audit['relax_level']}）→ 覆盖 {select_audit['scene_bucket_count']} 个场景桶，"
        f"最大单桶 {select_audit['scene_bucket_peak']} 张；类别分布 {select_audit['category_counts']}"
    )
    logger.info(
        f"[前段] 前 {top_segment} 张类别分布 {select_audit['top_segment_category_counts']}"
        f"（保底置换 {select_audit['top_segment_changes']} 张）"
    )

    logger.info(f"[终审] 平台: {platform_config['name']}, 候选: {len(aes_results)}, 入选: {len(top_photos)}")

    # 三桶分级：精选/可用/淘汰（门槛 = max(批次分位数, 固定底线)）
    bucket_info = _assign_buckets(score_map, {r.path for r in top_photos})
    if bucket_info["thresholds"]:
        t = bucket_info["thresholds"]
        c = bucket_info["counts"]
        logger.info(
            f"[三桶] 精选 {c['select']} / 可用 {c['maybe']} / 淘汰 {c['reject']}"
            f"（门槛 精选≥{t['select']:.2f} [P85={t['select_p85']:.2f} 底线{t['select_floor']}]，"
            f"淘汰<{t['reject']:.2f} [P20={t['reject_p20']:.2f} 底线{t['reject_floor']}]"
            + (f"，已入选提级 {c['promoted']}" if c.get("promoted") else "")
            + "）"
        )
    select_audit = {
        **select_audit,
        "bucket_thresholds": bucket_info["thresholds"],
        "bucket_counts": bucket_info["counts"],
        "bucket_map": bucket_info["map"],
    }
    logger.info(
        f"[终审] 融合权重: VLM{FUSION_WEIGHTS['vlm']} TOPIQ{FUSION_WEIGHTS['topiq']} "
        f"人脸{FUSION_WEIGHTS['face']}（VLM 评分时归零） 客观{FUSION_WEIGHTS['objective']} | "
        f"平台维度权重: 构图{weights.get('composition', 0.35)} "
        f"色彩{weights.get('color', 0.30)} "
        f"光影{weights.get('lighting', 0.20)} "
        f"人脸{weights.get('face', 0.15)}"
    )

    return score_map, top_photos, top_n, weights, dim_w_sum, select_audit


def _save_outputs(output_dir: Path, platform: str, prompt_hash: str, photos: list[Path],
    rejected: list, merged: int, top_n: int, skipped_layers: list[str],
    aes_results: list[Layer3Result], top_photos: list[Layer3Result],
    score_map: dict[str, float], weights: dict, dim_w_sum: float,
    select_audit: dict | None = None):
    """保存榜单产物（ranking/full_ranking/run_meta + 展示照片复制）并汇总"""
    # ── 保存结果 ──
    output_dir.mkdir(parents=True, exist_ok=True)

    # 清理旧输出（保留 checkpoint 在最终清理时删除）
    for old in output_dir.glob("*"):
        if old.is_file() and old.name != "checkpoint.json":
            old.unlink()

    # 三桶分级明细（每位候选的桶 + 批次百分位），排行榜与完整榜共用
    bucket_map = (select_audit or {}).get("bucket_map", {})

    # 合并结果并保存
    final_results: list[FinalResult] = []
    for r in top_photos:
        info = bucket_map.get(r.path, {})
        final_results.append(FinalResult(
            path=r.path,
            filename=r.filename,
            aesthetic_score=r.aesthetic_score,
            overall_score=r.overall_score,
            final_score=score_map[r.path],  # 融合综合分（0-10），已过场景/类别配额选片
            category=r.category,  # L3 VLM 分类标签（风景/人像/夜景等）
            bucket=info.get("bucket", ""),
            percentile=info.get("percentile", 0.0),
        ))

    # 复制入选照片（序号名前缀，供前端展示）
    # 必须在写榜单 JSON 之前：榜单里的 filename 要指向磁盘上真实存在的展示副本，
    # 否则前端 <img src="/api/results/.../{filename}"> 会对原名 404。
    display_names: dict[str, str] = {}
    for i, r in enumerate(top_photos):
        src = Path(r.path)
        dst = output_dir / f"{i+1:02d}_{src.name}"
        shutil.copy2(src, dst)
        display_names[r.path] = dst.name
    for fr in final_results:
        fr.filename = display_names.get(fr.path, fr.filename)

    ranking_file = output_dir / "ranking.json"
    with open(ranking_file, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in final_results], f, ensure_ascii=False, indent=2)

    # 保存全部评分（完整排行，不仅仅是 Top N）
    full_ranking = [
        {
            "path": r.path,
            # 入选者有展示副本名，未入选者保留原名
            "filename": display_names.get(r.path, r.filename),
            "aesthetic_score": r.aesthetic_score,
            "overall_score": r.overall_score,
            "final_score": score_map[r.path],
            # 三桶分级：精选/可用/淘汰 + 批次百分位（跨批次可比的相对位次）
            "bucket": bucket_map.get(r.path, {}).get("bucket", ""),
            "percentile": bucket_map.get(r.path, {}).get("percentile", 0.0),
            "bucket_promoted": bucket_map.get(r.path, {}).get("promoted", False),
            "category": r.category,
            "vlm_overall_score": r.vlm_overall_score,
            # 归一化前原始分（增量全量重归一化的参照，取证用）
            "vlm_raw_score": r.vlm_raw_score,
            "topiq_raw_score": r.topiq_raw_score,
            "composition_score": r.composition_score,
            "color_score": r.color_score,
            "lighting_score": r.lighting_score,
            # 审计三件套：钳制决策 + 规则路径 + VLM 主体裁断（跑后取证用）
            "vlm_clamp": r.vlm_clamp,
            "clamp_source": r.clamp_source,
            "person_is_subject": r.person_is_subject,
            "vlm_saturated": r.vlm_saturated,
            "hard_flaws": r.hard_flaws,
            # 硬伤封顶审计 + VLM 评语（前端展示"为什么入选/落选"，历史版本只存分数不存理由）
            "capped_dims": r.capped_dims,
            "score_reason": r.score_reason,
            "reason_source": r.reason_source,
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

    # 跑批元信息（与 checkpoint meta 同源）：最终产物可追溯 prompt/模型版本
    # 剔除 bucket_map（逐张明细已在 full_ranking 里，此处只留门槛与计数）
    audit_light = {k: v for k, v in (select_audit or {}).items() if k != "bucket_map"}
    run_meta = {
        "prompt_hash": prompt_hash,
        "model": QWEN_MODEL,
        "platform": platform,
        "photo_count": len(full_ranking),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        # 选片审计：场景配额放宽级别 + 覆盖桶数 + 最大单桶 + 类别分布 + 三桶门槛
        "selection": audit_light,
    }
    with open(output_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

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
    # 超发收敛：按"占比偏差最大者"逐减（偏差大 = 分多了）
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
    """视频帧独立竞争排名（双榜单中的"视频精选榜"）

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
    # 视频榜产物独立子目录（与照片榜物理分离：output/{platform}/videos/）
    output_dir = OUTPUT_DIR / platform / "videos"

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
    face_hard_flags = {r.path: _face_hard_flag(r) for r in face_results}
    try:
        aes_results = apply_vlm_dimensions(
            aes_topiq_results,
            face_summaries=face_summaries,
            face_hard_flags=face_hard_flags,
        )
        aes_results.sort(key=lambda x: x.overall_score, reverse=True)
        _report_scoring_health(aes_results, "视频榜")
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
    # 先清理上次运行的全部产物（展示副本 + 榜单 JSON），避免 vNN_ 序号跨运行累积
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*"):
        if old.is_file():
            old.unlink()
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

    # 跑批元信息（与照片榜 run_meta.json 同源）：视频榜产物同样可追溯 prompt/模型版本
    from core.layer3_aesthetic import vlm_prompt_fingerprint
    video_run_meta = {
        "prompt_hash": vlm_prompt_fingerprint(),
        "model": QWEN_MODEL,
        "platform": platform,
        "frame_count": len(full),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(output_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(video_run_meta, f, ensure_ascii=False, indent=2)

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
