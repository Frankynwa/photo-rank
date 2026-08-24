"""Layer 3: 审美评分 — TOPIQ-IAA（批次归一化）+ VLM 多维度专业评分（rubric + 重试 + 校验）"""
import asyncio
import base64
import concurrent.futures
import gc
import hashlib
import io
import json
import logging
import os
import re
import threading
from pathlib import Path

from PIL import Image

from config import VLM_IMAGE_MAX_SIDE, VLM_JPEG_QUALITY, VLM_SATURATION_RESAMPLE
from core.models import Layer3Result

logger = logging.getLogger(__name__)

# VLM 维度评分参数
_VLM_TIMEOUT = 60  # 单次 API 超时（秒）
_VLM_RETRIES = 3  # 失败重试次数（指数退避）
_VLM_CONCURRENCY = 8  # VLM 维度分析并发数（网络 IO 密集，图片级流水）
_VLM_DIM_KEYS = ("composition_score", "color_score", "lighting_score", "overall_aesthetic_score")

# TOPIQ 模型模块级缓存：{实际device: (model, pyiqa模块引用)}
# - 按 device 缓存避免每次运行重复加载（实测加载 ~4s，15 张纯推理仅 ~2s），
#   照片榜/视频榜/多平台重跑共用同一缓存
# - 缓存同时保存 pyiqa 模块引用，通过 `is` 身份校验区分真实模块与测试 mock：
#   测试中 patch.dict(sys.modules) 替换 pyiqa 时缓存自动失效，不会误复用 mock 模型
_topiq_model_cache: dict[str, tuple] = {}
_topiq_model_lock = threading.Lock()


def release_topiq_model(device: str | None = None) -> None:
    """释放 TOPIQ 模型缓存（显存分时加载时调用；device=None 释放全部）

    模型默认常驻缓存；接入 Q-Align/HumanAesExpert 等 GPU 模型需要腾显存时，
    可先调用本函数再加载新模型（与显存分时加载策略兼容）。
    """
    with _topiq_model_lock:
        keys = list(_topiq_model_cache) if device is None else [device]
        for k in keys:
            if k in _topiq_model_cache:
                _topiq_model_cache.pop(k)
                logger.info(f"已释放 TOPIQ 模型缓存: {k}")
        if "cuda" in keys:
            import torch
            torch.cuda.empty_cache()
        gc.collect()

# VLM 分类标签枚举（prompt 中的分类清单由本元组动态生成，单一来源防契约错位）
_CATEGORIES = ("风景", "建筑", "人像", "人文纪实", "美食", "动物", "夜景", "其他")

# 分类别名归一：模型输出同义/近义词时映射回标准枚举，避免丢弃成"未分类"
_CATEGORY_ALIASES = {
    "人文": "人文纪实", "纪实": "人文纪实", "街头": "人文纪实", "街头人文": "人文纪实",
    "人物": "人像", "夜景人像": "人像", "环境人像": "人像",
    "风光": "风景", "自然风光": "风景",
    "宠物": "动物", "野生动物": "动物",
    "食物": "美食",
    "夜色": "夜景",
}

# VLM 硬伤枚举白名单（prompt 中的硬伤清单由本元组动态生成，单一来源防契约错位）
_HARD_FLAW_ENUMS = ("模糊", "过曝", "欠曝", "噪点明显", "歪斜", "画面杂乱", "主体裁切", "逆光发黑", "地平线穿头", "闭眼", "人脸模糊")

# L4 "未检测到人脸"文案标记：命中时走瘦身版 prompt（省去人脸判读段落）
_NO_FACE_MARKER = "未检测到人脸"

# prompt 分段组装：_PROMPT_HEADER + （有人脸时）_FACE_SECTION + 硬伤预检 + _RUBRIC + _OUTPUT_REQ
# 无人脸照片走瘦身版（不含【主体判断】【合影细则】与闭眼相关规则，省 token 且防模型脑补人脸角色）
_PROMPT_HEADER = """你是从业 15 年的资深旅行摄影师兼选片师，为摄影比赛和社交平台评审过大量照片。
请以大众审美标准评审这张旅行照片。
"""

_FACE_SECTION = """
{FACE_INFO}

【主体判断】
以上人脸信息仅作客观参考，请结合画面实际内容自行裁断：
A. 人脸是拍摄主体（单人像/双人/合影）→ 按人像标准评分，人脸质量参与综合审美
B. 人脸只是环境背景（景区人群/观众席/街头行人/远处路人）→ 忽略人脸质量，聚焦真正的场景主体
C. 画面同时存在主体人物与环境人群（如舞台表演者与观众席）→ 只评估主体人物的人脸质量，观众忽略
判断依据：人脸大小与占比、人脸面积分布（均匀=合影；主次分明=单主体+背景）、人脸的位置与姿态。
无论何种情况，不要因为“人多”就判定为合影，一切以画面实际内容为准。

【合影评分细则】（仅当判定为合影/多人主体时启用）
- 硬伤：任一主体人物闭眼、明显表情失控 → 综合审美上限 6 分
- 清晰度：主体人脸均清晰为佳，边缘人物轻微模糊属正常，不苛求
- 构图：重点检查边缘人物被裁切、头顶间距过挤、地平线穿头/穿肩
- 表情协调：整体自然、有互动感、氛围愉悦为加分项；表情僵硬/各异为减分项
- 综合审美：合影侧重情感氛围与纪念价值，不苛求单张人像的精致度
"""

_HARD_FLAW_CHECK_FACE = """
【评审流程】
第一步，先在心里快速检查以下硬伤（不用输出）：
1. 画面模糊/失焦  2. 严重过曝或欠曝（死白/死黑）  3. 噪点明显
4. 水平线明显歪斜  5. 画面杂乱/多余元素入画  6. 主体被裁切
7. 逆光导致主体发黑  8. 地平线穿头/穿肩
9. 闭眼：以【人脸检测信息】中"主脸闭眼/存在闭眼者"标注为准（无需肉眼再判断）
存在任何硬伤：对应维度的分数上限为 4 分；闭眼作为人像硬伤时，综合审美上限 6 分。
人脸模糊不做硬性上限——清晰度问题请在对应维度分中自然体现，人脸模糊的钳制由系统基于人脸检测客观数据另行把关。
"""

_HARD_FLAW_CHECK_PLAIN = """
【评审流程】
第一步，先在心里快速检查以下硬伤（不用输出）：
1. 画面模糊/失焦  2. 严重过曝或欠曝（死白/死黑）  3. 噪点明显
4. 水平线明显歪斜  5. 画面杂乱/多余元素入画  6. 主体被裁切
7. 逆光导致主体发黑  8. 地平线穿头/穿肩
存在任何硬伤：对应维度的分数上限为 4 分。
"""

_RUBRIC = """
第二步，按以下标准给四个维度打分（1-10 分，允许一位小数）：

- composition_score 构图：
  9-10 构图一流：主体突出且位置考究，三分法/对称/引导线/框架等技法运用到位且有巧思，画面简洁有层次
  7-8  构图良好：主体清晰，画面干净，无明显硬伤
  5-6  构图平庸：没有明显错误，但也无亮点（典型随手拍）
  3-4  构图较差：画面杂乱、主体不明确、明显歪斜或裁切不当（或存在上述硬伤）
  1-2  构图失败：找不到主体，画面混乱

- color_score 色彩：
  9-10 色彩和谐且出彩：色调统一有氛围感，饱和度恰到好处，肤色自然
  7-8  色彩和谐悦目
  5-6  色彩一般，不和谐也不难看
  3-4  色彩杂乱，或明显发灰发闷，或明显偏色（或存在上述硬伤）
  1-2  严重偏色，色彩关系混乱

- lighting_score 光影：
  9-10 光线质感极佳：立体感强、氛围到位、曝光精准，明暗过渡自然
  7-8  光线良好，曝光合理
  5-6  光线平淡，无明显曝光问题
  3-4  光线不佳，轻微过曝/欠曝，或逆光主体发黑（或存在上述硬伤）
  1-2  曝光严重失败，死白或死黑

- overall_aesthetic_score 综合审美：你个人判断"大众会不会觉得这张照片好看"的整体分。
  不是以上三项的简单平均，需综合主体吸引力（景色震撼度/人物表情/瞬间捕捉）、
  情绪感染力（氛围/故事感）、画面稀有度。

【评分校准】
- 5 分 = 普通随手拍的平均水准
- 6-7 分 = 拍得不错的旅行照片，值得分享
- 8 分以上 = 专业级/封面水平，仅在画面确实达到该水准时给出，宁缺毋滥
- 4 分以下 = 有明显问题，一般不值得保留
"""

_OUTPUT_REQ = """
【输出要求】
- 先在心里完成评审流程，再简要说明打分理由（30-50 字），最后给分数。
- 分数精确到小数点后一位（如 7.3、8.6），根据实际水平细分，
  不要默认给 x.0/x.5 的整档分；不同照片的分数应体现差异，避免多张都落在同一锚点值。
- hard_flaws：从以下枚举中选你实际检测到的硬伤（可多个，无则填"无硬伤"）：
  {HARD_FLAW_ENUM}
  必须输出该字段，硬伤检查不可跳过；存在任何硬伤时对应维度上限 4 分（人脸模糊除外，其仅按自然减分处理）。
- 照片类型（category）从以下枚举中选一个：{CATEGORY_ENUM}。
  按画面中最突出的主体与场景判断，只选一个，不要输出枚举之外的词。
  主体与场景属性冲突时按主体归类（如夜晚拍摄的人像 → 人像）；夜景仅用于以夜景本身为主要观赏对象的照片。
- person_is_subject：true/false。画面中人物/人脸是否为拍摄主体（无人物/人脸时填 false）。
  墨镜/口罩/侧脸/虚化等遮挡不影响该判断，以画面实际内容为准。
- 人脸模糊：仅当你从画面中实际观察到主体人脸失焦/模糊时才在 hard_flaws 中报告，
  不要依据【人脸检测信息】中的清晰度指标推断。
- 只返回 JSON：{"reason": "理由", "hard_flaws": [...], "composition_score": x, "color_score": x, "lighting_score": x, "overall_aesthetic_score": x, "category": "分类", "person_is_subject": true}"""


def _build_dimension_prompt(face_summary: str | None) -> str:
    """组装 VLM 评分 prompt

    - 有人脸信息 → 完整版（含【主体判断】【合影细则】与闭眼规则）
    - 无人脸（L4 未检出 / L3 独立运行）→ 瘦身版（省去人脸判读段落，
      省 token 且防模型对无脸照片脑补人脸角色）
    - 硬伤/分类枚举由 _HARD_FLAW_ENUMS / _CATEGORIES 元组动态拼入，单一来源
    """
    has_face_info = bool(face_summary) and _NO_FACE_MARKER not in face_summary
    parts = [_PROMPT_HEADER]
    if has_face_info:
        parts.append(_FACE_SECTION.replace("{FACE_INFO}", face_summary))
        parts.append(_HARD_FLAW_CHECK_FACE)
    else:
        parts.append(_HARD_FLAW_CHECK_PLAIN)
    parts.append(_RUBRIC)
    parts.append(_OUTPUT_REQ)
    prompt = "".join(parts)
    prompt = prompt.replace("{HARD_FLAW_ENUM}", " / ".join(_HARD_FLAW_ENUMS))
    prompt = prompt.replace("{CATEGORY_ENUM}", " / ".join(_CATEGORIES))
    return prompt


def vlm_prompt_fingerprint() -> str:
    """VLM 评分配置指纹：静态 prompt 模板（含硬伤/分类枚举，不含逐照片
    注入的 FACE_INFO）+ 模型名的 sha1 前 16 位

    供 checkpoint 新鲜度校验：prompt 或模型变更后旧 VLM 分与新分不再
    可比（分数语义已变），pipeline 据此自动触发 VLM 重评，
    防止新旧 prompt 分数静默混排（历史教训：旧 checkpoint 无
    person_is_subject 字段被回填逻辑静默混入新规则）。
    """
    from config import QWEN_MODEL

    tpl = "".join([
        _PROMPT_HEADER, _FACE_SECTION, _HARD_FLAW_CHECK_FACE,
        _HARD_FLAW_CHECK_PLAIN, _RUBRIC, _OUTPUT_REQ,
    ])
    tpl = tpl.replace("{HARD_FLAW_ENUM}", " / ".join(_HARD_FLAW_ENUMS))
    tpl = tpl.replace("{CATEGORY_ENUM}", " / ".join(_CATEGORIES))
    return hashlib.sha1(f"{QWEN_MODEL}::{tpl}".encode("utf-8")).hexdigest()[:16]


def score_topiq(
    image_paths: list[str | Path],
    device: str = "auto",
) -> list[Layer3Result]:
    """TOPIQ-IAA 客观审美评分（逐张推理 + 批次归一化），不含 VLM 维度分析

    模型按 device 模块级缓存复用（实测加载 ~4s，缓存命中跳过）：
    - 缓存键为实际 device（"cuda"/"cpu"），并校验 pyiqa 模块身份，
      测试中 mock 替换 pyiqa 时缓存自动失效，不会误复用
    - 显存紧张时可调用 release_topiq_model() 主动释放（分时加载场景）

    Args:
        image_paths: 照片路径列表
        device: "auto" | "cuda" | "cpu"
    """
    import pyiqa
    import torch

    # 设备检测
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")

    if device == "cpu":
        logger.warning("CPU 推理，速度较慢")

    # 加载模型（带容错 + CPU fallback + 模块级缓存复用）
    model = None
    with _topiq_model_lock:
        cached = _topiq_model_cache.get(device)
        if cached is not None and cached[1] is pyiqa:
            model = cached[0]
            logger.info(f"TOPIQ 模型缓存命中（{device}）")
    if model is None:
        try:
            logger.info("加载 TOPIQ-IAA...")
            model = pyiqa.create_metric("topiq_iaa", device=device)
            with _topiq_model_lock:
                _topiq_model_cache[device] = (model, pyiqa)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            if device != "cpu":
                try:
                    logger.info("尝试 CPU fallback...")
                    model = pyiqa.create_metric("topiq_iaa", device="cpu")
                    device = "cpu"
                    with _topiq_model_lock:
                        _topiq_model_cache["cpu"] = (model, pyiqa)
                except Exception as e2:
                    logger.error(f"CPU fallback 也失败: {e2}")
                    return [
                        Layer3Result(
                            path=str(p), filename=Path(p).name,
                            error=str(e), device="cpu",
                        )
                        for p in image_paths
                    ]

    results = []
    total = len(image_paths)

    try:
        for i, path in enumerate(image_paths):
            try:
                from core.image_cache import get_cached_path
                img = Image.open(get_cached_path(path)).convert("RGB")
                # pyiqa topiq_iaa 经 dist_to_mos 输出已是 1-10 的 MOS 分（score_range='1,10'），
                # 直接使用即可；偶发超范围 → 钳制防分数体系崩坏
                score = max(0.0, min(10.0, model(img).item()))
                results.append(Layer3Result(
                    path=str(path), filename=Path(path).name,
                    aesthetic_score=round(score, 4),
                    technical_score=round(score, 4),
                    overall_score=round(score, 4),
                    final_score=round(score, 4),
                    device=device,
                ))
            except Exception as e:
                logger.warning(f"评分失败 {Path(path).name}: {e}")
                results.append(Layer3Result(
                    path=str(path), filename=Path(path).name,
                    error=str(e), device=device,
                ))

            if (i + 1) % 10 == 0:
                logger.info(f"评分进度: {i + 1}/{total}")

        logger.info(f"TOPIQ-IAA 完成（{device}）")
        if device == "cuda":
            logger.info(f"显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    finally:
        # 模型保留在模块级缓存中复用，不再 del/empty_cache；
        # 显存紧张时调用 release_topiq_model() 主动释放
        if device == "cuda":
            logger.debug("TOPIQ 模型常驻显存（模块级缓存），可调用 release_topiq_model() 释放")

    # TOPIQ 批次归一化：p5-p95 拉伸，拉开区分度
    _normalize_batch_scores(results)

    # TOPIQ 批次归一化后直接返回，VLM 维度分析由 apply_vlm_dimensions 独立完成
    return results


def clamp_hard_flaws_after_normalize(results: list[Layer3Result]) -> None:
    """归一化后对人像硬伤照片统一钳制 ≤6（单次惩罚）

    钳制判定由 _apply_dims_to_result 写入 result.vlm_clamp；
    必须在批次归一化之后调用——若先钳制再归一化，钳制值会被
    p5-p95 拉伸二次压分（实测原始 8 分好片被压到 2.63）。
    """
    for r in results:
        if r.vlm_overall_score > 6.0 and r.vlm_clamp:
            r.vlm_overall_score = 6.0
            if "已钳制至 6 分上限" not in r.score_reason:
                r.score_reason += "（人像硬伤，已钳制至 6 分上限）"


def _decide_vlm_clamp(hard_flag: "str | bool | None", subject: bool | None, flaws: list) -> tuple[bool, str]:
    """人像硬伤钳制判定（L4 客观触发源 + VLM 主体裁断），返回 (是否钳制, 规则路径)

    规则路径供审计（clamp_source）：blink_l4（L4 检出主脸闭眼）/ blink_legacy（遗留布尔）/
    blink_vlm（VLM 自报闭眼）/ blur_dual（模糊双证据）/ blur_fallback（subject 缺失回落旧逻辑）；
    未钳制返回空串。

    L4 无法判定"脸是否主体"与墨镜/侧脸等遮挡，VLM 裁断为准：
    - subject=False（人物/人脸非主体，含背景路人/误检假脸）→ 永不钳制
    - 闭眼路径（hard_flag="blink"/遗留 truthy 布尔，或 VLM 自报闭眼）→ 钳制
    - 清晰度路径（hard_flag="blur"）→ 双证据：VLM 从像素确认"人脸模糊"才钳制
      （豁免墨镜/侧脸/纹理误检）；VLM 未输出 person_is_subject（解析降级）时
      回落旧逻辑钳制，保住程序化兜底
    """
    if subject is False:
        return False, ""  # 人物/人脸非主体：L4 人脸数据不参与钳制
    vlm_blink = any("闭眼" in f for f in flaws)
    if hard_flag == "blur":
        # 清晰度触发需 VLM 像素级佐证；字段缺失时回落旧逻辑
        if subject is not None:
            return (True, "blur_dual") if any("人脸模糊" in f for f in flaws) else (False, "")
        return True, "blur_fallback"
    if hard_flag == "blink":
        return True, "blink_l4"  # L4 客观检出主脸闭眼
    if hard_flag:
        return True, "blink_legacy"  # 遗留 truthy 布尔（旧接口兼容）
    if vlm_blink:
        return True, "blink_vlm"  # VLM 自报闭眼（辅助信号）
    return False, ""


def _is_saturated(dims: dict) -> bool:
    """VLM 原始分饱和判定：综合分 ≥9.8，或构图/色彩/光影三项全 ≥9

    实证背景：801 张历史数据中 11.1% 拿满分、最大同分簇 29 张（理由各异但
    分数一致）——模型把 8~9 档当锚点，头部照片区分度崩坏。饱和旗标供
    跑后体检与可选重采样（VLM_SATURATION_RESAMPLE）使用，不改分。
    """
    if not dims or dims.get("overall_aesthetic_score", 0) <= 0:
        return False
    if dims["overall_aesthetic_score"] >= 9.8:
        return True
    trio = (dims.get("composition_score", 0), dims.get("color_score", 0), dims.get("lighting_score", 0))
    return all(v >= 9 for v in trio)


def _merge_dims(first: dict, second: dict) -> dict:
    """两次 VLM 评分取均值（饱和重采样）：分数取均值，非分数字段以首次为准"""
    if not second:
        return first
    out = dict(first)
    for k in _VLM_DIM_KEYS:
        a, b = first.get(k), second.get(k)
        if a is not None and b is not None:
            out[k] = round((a + b) / 2, 1)
    return out


def _apply_dims_to_result(result: Layer3Result, dims: dict, hard_flag: "str | bool" = False) -> None:
    """将 VLM 维度评分写入 Layer3Result（缺失字段回落默认值 5.0）

    hard_flag: L4 客观触发源——"blink"（主脸闭眼）/ "blur"（主脸清晰度低于阈值）/
    遗留 truthy 布尔 / False。

    钳制判定规则见 _decide_vlm_clamp；本函数不修改分数（F3），仅记录 vlm_clamp/
    clamp_source 与理由后缀，归一化后由 clamp_hard_flaws_after_normalize 统一钳制。
    """
    result.composition_score = round(dims.get("composition_score", 5.0), 4)
    result.color_score = round(dims.get("color_score", 5.0), 4)
    result.lighting_score = round(dims.get("lighting_score", 5.0), 4)
    if dims.get("hard_flaws"):
        result.hard_flaws = list(dims["hard_flaws"])
    result.vlm_saturated = _is_saturated(dims)
    if dims.get("overall_aesthetic_score", 0) > 0:
        score = round(dims["overall_aesthetic_score"], 4)
        subject = dims.get("person_is_subject")
        result.person_is_subject = subject if isinstance(subject, bool) else None
        flaws = dims.get("hard_flaws", [])
        clamp, source = _decide_vlm_clamp(hard_flag, result.person_is_subject, flaws)
        result.vlm_clamp = clamp
        result.clamp_source = source
        if clamp:
            reason = dims.get("reason", "")
            suffix = "（人像硬伤：闭眼/主脸模糊，已钳制至 6 分上限）"
            result.score_reason = f"{reason}{suffix}" if reason else suffix.strip("（）")
        else:
            result.score_reason = dims.get("reason", "")
        result.vlm_overall_score = score
    if dims.get("category"):
        result.category = dims["category"]


def apply_vlm_dimensions(
    results: list[Layer3Result],
    face_summaries: dict[str, str] | None = None,
    max_workers: int = _VLM_CONCURRENCY,
    face_hard_flags: dict[str, bool] | None = None,
) -> list[Layer3Result]:
    """对已有 TOPIQ 结果补 VLM 多维度评分（构图/色彩/光影/综合审美）

    - VLM 调用为网络 IO、逐张独立 → 线程池并发（图片级流水：先完成的先返回）
    - 无 QWEN_API_KEY 时全部使用默认值 5.0
    - 评分失败的照片（error 非空）不调用 API，维度默认 5.0
    - face_hard_flags: path -> L4 触发源（"blink"/"blur"/None），钳制判定结合 VLM 主体裁断
    """
    api_key = os.environ.get("QWEN_API_KEY", "")

    if api_key:
        logger.info("开始多维度审美分析（构图/色彩/光影/综合审美）...")
    else:
        logger.warning("未找到 QWEN_API_KEY，跳过多维度分析，使用默认值 5.0")

    targets = [r for r in results if not r.error and api_key]
    if targets:
        def _work(result: Layer3Result) -> tuple[Layer3Result, dict, bool]:
            try:
                face_summary = face_summaries.get(result.path) if face_summaries else None
                hard_flag = face_hard_flags.get(result.path) if face_hard_flags else False
                dims = _run_dimension_analysis(result.path, face_summary=face_summary)
                if VLM_SATURATION_RESAMPLE and _is_saturated(dims):
                    # 饱和重采样（默认关）：追加一次评分取均值，缓解锚点打分
                    try:
                        dims = _merge_dims(dims, _run_dimension_analysis(result.path, face_summary=face_summary))
                    except Exception as e:
                        logger.warning(f"饱和重采样失败 {result.filename}: {e}（采用首次评分）")
                return result, dims, hard_flag
            except Exception as e:
                logger.warning(f"维度分析失败 {result.filename}: {e}")
                return result, {}, False

        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_work, r) for r in targets]
            for future in concurrent.futures.as_completed(futures):
                result, dims, hard_flag = future.result()
                _apply_dims_to_result(result, dims, hard_flag=hard_flag)
                done += 1
                if done % 5 == 0:
                    logger.info(f"多维度分析进度: {done}/{len(targets)}")

    # 无 API key 或评分失败的照片：维度默认值 5.0
    for result in results:
        if result.error or not api_key:
            result.composition_score = 5.0
            result.color_score = 5.0
            result.lighting_score = 5.0

    if api_key:
        logger.info("多维度审美分析完成")
        # VLM 综合分批次归一化：拉开区分度、防分数膨胀（与 TOPIQ 归一化同构）
        _normalize_vlm_batch(results)
        # 归一化后对人像硬伤照片钳制 ≤6（单次惩罚，避免先钳制再归一化的二次压分）
        clamp_hard_flaws_after_normalize(results)
    return results


def _normalize_vlm_batch(results: list[Layer3Result]) -> None:
    """VLM 综合审美分批次归一化（p5-p95 拉伸到 0-10），防分数膨胀/缺乏区分度

    - 仅对真实 VLM 评分（vlm_overall_score > 0）的照片生效
    - 少于 2 张或分数分布集中时保持不变（避免噪声放大，与 TOPIQ 归一化一致）
    """
    valid = [r for r in results if r.vlm_overall_score > 0]
    if len(valid) < 2:
        return
    vals = sorted(r.vlm_overall_score for r in valid)
    n = len(vals)
    lo = vals[max(0, int(n * 0.05) - 1)]
    hi = vals[min(n - 1, int(n * 0.95) - 1)]
    span = hi - lo
    if span < 0.5:
        return
    for r in valid:
        norm = max(0.0, min(1.0, (r.vlm_overall_score - lo) / span))
        # 正下限 0.01：避免最低分归一化到精确 0.0，被全链路 "> 0 = 已评分"
        # 判定误判为 VLM 未评分（导致 face 权重不归零、q_vlm 错误回退 TOPIQ）
        r.vlm_overall_score = round(max(norm, 0.01) * 10, 4)


def score_photos(
    image_paths: list[str | Path],
    device: str = "auto",
    face_summaries: dict[str, str] | None = None,
) -> list[Layer3Result]:
    """评分照片（完整流程）：TOPIQ-IAA 客观评分 → VLM 多维度分析

    Args:
        image_paths: 照片路径列表
        device: "auto" | "cuda" | "cpu"
        face_summaries: path -> L4 人脸检测提示文案（见 to_face_prompt_summary），
                        注入 VLM prompt 的【人脸检测信息】段；为 None 或缺失时
                        使用中立占位文案，保证 L3 可独立运行。
    """
    results = score_topiq(image_paths, device=device)
    return apply_vlm_dimensions(results, face_summaries=face_summaries)


def _run_dimension_analysis(image_path: str, face_summary: str | None = None) -> dict:
    """同步调用 VLM 维度分析（兼容已有事件循环场景，用新线程兜底）"""
    from config import PROXY, QWEN_BASE_URL, QWEN_MODEL

    api_key = os.environ.get("QWEN_API_KEY", "")
    try:
        return asyncio.run(
            _analyze_dimensions(image_path, api_key, QWEN_BASE_URL, QWEN_MODEL, PROXY, face_summary=face_summary)
        )
    except RuntimeError:
        # 已有事件循环在运行，用新线程
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run,
                _analyze_dimensions(image_path, api_key, QWEN_BASE_URL, QWEN_MODEL, PROXY, face_summary=face_summary),
            ).result()


def _clamp_score(value) -> float | None:
    """将 VLM 分数钳制到 [1, 10]"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return max(1.0, min(10.0, v))


def _parse_dimension_json(text: str) -> dict:
    """从 VLM 回复中提取并校验维度分数（支持嵌套花括号，失败返回空 dict）"""
    text = text.strip()
    candidates: list[str] = []
    if text.startswith("{"):
        candidates.append(text)
    # 提取所有最外层花括号包裹的片段
    for m in re.finditer(r"\{[^{}]*\}", text):
        candidates.append(m.group())
    if not candidates:
        i, j = text.find("{"), text.rfind("}")
        if 0 <= i < j:
            candidates.append(text[i:j + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        scores = {k: _clamp_score(obj.get(k)) for k in _VLM_DIM_KEYS}
        if any(v is None for v in scores.values()):
            continue
        out = {k: round(v, 1) for k, v in scores.items()}
        reason = obj.get("reason")
        if isinstance(reason, str) and reason.strip():
            out["reason"] = reason.strip()[:200]
        # hard_flaws：仅接受枚举白名单内字符串（过滤"无硬伤"占位符与 LLM 幻觉项），
        # 剔除空串，去重保序，上限 5 个
        flaws = obj.get("hard_flaws")
        if isinstance(flaws, list):
            seen, valid_flaws = set(), []
            for f in flaws:
                f = f.strip() if isinstance(f, str) else ""
                if f and f not in seen and f in _HARD_FLAW_ENUMS:
                    seen.add(f)
                    valid_flaws.append(f)
            if valid_flaws:
                out["hard_flaws"] = valid_flaws[:5]
        category = obj.get("category")
        if isinstance(category, str):
            cat = category.strip()
            if cat in _CATEGORIES:
                out["category"] = cat
            elif cat in _CATEGORY_ALIASES:
                out["category"] = _CATEGORY_ALIASES[cat]
        # person_is_subject：VLM 主体裁断（钳制门控依据）；容忍布尔字符串
        pis = obj.get("person_is_subject")
        if isinstance(pis, bool):
            out["person_is_subject"] = pis
        elif isinstance(pis, str) and pis.strip().lower() in ("true", "false"):
            out["person_is_subject"] = pis.strip().lower() == "true"
        return out
    logger.warning(f"无法从 VLM 回复中解析维度分数: {text[:120]}")
    return {}


def _normalize_batch_scores(results: list[Layer3Result]) -> None:
    """批次归一化 TOPIQ 综合分（p5-p95 拉伸到 0-10），拉开区分度

    - 少于 2 张或分数分布过于集中时保持不变（避免噪声放大）
    """
    valid = [r for r in results if not r.error and r.overall_score > 0]
    if len(valid) < 2:
        return
    vals = sorted(r.overall_score for r in valid)
    n = len(vals)
    lo = vals[max(0, int(n * 0.05) - 1)]
    hi = vals[min(n - 1, int(n * 0.95) - 1)]
    span = hi - lo
    if span < 0.5:
        return
    for r in valid:
        norm = max(0.0, min(1.0, (r.overall_score - lo) / span))
        r.overall_score = round(norm * 10, 4)


async def _analyze_dimensions(
    image_path: str, api_key: str, base_url: str, model: str,
    proxy: str = "", max_retries: int = _VLM_RETRIES,
    face_summary: str | None = None,
) -> dict:
    """用 Qwen3-VL 按专业评分标准分析照片的构图/色彩/光影/综合审美分

    - 图片先缩放到长边 VLM_IMAGE_MAX_SIDE（减小上传体积、提速）
    - 指数退避重试 + 分数钳制 [1,10]
    """
    import httpx

    from core.image_cache import get_cached_path

    img = Image.open(get_cached_path(image_path)).convert("RGB")
    w, h = img.size
    if max(w, h) > VLM_IMAGE_MAX_SIDE:
        ratio = VLM_IMAGE_MAX_SIDE / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=VLM_JPEG_QUALITY)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = _build_dimension_prompt(face_summary)

    url = f"{base_url}/chat/completions"
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=_VLM_TIMEOUT, proxy=proxy or None) as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                                {"type": "text", "text": prompt},
                            ],
                        }],
                        "max_tokens": 400,
                        "temperature": 0.1,
                        # 评分任务不需要思考链：显式关闭，防思考 token 挤占
                        # max_tokens 导致 JSON 截断（→ 解析失败白重试），同时提速省费
                        "enable_thinking": False,
                        # JSON 强约束输出，减少解析失败（_parse_dimension_json 正则保留作兜底）
                        "response_format": {"type": "json_object"},
                    },
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                parsed = _parse_dimension_json(content)
                if parsed:
                    return parsed
                last_err = ValueError(f"JSON 提取失败: {content[:100]}")
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(
                    f"维度分析失败 (尝试 {attempt + 1}/{max_retries}): {e}，{wait}s 后重试"
                )
                await asyncio.sleep(wait)
    raise RuntimeError(f"维度分析失败: {last_err}")
