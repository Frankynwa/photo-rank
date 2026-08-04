"""Layer 3: 审美评分 — TOPIQ-IAA（批次归一化）+ VLM 多维度专业评分（rubric + 重试 + 校验）"""
import asyncio
import base64
import concurrent.futures
import gc
import io
import json
import logging
import os
import re
from pathlib import Path

from PIL import Image

from config import VLM_IMAGE_MAX_SIDE
from core.models import Layer3Result

logger = logging.getLogger(__name__)

# VLM 维度评分参数
_VLM_TIMEOUT = 60  # 单次 API 超时（秒）
_VLM_RETRIES = 3  # 失败重试次数（指数退避）
_VLM_DIM_KEYS = ("composition_score", "color_score", "lighting_score", "overall_aesthetic_score")

# VLM 分类标签枚举（与 prompt 中的分类清单保持一致）
_CATEGORIES = ("风景", "建筑", "人像", "人文纪实", "美食", "动物", "夜景", "其他")

_DIMENSION_PROMPT = """你是从业 15 年的资深旅行摄影师兼选片师，为摄影比赛和社交平台评审过大量照片。
请以大众审美标准评审这张旅行照片。

【评审流程】
第一步，先在心里快速检查以下硬伤（不用输出）：
1. 画面模糊/失焦  2. 严重过曝或欠曝（死白/死黑）  3. 噪点明显
4. 水平线明显歪斜  5. 画面杂乱/多余元素入画  6. 主体被裁切
7. 逆光导致主体发黑  8. 地平线穿头/穿肩
存在任何硬伤：对应维度的分数上限为 4 分。

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
- 8 分以上 = 专业级/封面水平，严格控制，10 张里最多 1 张
- 4 分以下 = 有明显问题，一般不值得保留

【输出要求】
- 先在心里完成评审流程，再简要说明打分理由（30-50 字），最后给分数。
- 照片类型（category）从以下枚举中选一个：风景 / 建筑 / 人像 / 人文纪实 / 美食 / 动物 / 夜景 / 其他。
  按画面中最突出的主体与场景判断，只选一个，不要输出枚举之外的词。
- 只返回 JSON：{"reason": "理由", "composition_score": x, "color_score": x, "lighting_score": x, "overall_aesthetic_score": x, "category": "分类"}"""


def score_photos(
    image_paths: list[str | Path],
    device: str = "auto",
) -> list[Layer3Result]:
    """评分照片（逐张推理。TOPIQ-IAA 不支持 batch 路径列表）

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

    # 加载模型（带容错 + CPU fallback）
    try:
        logger.info("加载 TOPIQ-IAA...")
        model = pyiqa.create_metric("topiq_iaa", device=device)
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        if device != "cpu":
            try:
                logger.info("尝试 CPU fallback...")
                model = pyiqa.create_metric("topiq_iaa", device="cpu")
                device = "cpu"
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
                img = Image.open(path).convert("RGB")
                score = model(img).item() * 10  # TOPIQ-IAA 输出 [0,1]，映射到 [0,10] 与 NIMA 兼容
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
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            logger.debug("GPU 显存已释放")

    # TOPIQ 批次归一化：p5-p95 拉伸，拉开区分度
    _normalize_batch_scores(results)

    # ---------- 多维度 API 分析 ----------
    api_key = os.environ.get("QWEN_API_KEY", "")

    if api_key:
        logger.info("开始多维度审美分析（构图/色彩/光影/综合审美）...")
    else:
        logger.warning("未找到 QWEN_API_KEY，跳过多维度分析，使用默认值 5.0")

    for idx, result in enumerate(results):
        if result.error:
            # 评分失败的图片不额外调用 API
            result.composition_score = 5.0
            result.color_score = 5.0
            result.lighting_score = 5.0
            continue

        dimensions = {"composition_score": 5.0, "color_score": 5.0, "lighting_score": 5.0}
        if api_key:
            try:
                dims = _run_dimension_analysis(result.path)
                dimensions.update({
                    "composition_score": dims.get("composition_score", 5.0),
                    "color_score": dims.get("color_score", 5.0),
                    "lighting_score": dims.get("lighting_score", 5.0),
                })
                if dims.get("overall_aesthetic_score"):
                    dimensions["overall_aesthetic_score"] = dims["overall_aesthetic_score"]
                if dims.get("reason"):
                    dimensions["reason"] = dims["reason"]
                if dims.get("category"):
                    result.category = dims["category"]
            except Exception as e:
                logger.warning(f"维度分析失败 {result.filename}: {e}")

        result.composition_score = round(dimensions["composition_score"], 4)
        result.color_score = round(dimensions["color_score"], 4)
        result.lighting_score = round(dimensions["lighting_score"], 4)
        if dimensions.get("overall_aesthetic_score", 0) > 0:
            result.vlm_overall_score = round(dimensions["overall_aesthetic_score"], 4)
            result.score_reason = dimensions.get("reason", "")

        if (idx + 1) % 5 == 0:
            logger.info(f"多维度分析进度: {idx + 1}/{len(results)}")

    if api_key:
        logger.info("多维度审美分析完成")

    return results


def _run_dimension_analysis(image_path: str) -> dict:
    """同步调用 VLM 维度分析（兼容已有事件循环场景，用新线程兜底）"""
    from config import PROXY, QWEN_BASE_URL, QWEN_MODEL

    api_key = os.environ.get("QWEN_API_KEY", "")
    try:
        return asyncio.run(
            _analyze_dimensions(image_path, api_key, QWEN_BASE_URL, QWEN_MODEL, PROXY)
        )
    except RuntimeError:
        # 已有事件循环在运行，用新线程
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run,
                _analyze_dimensions(image_path, api_key, QWEN_BASE_URL, QWEN_MODEL, PROXY),
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
        category = obj.get("category")
        if isinstance(category, str) and category.strip() in _CATEGORIES:
            out["category"] = category.strip()
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
) -> dict:
    """用 Qwen3-VL 按专业评分标准分析照片的构图/色彩/光影/综合审美分

    - 图片先缩放到长边 VLM_IMAGE_MAX_SIDE（减小上传体积、提速）
    - 指数退避重试 + 分数钳制 [1,10]
    """
    import httpx

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > VLM_IMAGE_MAX_SIDE:
        ratio = VLM_IMAGE_MAX_SIDE / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = _DIMENSION_PROMPT

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
