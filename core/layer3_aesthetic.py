"""Layer 3: 审美评分 — TOPIQ-IAA（GPU/CPU 自动切换，自动释放显存）+ 多维度 API 评分"""
import asyncio
import base64
import gc
import io
import json
import logging
import os
import re
from pathlib import Path

from PIL import Image

from core.models import Layer3Result

logger = logging.getLogger(__name__)


def score_photos(
    image_paths: list[str | Path],
    device: str = "auto",
) -> list[Layer3Result]:
    """评分照片（逐张推理。TOPIQ-IAA 不支持 batch 路径列表）"""
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

    # ---------- 多维度 API 分析 ----------
    from config import PROXY, QWEN_BASE_URL, QWEN_MODEL
    api_key = os.environ.get("QWEN_API_KEY", "")

    if api_key:
        logger.info("开始多维度审美分析（构图/色彩/光影）...")
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
                dims = asyncio.run(
                    _analyze_dimensions(result.path, api_key, QWEN_BASE_URL, QWEN_MODEL, PROXY)
                )
                dimensions.update({
                    "composition_score": dims.get("composition_score", 5.0),
                    "color_score": dims.get("color_score", 5.0),
                    "lighting_score": dims.get("lighting_score", 5.0),
                })
            except RuntimeError:
                # 已有事件循环在运行，用新线程
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    dims = pool.submit(
                        asyncio.run,
                        _analyze_dimensions(result.path, api_key, QWEN_BASE_URL, QWEN_MODEL, PROXY),
                    ).result()
                dimensions.update({
                    "composition_score": dims.get("composition_score", 5.0),
                    "color_score": dims.get("color_score", 5.0),
                    "lighting_score": dims.get("lighting_score", 5.0),
                })
            except Exception as e:
                logger.warning(f"维度分析失败 {result.filename}: {e}")

        result.composition_score = round(dimensions["composition_score"], 4)
        result.color_score = round(dimensions["color_score"], 4)
        result.lighting_score = round(dimensions["lighting_score"], 4)

        if (idx + 1) % 5 == 0:
            logger.info(f"多维度分析进度: {idx + 1}/{len(results)}")

    if api_key:
        logger.info("多维度审美分析完成")

    return results


async def _analyze_dimensions(
    image_path: str, api_key: str, base_url: str, model: str, proxy: str = ""
) -> dict:
    """用 Qwen3-VL 分析照片的构图/色彩/光影分数"""
    import httpx

    # 编码图片
    img = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = """分析这张照片的三个维度，以 JSON 返回：
{"composition_score": 8, "color_score": 7, "lighting_score": 9}

评分标准（1-10 分）：
- composition_score: 构图（三分法/对称/引导线/框架/留白）
- color_score: 色彩（和谐度/饱和度/色调统一）
- lighting_score: 光影（曝光合理/对比度/氛围感）

只返回 JSON，不要其他内容。"""

    url = f"{base_url}/chat/completions"
    async with httpx.AsyncClient(timeout=30, proxy=proxy or None) as client:
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
                "max_tokens": 200,
                "temperature": 0.3,
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # 提取 JSON
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            return json.loads(json_match.group())
        return {}
