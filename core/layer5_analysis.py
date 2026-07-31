"""Layer 5: 深度分析 — Qwen-VL API（分类+构图+改进+文案+情绪）

支持并行 API 调用，指数退避重试，JSON 提取增强。
"""
import asyncio
import base64
import io
import json
import logging
import os
import threading
from pathlib import Path

import httpx
from dotenv import load_dotenv
from PIL import Image

from config import PROXY, QWEN_BASE_URL, QWEN_MODEL
from core.models import EmotionAnalysis, Layer5Result

logger = logging.getLogger(__name__)

# 平台专属分析 prompt
PLATFORM_PROMPTS = {
    "xiaohongshu": {
        "classification": "分析这张照片的类型（风景/人像/美食/街拍/夜景/其他）。小红书用户偏好：氛围感、故事感、封面感强的照片。",
        "composition": "从构图角度分析这张照片（三分法/对称/引导线/框架/留白等）。小红书封面需要强构图感。给出构图评分（1-10）和具体建议。以 JSON 返回：{\"score\": 数字, \"analysis\": \"分析\", \"suggestion\": \"建议\"}",
        "improvement": "从小红书爆款角度，给出这张照片的改进建议（调色/裁剪/滤镜/文字排版）。",
        "emotion": "分析照片的情绪氛围和适合的小红书文案风格。以 JSON 返回：{\"primary_emotion\": \"主要情绪\", \"mood_keywords\": [\"关键词1\",\"关键词2\"], \"suitable_copywriting_style\": \"文案风格\"}",
        "copywriting": "为这张照片写一段小红书风格的文案（100字内）。要求：有氛围感、带 emoji、有故事感、适合做封面配文。",
    },
    "wechat": {
        "classification": "分析这张照片的类型。朋友圈照片需要：生活感、真实感、叙事性。",
        "composition": "从构图角度分析。朋友圈九宫格需要照片之间有节奏感。给出构图评分（1-10）和分析。以 JSON 返回：{\"score\": 数字, \"analysis\": \"分析\", \"suggestion\": \"建议\"}",
        "improvement": "从朋友圈分享角度，给出改进建议。",
        "emotion": "分析照片的情绪，适合朋友圈什么场景分享（日常/旅行/美食/感悟）。以 JSON 返回：{\"primary_emotion\": \"主要情绪\", \"mood_keywords\": [\"关键词1\",\"关键词2\"], \"suitable_copywriting_style\": \"文案风格\"}",
        "copywriting": "为这张照片写一段朋友圈文案（50字内）。要求：自然、有生活感、不做作、适合配九宫格。",
    },
    "douyin": {
        "classification": "分析这张照片的类型。抖音封面需要：视觉冲击力、竖版适配、好奇心驱动。",
        "composition": "从构图角度分析。抖音封面需要强视觉冲击。给出构图评分（1-10）和分析。以 JSON 返回：{\"score\": 数字, \"analysis\": \"分析\", \"suggestion\": \"建议\"}",
        "improvement": "从抖音爆款角度，给出改进建议（冲击力/色彩/竖版裁剪）。",
        "emotion": "分析照片的情绪张力，是否适合做抖音封面。以 JSON 返回：{\"primary_emotion\": \"主要情绪\", \"mood_keywords\": [\"关键词1\",\"关键词2\"], \"suitable_copywriting_style\": \"文案风格\"}",
        "copywriting": "为这张照片写一段抖音风格的文案/标题（30字内）。要求：有冲击力、引发好奇、适合做视频封面文字。",
    },
}

# API Key 缓存
_api_key_cache: str | None = None
_api_key_lock = threading.Lock()


def get_api_key() -> str:
    """获取 Qwen API Key（带缓存）"""
    global _api_key_cache
    with _api_key_lock:
        if _api_key_cache:
            return _api_key_cache

        api_key = os.environ.get("QWEN_API_KEY")
        if api_key:
            _api_key_cache = api_key
            return api_key

        load_dotenv()
        api_key = os.environ.get("QWEN_API_KEY")
        if api_key:
            _api_key_cache = api_key
            return api_key

        raise ValueError(
            "找不到 QWEN_API_KEY！请设置环境变量或在项目根目录创建 .env 文件"
        )


def encode_image(image_path: str | Path, max_size: int = 1024) -> str:
    """将图片压缩为 JPEG base64"""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _extract_json(text: str) -> dict:
    """增强 JSON 提取：处理多种格式"""
    text = text.strip()

    # 1. 纯 JSON
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # 2. ```json ... ``` 代码块
    import re
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. 查找第一个 { 到最后一个 }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    logger.warning(f"无法从回复中提取 JSON: {text[:100]}...")
    return {}


async def _call_api(prompt: str, image_path: str | Path, max_retries: int = 3) -> str:
    """调用 Qwen-VL API（带指数退避重试）"""
    api_key = get_api_key()
    b64 = encode_image(image_path)
    url = f"{QWEN_BASE_URL}/chat/completions"

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=120, proxy=PROXY or None) as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": QWEN_MODEL,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                                {"type": "text", "text": prompt},
                            ],
                        }],
                        "max_tokens": 2000,
                        "temperature": 0.3,
                    },
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 指数退避：1s, 2s, 4s
                logger.warning(f"API 调用失败 (尝试 {attempt+1}/{max_retries}): {e}，{wait}s 后重试...")
                await asyncio.sleep(wait)
            else:
                raise


async def classify_photo(image_path: str | Path, platform: str = "xiaohongshu") -> str:
    prompts = PLATFORM_PROMPTS.get(platform, PLATFORM_PROMPTS["xiaohongshu"])
    prompt = prompts["classification"]
    return (await _call_api(prompt, image_path)).strip()


async def analyze_composition(image_path: str | Path, platform: str = "xiaohongshu") -> str:
    prompts = PLATFORM_PROMPTS.get(platform, PLATFORM_PROMPTS["xiaohongshu"])
    prompt = prompts["composition"]
    return (await _call_api(prompt, image_path)).strip()


async def suggest_improvement(image_path: str | Path, platform: str = "xiaohongshu") -> str:
    prompts = PLATFORM_PROMPTS.get(platform, PLATFORM_PROMPTS["xiaohongshu"])
    prompt = prompts["improvement"]
    return (await _call_api(prompt, image_path)).strip()


async def analyze_emotion(image_path: str | Path, platform: str = "xiaohongshu") -> EmotionAnalysis:
    prompts = PLATFORM_PROMPTS.get(platform, PLATFORM_PROMPTS["xiaohongshu"])
    prompt = prompts["emotion"]
    result = await _call_api(prompt, image_path)
    parsed = _extract_json(result)
    if not parsed:
        logger.warning(f"情绪分析 JSON 解析失败: {Path(image_path).name}")
    default = EmotionAnalysis()
    return EmotionAnalysis(
        primary_emotion=parsed.get("primary_emotion", default.primary_emotion),
        emotion_intensity=float(parsed.get("emotion_intensity", default.emotion_intensity)),
        mood_keywords=parsed.get("mood_keywords", default.mood_keywords),
        suitable_copywriting_style=parsed.get("suitable_copywriting_style", default.suitable_copywriting_style),
    )


async def generate_copywriting(image_path: str | Path, platform: str = "xiaohongshu") -> str:
    prompts = PLATFORM_PROMPTS.get(platform, PLATFORM_PROMPTS["xiaohongshu"])
    prompt = prompts["copywriting"]
    return (await _call_api(prompt, image_path)).strip()


async def deep_analyze(image_path: str | Path, platform: str = "xiaohongshu") -> Layer5Result:
    """单张深度分析（5 个并行 API 调用）"""
    results = await asyncio.gather(
        classify_photo(image_path, platform),
        analyze_composition(image_path, platform),
        suggest_improvement(image_path, platform),
        analyze_emotion(image_path, platform),
        generate_copywriting(image_path, platform),
        return_exceptions=True,
    )

    default_emotion = EmotionAnalysis()
    return Layer5Result(
        path=str(image_path),
        filename=Path(image_path).name,
        classification=results[0] if not isinstance(results[0], Exception) else "API 调用失败",
        composition=results[1] if not isinstance(results[1], Exception) else "API 调用失败",
        improvement=results[2] if not isinstance(results[2], Exception) else "API 调用失败",
        emotion=results[3] if not isinstance(results[3], Exception) else default_emotion,
        copywriting=results[4] if not isinstance(results[4], Exception) else "API 调用失败",
        platform=platform,
    )


async def batch_deep_analyze(
    image_paths: list[str | Path], platform: str = "xiaohongshu",
) -> list[Layer5Result]:
    """批量深度分析（并行处理所有照片）"""
    tasks = [deep_analyze(p, platform) for p in image_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [
        r if not isinstance(r, Exception) else Layer5Result(
            path=str(image_paths[i]), filename=Path(image_paths[i]).name,
            classification="分析失败", composition=str(r),
            improvement=str(r), copywriting="分析失败",
            platform=platform,
        )
        for i, r in enumerate(results)
    ]
