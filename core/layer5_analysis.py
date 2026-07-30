"""Layer 5: 深度分析 — Qwen-VL API（分类+构图+改进+文案+情绪）

支持并行 API 调用，指数退避重试，JSON 提取增强。
"""
import os
import base64
import json
import asyncio
import threading
import logging
from pathlib import Path
from PIL import Image
import io
import httpx
from dotenv import load_dotenv
from config import QWEN_BASE_URL, QWEN_MODEL, PROXY

logger = logging.getLogger(__name__)

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


async def classify_photo(image_path: str | Path) -> str:
    prompt = """这张照片属于哪个类别？只回答编号和名称。
1. 风景  2. 人像  3. 人文  4. 美食  5. 夜景  6. 自拍  7. 合照  8. 其他
格式：编号. 名称"""
    return (await _call_api(prompt, image_path)).strip()


async def analyze_composition(image_path: str | Path) -> str:
    prompt = """分析这张照片的构图：1.主要构图方法 2.主体位置和留白 3.构图优点 4.改进建议。每点一句话。"""
    return (await _call_api(prompt, image_path)).strip()


async def suggest_improvement(image_path: str | Path) -> str:
    prompt = """给这张照片改进建议：1.裁剪（裁哪边，比例）2.调色（亮度/对比度/饱和度/色温）3.角度。每点具体参数。"""
    return (await _call_api(prompt, image_path)).strip()


async def analyze_emotion(image_path: str | Path) -> dict:
    prompt = """分析照片情绪（JSON）：{"primary_emotion":"快乐/宁静/震撼/孤独/温暖/自由/期待/感动","emotion_intensity":8,"mood_keywords":["k1","k2","k3"],"suitable_copywriting_style":"文艺/哲思/温暖/幽默/简洁"}"""
    result = await _call_api(prompt, image_path)
    parsed = _extract_json(result)
    if not parsed:
        logger.warning(f"情绪分析 JSON 解析失败: {Path(image_path).name}")
    return parsed or {
        "primary_emotion": "未知", "emotion_intensity": 5,
        "mood_keywords": [], "suitable_copywriting_style": "文艺",
    }


async def generate_copywriting(image_path: str | Path, platform: str = "xiaohongshu") -> str:
    names = {"xiaohongshu": "小红书", "wechat": "朋友圈", "douyin": "抖音"}
    name = names.get(platform, "社交平台")
    prompt = f"""为这张照片写{name}文案。有深度有哲思，低调谦卑不装。15-30字。禁用"绝了/太美了/氛围感拉满"、emoji、#标签。只输出文案。"""
    return (await _call_api(prompt, image_path)).strip()


async def deep_analyze(image_path: str | Path, platform: str = "xiaohongshu") -> dict:
    """单张深度分析（5 个并行 API 调用）"""
    results = await asyncio.gather(
        classify_photo(image_path),
        analyze_composition(image_path),
        suggest_improvement(image_path),
        analyze_emotion(image_path),
        generate_copywriting(image_path, platform),
        return_exceptions=True,
    )

    return {
        "path": str(image_path),
        "filename": Path(image_path).name,
        "classification": results[0] if not isinstance(results[0], Exception) else "API 调用失败",
        "composition": results[1] if not isinstance(results[1], Exception) else "API 调用失败",
        "improvement": results[2] if not isinstance(results[2], Exception) else "API 调用失败",
        "emotion": results[3] if not isinstance(results[3], Exception) else {},
        "copywriting": results[4] if not isinstance(results[4], Exception) else "API 调用失败",
        "platform": platform,
    }


async def batch_deep_analyze(
    image_paths: list[str | Path], platform: str = "xiaohongshu",
) -> list[dict]:
    """批量深度分析（并行处理所有照片）"""
    tasks = [deep_analyze(p, platform) for p in image_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [
        r if not isinstance(r, Exception) else {
            "path": str(image_paths[i]), "filename": Path(image_paths[i]).name,
            "classification": "分析失败", "composition": str(r),
            "improvement": str(r), "emotion": {}, "copywriting": "分析失败",
            "platform": platform,
        }
        for i, r in enumerate(results)
    ]
