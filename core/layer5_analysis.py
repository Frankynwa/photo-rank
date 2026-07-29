"""Layer 5: 深度分析 — Qwen-VL Max API（分类+构图+改进+文案+故事线+情绪）"""
import os
import base64
import json
import httpx
from pathlib import Path
from config import QWEN_BASE_URL, QWEN_MODEL, PROXY


def get_api_key() -> str:
    """从 .env 文件读取 API Key"""
    env_path = Path(r"C:\Users\Administrator\AppData\Local\hermes\.env")
    with open(env_path, "r") as f:
        for line in f:
            if line.startswith("QWEN_API_KEY"):
                return line.strip().split("=", 1)[1]
    raise ValueError("QWEN_API_KEY not found in .env")


def encode_image(image_path: str | Path, max_size: int = 1024) -> str:
    """将图片编码为 base64（压缩到 max_size 以内）"""
    from PIL import Image
    import io
    
    img = Image.open(image_path).convert("RGB")
    
    # 压缩到 max_size 以内
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    
    # 编码为 JPEG
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def call_qwen_vl(prompt: str, image_paths: list[str | Path] = None) -> str:
    """调用 Qwen-VL Max API"""
    api_key = get_api_key()
    
    content = []
    
    if image_paths:
        for path in image_paths:
            b64 = encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
    
    content.append({"type": "text", "text": prompt})
    
    async with httpx.AsyncClient(timeout=120, proxy=PROXY) as client:
        response = await client.post(
            f"{QWEN_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": QWEN_MODEL,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 2000,
                "temperature": 0.3,
            },
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]


async def classify_photo(image_path: str | Path) -> str:
    """分类照片（8类）"""
    prompt = """这张照片属于哪个类别？只回答编号和名称。

1. 风景（自然风光、草原、山、水、天空）
2. 人像（他拍，单人）
3. 人文（建筑、街景、文化、民俗）
4. 美食
5. 夜景
6. 自拍（手机前置摄像头）
7. 合照（多人合影）
8. 其他

回答格式：编号. 名称"""
    
    return (await call_qwen_vl(prompt, [image_path])).strip()


async def analyze_composition(image_path: str | Path) -> str:
    """构图解读"""
    prompt = """分析这张照片的构图，用简洁的语言：

1. 主要构图方法（三分法/黄金比例/引导线/对称/中心构图等）
2. 主体位置和留白方向
3. 构图优点
4. 构图改进建议

每点一句话。"""
    
    return (await call_qwen_vl(prompt, [image_path])).strip()


async def suggest_improvement(image_path: str | Path) -> str:
    """改进建议"""
    prompt = """这张照片如果要更好，给出具体可操作的建议：

1. 裁剪建议（裁掉哪边，比例多少）
2. 调色建议（亮度/对比度/饱和度/色温）
3. 角度建议（换什么角度拍会更好）

每点给出具体参数或方向。"""
    
    return (await call_qwen_vl(prompt, [image_path])).strip()


async def analyze_emotion(image_path: str | Path) -> dict:
    """情绪图谱"""
    prompt = """分析这张照片传递的情绪，用 JSON 格式：

{
  "primary_emotion": "主要情绪（快乐/宁静/震撼/孤独/温暖/自由/期待/感动）",
  "emotion_intensity": 8,
  "mood_keywords": ["关键词1", "关键词2", "关键词3"],
  "suitable_copywriting_style": "适合的文案风格（文艺/哲思/温暖/幽默/简洁）"
}

只输出 JSON。"""
    
    result = await call_qwen_vl(prompt, [image_path])
    try:
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0]
        elif "```" in result:
            result = result.split("```")[1].split("```")[0]
        return json.loads(result.strip())
    except:
        return {
            "primary_emotion": "未知",
            "emotion_intensity": 5,
            "mood_keywords": [],
            "suitable_copywriting_style": "文艺",
        }


async def generate_copywriting(image_path: str | Path, platform: str = "xiaohongshu") -> str:
    """生成文案"""
    platform_names = {
        "xiaohongshu": "小红书",
        "wechat": "朋友圈",
        "douyin": "抖音",
    }
    platform_name = platform_names.get(platform, "社交平台")
    
    prompt = f"""为这张照片写一段{platform_name}文案。

风格要求：
- 有深度有哲思
- 低调谦卑不装
- 不要过度文艺，不要假大空
- 真实、自然、有温度
- 15-30字

禁用："绝了/太美了/氛围感拉满"、emoji、#话题标签。

只输出文案本身。"""
    
    return (await call_qwen_vl(prompt, [image_path])).strip()


async def arrange_storyline(image_paths: list[str | Path]) -> str:
    """视觉故事线排列"""
    prompt = f"""这 {len(image_paths)} 张照片要组成一个视觉故事。

给出排列建议：
- 开头用哪张（引人入胜）
- 中间怎么递进（情绪起伏）
- 结尾用哪张（余韵）

用文件名引用，简洁说明。"""
    
    return (await call_qwen_vl(prompt, image_paths)).strip()


async def deep_analyze(image_path: str | Path, platform: str = "xiaohongshu") -> dict:
    """单张照片深度分析"""
    import asyncio
    
    results = await asyncio.gather(
        classify_photo(image_path),
        analyze_composition(image_path),
        suggest_improvement(image_path),
        analyze_emotion(image_path),
        generate_copywriting(image_path, platform),
    )
    
    return {
        "path": str(image_path),
        "filename": Path(image_path).name,
        "classification": results[0],
        "composition": results[1],
        "improvement": results[2],
        "emotion": results[3],
        "copywriting": results[4],
        "platform": platform,
    }


async def batch_deep_analyze(
    image_paths: list[str | Path],
    platform: str = "xiaohongshu",
) -> list[dict]:
    """批量深度分析"""
    results = []
    for i, path in enumerate(image_paths):
        print(f"  深度分析进度: {i + 1}/{len(image_paths)} - {Path(path).name}")
        result = await deep_analyze(path, platform)
        results.append(result)
    return results
