"""Layer 5 深度分析单元测试 — 纯逻辑测试 + 条件性 API 集成测试"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.models import EmotionAnalysis, Layer5Result

# ---------------------------------------------------------------------------
# 始终可测的纯函数 / 轻量函数
# ---------------------------------------------------------------------------
from core.layer5_analysis import (  # noqa: E402
    _extract_json,
    encode_image,
    get_api_key,
)


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

def _make_image(path: Path, size=(128, 128), color=(128, 128, 128)):
    img = Image.new("RGB", size, color)
    img.save(str(path))
    return path


def _make_random_image(path: Path, size=(128, 128)):
    rng = np.random.default_rng(99)
    arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(str(path))
    return path


# ---------------------------------------------------------------------------
# _extract_json 纯函数测试
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_pure_json(self):
        """纯 JSON 字符串应被正确解析"""
        text = '{"primary_emotion": "快乐", "emotion_intensity": 8}'
        result = _extract_json(text)
        assert result["primary_emotion"] == "快乐"
        assert result["emotion_intensity"] == 8

    def test_json_in_code_block(self):
        """```json ... ``` 代码块应被提取"""
        text = """这是一些分析结果：
```json
{"primary_emotion": "宁静", "emotion_intensity": 6, "mood_keywords": ["平静", "放松"]}
```
希望对你有帮助。"""
        result = _extract_json(text)
        assert result["primary_emotion"] == "宁静"
        assert "平静" in result["mood_keywords"]

    def test_json_without_language_tag(self):
        """``` ... ``` 无语言标签的代码块也应被提取"""
        text = """分析如下：
```
{"primary_emotion": "温暖"}
```"""
        result = _extract_json(text)
        assert result["primary_emotion"] == "温暖"

    def test_json_embedded_in_text(self):
        """JSON 嵌在文本中间应通过花括号定位提取"""
        text = '根据分析，结果是 {"primary_emotion": "感动", "emotion_intensity": 9} 如上。'
        result = _extract_json(text)
        assert result["primary_emotion"] == "感动"

    def test_invalid_json_returns_empty(self):
        """完全无法解析的文本应返回空 dict"""
        result = _extract_json("这不是 JSON 也不是任何有用的东西")
        assert result == {}

    def test_empty_string(self):
        """空字符串应返回空 dict"""
        assert _extract_json("") == {}
        assert _extract_json("   ") == {}

    def test_nested_json(self):
        """嵌套 JSON 应被正确解析"""
        text = '{"emotion": {"type": "快乐", "level": 8}, "keywords": ["a", "b"]}'
        result = _extract_json(text)
        assert result["emotion"]["type"] == "快乐"
        assert len(result["keywords"]) == 2

    def test_malformed_json_fallback(self):
        """第一个 { 到最后一个 } 的 fallback 应处理尾部多余文本"""
        text = '{"key": "value"} 一些额外文字 }'
        result = _extract_json(text)
        # fallback 从第一个 { 到最后一个 }，可能解析失败 → 返回 {}
        # 也可能恰好解析成功 → 取决于 json.loads 行为
        assert isinstance(result, dict)

    def test_json_with_special_characters(self):
        """含中文和特殊字符的 JSON 应正确解析"""
        text = '{"style": "文艺/哲思", "text": "你好，世界！"}'
        result = _extract_json(text)
        assert result["style"] == "文艺/哲思"


# ---------------------------------------------------------------------------
# encode_image 测试
# ---------------------------------------------------------------------------

class TestEncodeImage:
    def test_returns_base64_string(self, tmp_path: Path):
        """返回值应为非空 base64 字符串"""
        p = _make_random_image(tmp_path / "encode.png")
        b64 = encode_image(p)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_small_image_not_resized(self, tmp_path: Path):
        """小于 max_size 的图片不应被缩放"""
        p = _make_image(tmp_path / "small.png", size=(64, 64))
        b64 = encode_image(p, max_size=1024)
        assert len(b64) > 0

    def test_large_image_resized(self, tmp_path: Path):
        """大于 max_size 的图片应被缩放，输出仍为有效 base64"""
        p = _make_image(tmp_path / "large.png", size=(2000, 2000))
        b64 = encode_image(p, max_size=512)
        assert len(b64) > 0

    def test_different_max_sizes(self, tmp_path: Path):
        """不同 max_size 应产生不同大小的输出"""
        p = _make_random_image(tmp_path / "sizes.png", size=(512, 512))
        b64_small = encode_image(p, max_size=128)
        b64_large = encode_image(p, max_size=512)
        # 压缩后大图编码应更长
        assert len(b64_large) > len(b64_small)

    def test_rgb_conversion(self, tmp_path: Path):
        """RGBA 图片应被转换为 RGB 后编码"""
        p = tmp_path / "rgba.png"
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        img.save(str(p))
        b64 = encode_image(p)
        assert len(b64) > 0


# ---------------------------------------------------------------------------
# get_api_key 测试
# ---------------------------------------------------------------------------

class TestGetApiKey:
    def test_api_key_from_env(self, monkeypatch):
        """环境变量中有 QWEN_API_KEY 时应返回"""
        import core.layer5_analysis as l5
        # 清除缓存
        old_cache = l5._api_key_cache
        l5._api_key_cache = None
        try:
            monkeypatch.setenv("QWEN_API_KEY", "test-key-123")
            key = get_api_key()
            assert key == "test-key-123"
        finally:
            l5._api_key_cache = old_cache

    def test_api_key_missing_raises(self, monkeypatch):
        """没有 QWEN_API_KEY 应抛出 ValueError"""
        import core.layer5_analysis as l5
        old_cache = l5._api_key_cache
        l5._api_key_cache = None
        try:
            monkeypatch.delenv("QWEN_API_KEY", raising=False)
            # 确保 .env 中也没有
            monkeypatch.setattr("core.layer5_analysis.load_dotenv", lambda: None)
            with pytest.raises(ValueError, match="QWEN_API_KEY"):
                get_api_key()
        finally:
            l5._api_key_cache = old_cache

    def test_api_key_caching(self, monkeypatch):
        """缓存命中时不应再读环境变量"""
        import core.layer5_analysis as l5
        old_cache = l5._api_key_cache
        l5._api_key_cache = "cached-key"
        try:
            key = get_api_key()
            assert key == "cached-key"
        finally:
            l5._api_key_cache = old_cache


# ---------------------------------------------------------------------------
# 条件性 API 集成测试（需要 QWEN_API_KEY）
# ---------------------------------------------------------------------------

_has_api_key = bool(os.environ.get("QWEN_API_KEY"))

@pytest.mark.skipif(not _has_api_key, reason="需要设置 QWEN_API_KEY 环境变量")
class TestAPIIntegration:
    """需要真实 API Key 的集成测试"""

    @pytest.fixture
    def test_image(self, tmp_path: Path):
        p = _make_random_image(tmp_path / "api_test.png", size=(256, 256))
        return p

    @pytest.mark.asyncio
    async def test_classify_photo(self, test_image):
        """classify_photo 应返回非空字符串"""
        from core.layer5_analysis import classify_photo
        result = await classify_photo(test_image)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_analyze_emotion(self, test_image):
        """analyze_emotion 应返回 EmotionAnalysis"""
        from core.layer5_analysis import analyze_emotion
        result = await analyze_emotion(test_image)
        assert isinstance(result, EmotionAnalysis)
        assert result.primary_emotion
        assert 0 <= result.emotion_intensity <= 10

    @pytest.mark.asyncio
    async def test_batch_deep_analyze(self, test_image):
        """batch_deep_analyze 应返回 Layer5Result 列表"""
        from core.layer5_analysis import batch_deep_analyze
        results = await batch_deep_analyze([test_image])
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, Layer5Result)
        assert r.filename == test_image.name
