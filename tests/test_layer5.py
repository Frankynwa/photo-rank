"""Layer 5 深度分析单元测试 — mock httpx，不依赖网络"""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# 始终可测的纯函数 / 轻量函数（无需 mock）
# ---------------------------------------------------------------------------
from core.layer5_analysis import (
    _extract_json,
    encode_image,
    get_api_key,
)
from core.models import EmotionAnalysis, Layer5Result

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


def _mock_api_response(content: str):
    """构建模拟的 httpx 响应"""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    mock_response.raise_for_status = MagicMock()
    return mock_response


def _make_async_client_mock(response_content: str):
    """创建返回指定内容的 AsyncClient mock"""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        return_value=_mock_api_response(response_content)
    )
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


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

    def test_json_with_special_characters(self):
        """含中文和特殊字符的 JSON 应正确解析"""
        text = '{"style": "文艺/哲思", "text": "你好，世界！"}'
        result = _extract_json(text)
        assert result["style"] == "文艺/哲思"

    def test_json_array_not_extracted(self):
        """JSON 数组（不以 { 开头）应走 fallback 路径"""
        text = '[1, 2, 3]'
        result = _extract_json(text)
        # 不以 { 开头，没有 ``` 包裹，没有 { → 返回 {}
        assert result == {}


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
# 异步 API 调用测试（mock httpx）
# ---------------------------------------------------------------------------

class TestCallApi:
    @pytest.mark.asyncio
    async def test_successful_api_call(self, tmp_path):
        """成功的 API 调用应返回响应内容"""
        from core.layer5_analysis import _call_api
        p = _make_random_image(tmp_path / "api.png")

        mock_client = _make_async_client_mock("分类结果：1. 风景")
        with patch("core.layer5_analysis.httpx.AsyncClient", return_value=mock_client), \
             patch("core.layer5_analysis.get_api_key", return_value="fake-key"):
            result = await _call_api("test prompt", p)

        assert result == "分类结果：1. 风景"

    @pytest.mark.asyncio
    async def test_api_call_retries_on_failure(self, tmp_path):
        """API 调用失败应按指数退避重试"""
        from core.layer5_analysis import _call_api
        p = _make_random_image(tmp_path / "retry.png")

        mock_client = AsyncMock()
        # 前两次失败，第三次成功
        mock_client.post = AsyncMock(
            side_effect=[
                Exception("timeout"),
                Exception("timeout"),
                _mock_api_response("成功"),
            ]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("core.layer5_analysis.httpx.AsyncClient", return_value=mock_client), \
             patch("core.layer5_analysis.get_api_key", return_value="fake-key"), \
             patch("core.layer5_analysis.asyncio.sleep", new_callable=AsyncMock):
            result = await _call_api("test", p, max_retries=3)

        assert result == "成功"
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_api_call_exhausts_retries(self, tmp_path):
        """所有重试都失败应抛出异常"""
        from core.layer5_analysis import _call_api
        p = _make_random_image(tmp_path / "fail.png")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("always fails"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("core.layer5_analysis.httpx.AsyncClient", return_value=mock_client), \
             patch("core.layer5_analysis.get_api_key", return_value="fake-key"), \
             patch("core.layer5_analysis.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="always fails"):
                await _call_api("test", p, max_retries=2)


# ---------------------------------------------------------------------------
# 情绪分析结果解析测试
# ---------------------------------------------------------------------------

class TestAnalyzeEmotion:
    @pytest.mark.asyncio
    async def test_valid_emotion_json(self, tmp_path):
        """有效的情绪分析 JSON 应被正确解析为 EmotionAnalysis"""
        from core.layer5_analysis import analyze_emotion
        p = _make_random_image(tmp_path / "emotion.png")

        emotion_json = json.dumps({
            "primary_emotion": "快乐",
            "emotion_intensity": 8,
            "mood_keywords": ["阳光", "微笑"],
            "suitable_copywriting_style": "温暖",
        })
        mock_client = _make_async_client_mock(emotion_json)

        with patch("core.layer5_analysis.httpx.AsyncClient", return_value=mock_client), \
             patch("core.layer5_analysis.get_api_key", return_value="fake-key"):
            result = await analyze_emotion(p)

        assert isinstance(result, EmotionAnalysis)
        assert result.primary_emotion == "快乐"
        assert result.emotion_intensity == 8.0
        assert "阳光" in result.mood_keywords
        assert result.suitable_copywriting_style == "温暖"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_defaults(self, tmp_path):
        """无法解析的 JSON 应返回默认 EmotionAnalysis"""
        from core.layer5_analysis import analyze_emotion
        p = _make_random_image(tmp_path / "bad_json.png")

        mock_client = _make_async_client_mock("这不是 JSON")

        with patch("core.layer5_analysis.httpx.AsyncClient", return_value=mock_client), \
             patch("core.layer5_analysis.get_api_key", return_value="fake-key"):
            result = await analyze_emotion(p)

        default = EmotionAnalysis()
        assert result.primary_emotion == default.primary_emotion
        assert result.emotion_intensity == default.emotion_intensity

    @pytest.mark.asyncio
    async def test_partial_json_fills_defaults(self, tmp_path):
        """部分字段的 JSON 应填充默认值"""
        from core.layer5_analysis import analyze_emotion
        p = _make_random_image(tmp_path / "partial.png")

        emotion_json = json.dumps({"primary_emotion": "宁静"})
        mock_client = _make_async_client_mock(emotion_json)

        with patch("core.layer5_analysis.httpx.AsyncClient", return_value=mock_client), \
             patch("core.layer5_analysis.get_api_key", return_value="fake-key"):
            result = await analyze_emotion(p)

        assert result.primary_emotion == "宁静"
        # 缺失字段使用默认值
        default = EmotionAnalysis()
        assert result.emotion_intensity == default.emotion_intensity
        assert result.mood_keywords == default.mood_keywords


# ---------------------------------------------------------------------------
# 深度分析集成测试（mock API）
# ---------------------------------------------------------------------------

class TestDeepAnalyze:
    @pytest.mark.asyncio
    async def test_deep_analyze_all_success(self, tmp_path):
        """所有 API 调用成功时应返回完整 Layer5Result"""
        from core.layer5_analysis import deep_analyze
        p = _make_random_image(tmp_path / "deep.png")

        with patch("core.layer5_analysis._call_api", new_callable=AsyncMock) as mock_api:
            mock_api.side_effect = [
                "1. 风景",                    # classify_photo
                "三分法构图",                  # analyze_composition
                "提高亮度",                    # suggest_improvement
                '{"primary_emotion": "宁静"}', # analyze_emotion
                "光影之间，是时间的低语",       # generate_copywriting
            ]
            result = await deep_analyze(p)

        assert isinstance(result, Layer5Result)
        assert result.classification == "1. 风景"
        assert result.composition == "三分法构图"
        assert result.improvement == "提高亮度"
        assert result.emotion.primary_emotion == "宁静"
        assert result.copywriting == "光影之间，是时间的低语"

    @pytest.mark.asyncio
    async def test_deep_analyze_partial_failure(self, tmp_path):
        """部分 API 调用失败时应使用默认值"""
        from core.layer5_analysis import deep_analyze
        p = _make_random_image(tmp_path / "partial_fail.png")

        with patch("core.layer5_analysis._call_api", new_callable=AsyncMock) as mock_api:
            mock_api.side_effect = [
                "1. 风景",
                Exception("API error"),       # 构图失败
                "提高亮度",
                '{"primary_emotion": "温暖"}',
                "好文案",
            ]
            result = await deep_analyze(p)

        assert result.classification == "1. 风景"
        assert result.composition == "API 调用失败"
        assert result.improvement == "提高亮度"


# ---------------------------------------------------------------------------
# 并发控制测试
# ---------------------------------------------------------------------------

class TestBatchDeepAnalyze:
    @pytest.mark.asyncio
    async def test_batch_processes_all_images(self, tmp_path):
        """batch_deep_analyze 应处理所有图片"""
        from core.layer5_analysis import batch_deep_analyze
        paths = [_make_random_image(tmp_path / f"batch{i}.png") for i in range(3)]

        async def fake_deep_analyze(path, platform="xiaohongshu"):
            return Layer5Result(
                path=str(path), filename=Path(path).name,
                classification="风景", composition="构图好",
                improvement="调亮", copywriting="文案",
                platform=platform,
            )

        with patch("core.layer5_analysis.deep_analyze", side_effect=fake_deep_analyze):
            results = await batch_deep_analyze(paths)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, Layer5Result)
            assert r.classification == "风景"

    @pytest.mark.asyncio
    async def test_batch_handles_exception(self, tmp_path):
        """batch 中单个任务异常不应影响其他结果"""
        from core.layer5_analysis import batch_deep_analyze
        paths = [_make_random_image(tmp_path / f"exc{i}.png") for i in range(3)]

        call_count = 0

        async def flaky_deep_analyze(path, platform="xiaohongshu"):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("boom")
            return Layer5Result(
                path=str(path), filename=Path(path).name,
                classification="成功", platform=platform,
            )

        with patch("core.layer5_analysis.deep_analyze", side_effect=flaky_deep_analyze):
            results = await batch_deep_analyze(paths)

        assert len(results) == 3
        # 第一个和第三个成功
        assert results[0].classification == "成功"
        assert results[2].classification == "成功"
        # 第二个失败 → 填充错误信息
        assert results[1].classification == "分析失败"
        assert "boom" in results[1].composition

    @pytest.mark.asyncio
    async def test_batch_empty_list(self):
        """空列表应返回空结果"""
        from core.layer5_analysis import batch_deep_analyze
        results = await batch_deep_analyze([])
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_platform_propagated(self, tmp_path):
        """platform 参数应传递到每个 deep_analyze 调用"""
        from core.layer5_analysis import batch_deep_analyze
        paths = [_make_random_image(tmp_path / "plat.png")]

        received_platforms = []

        async def capture_platform(path, platform="xiaohongshu"):
            received_platforms.append(platform)
            return Layer5Result(
                path=str(path), filename=Path(path).name,
                platform=platform,
            )

        with patch("core.layer5_analysis.deep_analyze", side_effect=capture_platform):
            await batch_deep_analyze(paths, platform="wechat")

        assert received_platforms == ["wechat"]
