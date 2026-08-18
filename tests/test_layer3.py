"""Layer 3 审美评分单元测试 — mock pyiqa / torch，不依赖 GPU"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from core.models import Layer3Result

# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

def _make_image(path: Path, color=(128, 128, 128), size=(64, 64)):
    img = Image.new("RGB", size, color)
    img.save(str(path))
    return path


def _make_random_image(path: Path, size=(128, 128)):
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(str(path))
    return path


def _make_mock_model(score_value: float = 0.75):
    """创建一个模拟的 pyiqa 模型，返回固定分值"""
    mock_model = MagicMock()
    mock_tensor = MagicMock()
    mock_tensor.item.return_value = score_value
    mock_model.return_value = mock_tensor
    return mock_model


def _build_mock_modules(score_value=0.75, cuda_available=False, create_metric_side_effect=None):
    """构建 mock pyiqa 和 mock torch 模块"""
    mock_pyiqa = MagicMock()
    mock_model = _make_mock_model(score_value)
    if create_metric_side_effect:
        mock_pyiqa.create_metric.side_effect = create_metric_side_effect
    else:
        mock_pyiqa.create_metric.return_value = mock_model

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = cuda_available
    mock_torch.cuda.memory_allocated.return_value = 0
    return mock_pyiqa, mock_torch


# ---------------------------------------------------------------------------
# 设备选择逻辑测试
# ---------------------------------------------------------------------------

class TestDeviceSelection:
    def test_auto_selects_cuda_when_available(self, tmp_path):
        """torch.cuda.is_available()=True 时应使用 cuda"""
        mock_pyiqa, mock_torch = _build_mock_modules(cuda_available=True)
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            p = _make_random_image(tmp_path / "cuda.png")
            results = score_photos([p], device="auto")

        mock_pyiqa.create_metric.assert_called_once_with("topiq_iaa", device="cuda")
        assert results[0].device == "cuda"

    def test_auto_falls_back_to_cpu(self, tmp_path):
        """torch.cuda.is_available()=False 时应使用 cpu"""
        mock_pyiqa, mock_torch = _build_mock_modules(cuda_available=False)
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            p = _make_random_image(tmp_path / "cpu.png")
            results = score_photos([p], device="auto")

        mock_pyiqa.create_metric.assert_called_once_with("topiq_iaa", device="cpu")
        assert results[0].device == "cpu"

    def test_explicit_device_overrides_auto(self, tmp_path):
        """显式指定 device 应跳过自动检测"""
        mock_pyiqa, mock_torch = _build_mock_modules()
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            p = _make_random_image(tmp_path / "explicit.png")
            score_photos([p], device="cpu")

        mock_torch.cuda.is_available.assert_not_called()
        mock_pyiqa.create_metric.assert_called_once_with("topiq_iaa", device="cpu")


# ---------------------------------------------------------------------------
# score_photos 基本调用测试
# ---------------------------------------------------------------------------

class TestScorePhotos:
    def test_single_image(self, tmp_path):
        """单张图片应返回一个 Layer3Result"""
        mock_pyiqa, mock_torch = _build_mock_modules(score_value=0.8)
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            p = _make_random_image(tmp_path / "single.png")
            results = score_photos([p], device="cpu")

        assert len(results) == 1
        assert isinstance(results[0], Layer3Result)
        assert results[0].path == str(p)
        assert results[0].filename == "single.png"

    def test_multiple_images(self, tmp_path):
        """多张图片返回结果数量与输入一致"""
        mock_pyiqa, mock_torch = _build_mock_modules(score_value=0.6)
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            paths = [_make_random_image(tmp_path / f"img{i}.png") for i in range(3)]
            results = score_photos(paths, device="cpu")

        assert len(results) == 3
        for r in results:
            assert r.path
            assert r.filename

    def test_empty_list(self):
        """空列表应返回空结果"""
        mock_pyiqa, mock_torch = _build_mock_modules()
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            results = score_photos([], device="cpu")
        assert results == []


# ---------------------------------------------------------------------------
# 分值范围验证
# ---------------------------------------------------------------------------

class TestScoreRange:
    def test_score_passed_through_0_10(self, tmp_path):
        """模型输出即 0-10 的 MOS 分（dist_to_mos 已映射），不再 ×10"""
        mock_pyiqa, mock_torch = _build_mock_modules(score_value=0.75)
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            p = _make_random_image(tmp_path / "range.png")
            results = score_photos([p], device="cpu")

        r = results[0]
        expected = round(0.75, 4)
        assert r.aesthetic_score == expected
        assert r.technical_score == expected
        assert r.overall_score == expected
        assert r.final_score == expected

    def test_score_boundary_zero(self, tmp_path):
        """模型输出 0 应映射到 0"""
        mock_pyiqa, mock_torch = _build_mock_modules(score_value=0.0)
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            p = _make_random_image(tmp_path / "zero.png")
            results = score_photos([p], device="cpu")
        assert results[0].aesthetic_score == 0.0

    def test_score_boundary_ten(self, tmp_path):
        """模型输出 1.0 即 1.0 分（不缩放）"""
        mock_pyiqa, mock_torch = _build_mock_modules(score_value=1.0)
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            p = _make_random_image(tmp_path / "ten.png")
            results = score_photos([p], device="cpu")
        assert results[0].aesthetic_score == 1.0

    def test_score_clamped_above_ten(self, tmp_path):
        """TOPIQ raw 分超 10 时应钳制到 10.0（防分数体系崩坏）"""
        mock_pyiqa, mock_torch = _build_mock_modules(score_value=12.0)  # raw 12.0
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            p = _make_random_image(tmp_path / "over.png")
            results = score_photos([p], device="cpu")
        r = results[0]
        assert r.aesthetic_score == 10.0
        assert r.overall_score == 10.0
        assert r.final_score == 10.0

    def test_score_clamped_below_zero(self, tmp_path):
        """TOPIQ raw 分低于 0 时应钳制到 0.0"""
        mock_pyiqa, mock_torch = _build_mock_modules(score_value=-0.5)  # raw -5.0
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            p = _make_random_image(tmp_path / "under.png")
            results = score_photos([p], device="cpu")
        assert results[0].aesthetic_score == 0.0
        assert results[0].overall_score == 0.0

    def test_all_scores_equal_for_topiq(self, tmp_path):
        """TOPIQ-IAA 的四个分值应相同（源码中统一赋值）"""
        mock_pyiqa, mock_torch = _build_mock_modules(score_value=0.5)
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            p = _make_random_image(tmp_path / "equal.png")
            results = score_photos([p], device="cpu")
        r = results[0]
        assert r.aesthetic_score == r.technical_score == r.overall_score == r.final_score


# ---------------------------------------------------------------------------
# 无效路径处理
# ---------------------------------------------------------------------------

class TestInvalidPaths:
    def test_nonexistent_file(self, tmp_path):
        """不存在的文件应返回带 error 的 Layer3Result"""
        mock_pyiqa, mock_torch = _build_mock_modules()
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            fake = tmp_path / "nonexistent.png"
            results = score_photos([fake], device="cpu")

        assert len(results) == 1
        assert results[0].error is not None
        assert results[0].path == str(fake)

    def test_corrupted_file(self, tmp_path):
        """损坏的图片文件应返回 error 而非崩溃"""
        mock_pyiqa, mock_torch = _build_mock_modules()
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            bad = tmp_path / "corrupt.png"
            bad.write_text("this is not a real image")
            results = score_photos([bad], device="cpu")

        assert len(results) == 1
        assert results[0].error is not None

    def test_mixed_valid_and_invalid(self, tmp_path):
        """混合有效和无效路径应分别处理"""
        mock_pyiqa, mock_torch = _build_mock_modules(score_value=0.8)
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            valid = _make_random_image(tmp_path / "valid.png")
            invalid = tmp_path / "missing.png"
            results = score_photos([valid, invalid], device="cpu")

        assert len(results) == 2
        has_score = any(r.aesthetic_score > 0 for r in results)
        has_error = any(r.error is not None for r in results)
        assert has_score, "有效图片应获得评分"
        assert has_error, "无效图片应标记错误"


# ---------------------------------------------------------------------------
# TOPIQ 批次归一化测试
# ---------------------------------------------------------------------------

class TestNormalizeBatchScores:
    def test_batch_normalization_spreads_scores(self):
        """多张不同分数照片应被 p5-p95 拉伸拉开区分度，且保持单调"""
        from core.layer3_aesthetic import _normalize_batch_scores
        from core.models import Layer3Result

        results = [
            Layer3Result(path=f"/x/{i}", filename=f"{i}.jpg", overall_score=s)
            for i, s in enumerate([5.0, 6.0, 7.0, 8.0, 9.0], 1)
        ]
        _normalize_batch_scores(results)
        scores = [r.overall_score for r in results]
        assert scores[0] == 0.0
        assert scores[-1] == 10.0
        assert scores == sorted(scores)  # 保持单调

    def test_single_image_no_change(self):
        """单张图片不应被归一化"""
        from core.layer3_aesthetic import _normalize_batch_scores
        from core.models import Layer3Result

        r = Layer3Result(path="/x/1", filename="1.jpg", overall_score=6.5)
        _normalize_batch_scores([r])
        assert r.overall_score == 6.5

    def test_concentrated_scores_no_change(self):
        """分数分布过于集中时保持不变，避免噪声放大"""
        from core.layer3_aesthetic import _normalize_batch_scores
        from core.models import Layer3Result

        results = [
            Layer3Result(path=f"/x/{i}", filename=f"{i}.jpg", overall_score=6.0)
            for i in range(3)
        ]
        _normalize_batch_scores(results)
        assert all(r.overall_score == 6.0 for r in results)


# ---------------------------------------------------------------------------
# 模型加载容错测试
# ---------------------------------------------------------------------------

class TestModelLoading:
    def test_gpu_failure_triggers_cpu_fallback(self, tmp_path):
        """GPU 模型加载失败应尝试 CPU fallback"""
        mock_model = _make_mock_model(0.6)
        mock_pyiqa = MagicMock()
        mock_pyiqa.create_metric.side_effect = [
            RuntimeError("CUDA out of memory"),
            mock_model,
        ]
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 0

        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            p = _make_random_image(tmp_path / "fallback.png")
            results = score_photos([p], device="auto")

        assert mock_pyiqa.create_metric.call_count == 2
        assert results[0].device == "cpu"

    def test_both_gpu_and_cpu_failure(self, tmp_path):
        """GPU 和 CPU 都加载失败应返回带 error 的结果"""
        mock_pyiqa = MagicMock()
        mock_pyiqa.create_metric.side_effect = RuntimeError("model load failed")
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            p = _make_random_image(tmp_path / "fail.png")
            results = score_photos([p], device="auto")

        assert len(results) == 1
        assert results[0].error is not None
        assert results[0].device == "cpu"


# ---------------------------------------------------------------------------
# TOPIQ 模型模块级缓存测试（按 device 复用，pyiqa 身份校验防 mock 误命中）
# ---------------------------------------------------------------------------

class TestTopiqModelCache:
    def test_same_module_reuses_model(self, tmp_path):
        """同一 pyiqa 模块再次评分应复用缓存模型（create_metric 仅调用一次）"""
        mock_pyiqa, mock_torch = _build_mock_modules(score_value=0.6)
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_topiq
            p1 = _make_random_image(tmp_path / "a.png")
            p2 = _make_random_image(tmp_path / "b.png")
            score_topiq([p1], device="cpu")
            score_topiq([p2], device="cpu")
        assert mock_pyiqa.create_metric.call_count == 1

    def test_different_module_reloads(self, tmp_path):
        """pyiqa 模块身份变化（测试 mock 替换）时缓存应失效并重新加载"""
        from core.layer3_aesthetic import score_topiq
        for i in range(2):
            mock_pyiqa, mock_torch = _build_mock_modules(score_value=0.6)
            with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
                p = _make_random_image(tmp_path / f"img{i}.png")
                score_topiq([p], device="cpu")
                assert mock_pyiqa.create_metric.call_count == 1

    def test_release_topiq_model(self, tmp_path):
        """release_topiq_model 应清空缓存，再次评分重新加载"""
        from core.layer3_aesthetic import release_topiq_model, score_topiq
        mock_pyiqa, mock_torch = _build_mock_modules(score_value=0.6)
        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            p = _make_random_image(tmp_path / "a.png")
            score_topiq([p], device="cpu")
            release_topiq_model("cpu")
            score_topiq([p], device="cpu")
        assert mock_pyiqa.create_metric.call_count == 2


# ---------------------------------------------------------------------------
# VLM 分类标签解析测试
# ---------------------------------------------------------------------------

class TestParseDimensionJson:
    def test_category_extracted(self):
        """合法的 category 应被提取"""
        from core.layer3_aesthetic import _parse_dimension_json
        text = ('{"reason": "主体突出", "composition_score": 8.0, "color_score": 7.5, '
                '"lighting_score": 7.0, "overall_aesthetic_score": 8.0, "category": "风景"}')
        out = _parse_dimension_json(text)
        assert out["category"] == "风景"
        assert out["composition_score"] == 8.0

    def test_category_missing_returns_scores_only(self):
        """缺 category 时分数正常返回，但不含 category 键"""
        from core.layer3_aesthetic import _parse_dimension_json
        text = ('{"reason": "主体突出", "composition_score": 8.0, "color_score": 7.5, '
                '"lighting_score": 7.0, "overall_aesthetic_score": 8.0}')
        out = _parse_dimension_json(text)
        assert "composition_score" in out
        assert "category" not in out

    def test_category_invalid_ignored(self):
        """不在枚举中的 category 应被忽略"""
        from core.layer3_aesthetic import _parse_dimension_json
        text = ('{"reason": "主体突出", "composition_score": 8.0, "color_score": 7.5, '
                '"lighting_score": 7.0, "overall_aesthetic_score": 8.0, "category": "抽象艺术"}')
        out = _parse_dimension_json(text)
        assert "category" not in out

    def test_category_non_string_ignored(self):
        """category 为数字时应被忽略"""
        from core.layer3_aesthetic import _parse_dimension_json
        text = ('{"reason": "主体突出", "composition_score": 8.0, "color_score": 7.5, '
                '"lighting_score": 7.0, "overall_aesthetic_score": 8.0, "category": 123}')
        out = _parse_dimension_json(text)
        assert "category" not in out


# ---------------------------------------------------------------------------
# VLM 人像硬伤钳制测试（程序化兜底，不依赖 LLM 自觉）
# ---------------------------------------------------------------------------

class TestHardFlagClamping:
    def test_hard_flag_clamps_overall_to_6(self):
        """人像硬伤（闭眼/主脸模糊）时综合审美应钳制至 6 分上限"""
        from core.layer3_aesthetic import _apply_dims_to_result
        r = Layer3Result(path="/a.jpg", filename="a.jpg")
        dims = {"composition_score": 8.0, "color_score": 7.5,
                "lighting_score": 7.0, "overall_aesthetic_score": 6.8,
                "reason": "构图不错但闭眼"}
        _apply_dims_to_result(r, dims, hard_flag=True)
        assert r.vlm_overall_score == 6.0
        assert "已钳制至 6 分上限" in r.score_reason

    def test_no_hard_flag_keeps_vlm_score(self):
        """无人像硬伤时综合审美保持 VLM 原分"""
        from core.layer3_aesthetic import _apply_dims_to_result
        r = Layer3Result(path="/a.jpg", filename="a.jpg")
        dims = {"composition_score": 8.0, "color_score": 7.5,
                "lighting_score": 7.0, "overall_aesthetic_score": 6.8,
                "reason": "整体不错"}
        _apply_dims_to_result(r, dims, hard_flag=False)
        assert r.vlm_overall_score == 6.8
        assert r.score_reason == "整体不错"

    def test_hard_flag_below_6_not_raised(self):
        """硬伤照片 VLM 分本就低于 6 时不应被抬高"""
        from core.layer3_aesthetic import _apply_dims_to_result
        r = Layer3Result(path="/a.jpg", filename="a.jpg")
        dims = {"composition_score": 5.0, "color_score": 5.0,
                "lighting_score": 5.0, "overall_aesthetic_score": 5.5}
        _apply_dims_to_result(r, dims, hard_flag=True)
        assert r.vlm_overall_score == 5.5


# ---------------------------------------------------------------------------
# hard_flaws 解析与 VLM 自报硬伤钳制测试
# ---------------------------------------------------------------------------

class TestHardFlawsParsing:
    def test_hard_flaws_extracted_deduped(self):
        """hard_flaws 应被解析且去重保序"""
        from core.layer3_aesthetic import _parse_dimension_json
        text = ('{"reason": "模糊且闭眼", "hard_flaws": ["模糊", "闭眼", "模糊"], '
                '"composition_score": 3.0, "color_score": 5.0, '
                '"lighting_score": 5.0, "overall_aesthetic_score": 6.0}')
        out = _parse_dimension_json(text)
        assert out["hard_flaws"] == ["模糊", "闭眼"]

    def test_hard_flaws_invalid_ignored(self):
        """非字符串/空串的 hard_flaws 应被忽略"""
        from core.layer3_aesthetic import _parse_dimension_json
        text = ('{"reason": "ok", "hard_flaws": [123, "", "  "], '
                '"composition_score": 5.0, "color_score": 5.0, '
                '"lighting_score": 5.0, "overall_aesthetic_score": 6.0}')
        out = _parse_dimension_json(text)
        assert "hard_flaws" not in out

    def test_vlm_hard_flaw_triggers_clamp(self):
        """VLM 自报闭眼应触发钳制（辅助信号，即使 L4 未标记）

        注意：自报"人脸模糊"不再触发钳制（实测被 L4 prompt 注入文案诱导误报，
        人脸模糊仅由 L4 客观阈值把关），见 test_face_blur_no_clamp。
        """
        from core.layer3_aesthetic import _apply_dims_to_result
        r = Layer3Result(path="/a.jpg", filename="a.jpg")
        dims = {"composition_score": 8.0, "color_score": 7.5,
                "lighting_score": 7.0, "overall_aesthetic_score": 6.8,
                "hard_flaws": ["闭眼"]}
        _apply_dims_to_result(r, dims, hard_flag=False)  # L4 未标记
        assert r.vlm_overall_score == 6.0  # VLM 自报闭眼 → 钳制
        assert "闭眼" in r.hard_flaws

    def test_face_blur_no_clamp(self):
        """VLM 自报"人脸模糊"不再触发 6 分钳制（L4 客观数据已另行把关）"""
        from core.layer3_aesthetic import _apply_dims_to_result
        r = Layer3Result(path="/a.jpg", filename="a.jpg")
        dims = {"composition_score": 8.0, "color_score": 7.5,
                "lighting_score": 7.0, "overall_aesthetic_score": 6.8,
                "hard_flaws": ["人脸模糊"]}
        _apply_dims_to_result(r, dims, hard_flag=False)
        assert r.vlm_overall_score == 6.8  # 人脸模糊自报 → 不钳制，保留原分
        assert "人脸模糊" in r.hard_flaws  # 仍记录硬伤（供维度分参考）

    def test_l4_hard_flag_clamp(self):
        """L4 客观硬伤（hard_flag=True）仍触发钳制，不依赖 VLM 自报"""
        from core.layer3_aesthetic import _apply_dims_to_result
        r = Layer3Result(path="/a.jpg", filename="a.jpg")
        dims = {"composition_score": 8.0, "color_score": 7.5,
                "lighting_score": 7.0, "overall_aesthetic_score": 6.8,
                "hard_flaws": []}
        _apply_dims_to_result(r, dims, hard_flag=True)  # L4 客观判定硬伤
        assert r.vlm_overall_score == 6.0

    def test_non_face_flaw_no_clamp(self):
        """非人像硬伤（如过曝）不应触发 6 分钳制"""
        from core.layer3_aesthetic import _apply_dims_to_result
        r = Layer3Result(path="/a.jpg", filename="a.jpg")
        dims = {"composition_score": 8.0, "color_score": 7.5,
                "lighting_score": 7.0, "overall_aesthetic_score": 6.8,
                "hard_flaws": ["过曝"]}
        _apply_dims_to_result(r, dims, hard_flag=False)
        assert r.vlm_overall_score == 6.8

    def test_no_flaw_placeholder_filtered(self):
        """prompt 要求无硬伤时填"无硬伤"占位符，不应被存入 hard_flaws"""
        from core.layer3_aesthetic import _parse_dimension_json
        text = ('{"reason": "整体不错", "hard_flaws": ["无硬伤"], '
                '"composition_score": 5.0, "color_score": 5.0, '
                '"lighting_score": 5.0, "overall_aesthetic_score": 6.0}')
        out = _parse_dimension_json(text)
        assert "hard_flaws" not in out

    def test_enum_whitelist_enforced(self):
        """枚举白名单之外的硬伤字符串（LLM 幻觉项）应被忽略，白名单内保留"""
        from core.layer3_aesthetic import _parse_dimension_json
        text = ('{"reason": "ok", "hard_flaws": ["颜色发灰", "模糊"], '
                '"composition_score": 5.0, "color_score": 5.0, '
                '"lighting_score": 5.0, "overall_aesthetic_score": 6.0}')
        out = _parse_dimension_json(text)
        assert out["hard_flaws"] == ["模糊"]

    def test_hard_flaws_capped_at_5(self):
        """hard_flaws 超过 5 个时应截断（去重保序后取前 5）"""
        from core.layer3_aesthetic import _parse_dimension_json
        text = ('{"reason": "ok", "hard_flaws": ["模糊", "过曝", "欠曝", "噪点明显", "歪斜", "画面杂乱", "主体裁切"], '
                '"composition_score": 5.0, "color_score": 5.0, '
                '"lighting_score": 5.0, "overall_aesthetic_score": 6.0}')
        out = _parse_dimension_json(text)
        assert out["hard_flaws"] == ["模糊", "过曝", "欠曝", "噪点明显", "歪斜"]


# ---------------------------------------------------------------------------
# VLM 综合分批次归一化测试（防分数膨胀）
# ---------------------------------------------------------------------------

class TestNormalizeVlmBatch:
    def test_spreads_scores(self):
        """多张 VLM 分应被 p5-p95 拉伸拉开区分度，保持单调，且最低分不为精确 0（避免被误判为未评分）"""
        from core.layer3_aesthetic import _normalize_vlm_batch
        results = [
            Layer3Result(path=f"/x/{i}", filename=f"{i}.jpg", vlm_overall_score=s)
            for i, s in enumerate([5.0, 6.0, 7.0, 8.0, 9.0], 1)
        ]
        _normalize_vlm_batch(results)
        scores = [r.vlm_overall_score for r in results]
        assert scores[0] == 0.1  # 正下限 0.01*10
        assert scores[-1] == 10.0
        assert scores == sorted(scores)

    def test_single_no_change(self):
        """单张照片不应被归一化"""
        from core.layer3_aesthetic import _normalize_vlm_batch
        r = Layer3Result(path="/x/1", filename="1.jpg", vlm_overall_score=7.5)
        _normalize_vlm_batch([r])
        assert r.vlm_overall_score == 7.5

    def test_zero_scores_skipped(self):
        """vlm_overall_score=0（未评分）的照片不参与归一化"""
        from core.layer3_aesthetic import _normalize_vlm_batch
        r0 = Layer3Result(path="/x/0", filename="0.jpg", vlm_overall_score=0.0)
        r1 = Layer3Result(path="/x/1", filename="1.jpg", vlm_overall_score=8.0)
        _normalize_vlm_batch([r0, r1])
        assert r0.vlm_overall_score == 0.0
        assert r1.vlm_overall_score == 8.0  # 仅 1 个有效分，不归一化

    def test_concentrated_no_change(self):
        """分数分布集中时保持不变，避免噪声放大"""
        from core.layer3_aesthetic import _normalize_vlm_batch
        results = [
            Layer3Result(path=f"/x/{i}", filename=f"{i}.jpg", vlm_overall_score=6.0)
            for i in range(3)
        ]
        _normalize_vlm_batch(results)
        assert all(r.vlm_overall_score == 6.0 for r in results)
