"""Layer 1 客观指标单元测试"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.layer1_objective import (
    ObjectiveScore,
    _analyze_metrics,
    analyze_image,
    batch_analyze,
    load_image,
)

# ---------------------------------------------------------------------------
# _analyze_metrics 单元测试（直接传 numpy array，无需文件 I/O）
# ---------------------------------------------------------------------------

class TestAnalyzeMetrics:
    """测试 _analyze_metrics(image: np.ndarray) -> tuple"""

    def _make_bgr(self, h: int = 100, w: int = 100, color=(128, 128, 128)) -> np.ndarray:
        """生成纯色 BGR 图片"""
        img = np.full((h, w, 3), color, dtype=np.uint8)
        return img

    def test_clear_image_returns_positive_sharpness(self):
        """清晰（含纹理/噪声）图片的 sharpness 应为正值"""
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
        sharpness, exposure, over_ratio, under_ratio, noise = _analyze_metrics(img)
        assert sharpness > 0, "随机纹理图片的清晰度应 > 0"

    def test_pure_color_image_zero_sharpness(self):
        """纯色图片没有边缘，Laplacian 方差应为 0"""
        img = self._make_bgr(color=(100, 100, 100))
        sharpness, *_ = _analyze_metrics(img)
        assert sharpness == 0.0, "纯色图片清晰度应为 0"

    def test_normal_exposure_range(self):
        """中等亮度图片的 exposure_score 应在 [0, 1] 且接近 1"""
        img = self._make_bgr(color=(128, 128, 128))  # 中灰
        _, exposure, over_ratio, under_ratio, _ = _analyze_metrics(img)
        assert 0.0 <= exposure <= 1.0
        # 中灰 → mean_brightness=128 → 落在合理区间 → exposure_score = 1.0
        assert exposure == pytest.approx(1.0, abs=0.02)
        assert over_ratio == 0.0
        assert under_ratio == 0.0

    def test_exposure_band_mid_brightness_full_score(self):
        """亮度 100（合理区间内）应满分，不再按远离 128 扣分"""
        img = self._make_bgr(color=(100, 100, 100))
        _, exposure, _, _, _ = _analyze_metrics(img)
        assert exposure == pytest.approx(1.0, abs=0.02)

    def test_exposure_dark_night_penalty_softened(self):
        """夜景均值 30 不再被重罚（旧公式 0.23，新区间法明显更高）"""
        img = self._make_bgr(color=(30, 30, 30))
        _, exposure, _, _, _ = _analyze_metrics(img)
        assert exposure > 0.6, f"夜景曝光分应 > 0.6，实际 {exposure}"

    def test_overexposed_image(self):
        """接近全白的图片 over_ratio 应很高"""
        img = self._make_bgr(color=(255, 255, 255))
        _, _, over_ratio, _, _ = _analyze_metrics(img)
        assert over_ratio > 0.3, "全白图片 over_ratio 应 > 0.3"

    def test_underexposed_image(self):
        """接近全黑的图片 under_ratio 应很高"""
        img = self._make_bgr(color=(0, 0, 0))
        _, _, _, under_ratio, _ = _analyze_metrics(img)
        assert under_ratio > 0.5, "全黑图片 under_ratio 应 > 0.5"

    def test_noise_score_range(self):
        """noise_score 应在 [0, 1]"""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        *_, noise = _analyze_metrics(img)
        assert 0.0 <= noise <= 1.0

    def test_noise_score_pure_color(self):
        """纯色图片 Laplacian std=0 → snr=999 → noise_score 截断到 1.0"""
        img = self._make_bgr(color=(80, 80, 80))
        *_, noise = _analyze_metrics(img)
        # noise_std=0 → snr=999 → min(999/100, 1.0) = 1.0
        assert noise == pytest.approx(1.0, abs=0.01)

    def test_returns_five_element_tuple(self):
        """返回值应为 5 元组"""
        img = self._make_bgr()
        result = _analyze_metrics(img)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# ObjectiveScore dataclass 测试
# ---------------------------------------------------------------------------

class TestObjectiveScore:
    def test_default_values(self):
        score = ObjectiveScore()
        assert score.sharpness == 0
        assert score.overall == 0
        assert score.rejected is False
        assert score.reject_reason == ""

    def test_custom_values(self):
        score = ObjectiveScore(sharpness=200.0, is_blurry=False, rejected=False)
        assert score.sharpness == 200.0
        assert score.is_blurry is False


# ---------------------------------------------------------------------------
# load_image / analyze_image 文件级测试（用临时 PNG）
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_load_valid_png(self, tmp_path: Path):
        """写入一张有效 PNG，load_image 应返回非 None"""
        img = np.full((50, 50, 3), 120, dtype=np.uint8)
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)
        result = load_image(path)
        assert result is not None
        assert result.shape == (50, 50, 3)

    def test_load_nonexistent_returns_none(self, tmp_path: Path):
        """不存在的文件应返回 None"""
        result = load_image(tmp_path / "nonexistent.jpg")
        assert result is None


class TestAnalyzeImage:
    def test_analyze_valid_image(self, tmp_path: Path):
        """正常图片应返回 ObjectiveScore，且 rejected=False"""
        rng = np.random.default_rng(7)
        img = rng.integers(50, 200, (100, 100, 3), dtype=np.uint8)
        path = tmp_path / "normal.png"
        cv2.imwrite(str(path), img)

        score = analyze_image(path)
        assert isinstance(score, ObjectiveScore)
        assert score.sharpness > 0
        assert 0.0 <= score.exposure <= 1.0
        assert 0.0 <= score.noise <= 1.0

    def test_analyze_nonexistent_returns_rejected(self, tmp_path: Path):
        """不存在的文件应返回 rejected=True"""
        score = analyze_image(tmp_path / "missing.png")
        assert score.rejected is True
        assert "无法读取" in score.reject_reason

    def test_blurry_image_detected(self, tmp_path: Path):
        """全黑（纯色）图片 sharpness=0，应被判定为模糊"""
        img = np.full((100, 100, 3), 50, dtype=np.uint8)
        path = tmp_path / "blurry.png"
        cv2.imwrite(str(path), img)

        score = analyze_image(path)
        assert score.is_blurry
        assert "模糊" in score.reject_reason

    def test_dark_night_scene_not_rejected(self, tmp_path: Path):
        """夜景照片（暗部占比高但仍有亮部细节）不应被误判为欠曝淘汰"""
        rng = np.random.default_rng(7)
        img = np.full((100, 100, 3), 10, dtype=np.uint8)  # 65% 暗部（街景阴影）
        img[:35, :100] = rng.integers(150, 220, (35, 100, 3), dtype=np.uint8)  # 35% 灯火
        path = tmp_path / "night.png"
        cv2.imwrite(str(path), img)

        score = analyze_image(path)
        assert not score.is_underexposed, "暗部 65% 的夜景不应被误淘汰"
        assert not score.is_blurry


class TestBatchAnalyze:
    def test_batch_analyze_multiple(self, tmp_path: Path):
        """批量分析多张图片，结果数量与输入一致"""
        paths = []
        for i in range(3):
            img = np.full((60, 60, 3), 80 + i * 30, dtype=np.uint8)
            p = tmp_path / f"img{i}.png"
            cv2.imwrite(str(p), img)
            paths.append(p)

        results = batch_analyze(paths)
        assert len(results) == 3
        for r in results:
            assert r.path  # Layer1Result Pydantic 模型
            assert r.filename
            assert r.sharpness >= 0

    def test_batch_analyze_empty_list(self):
        """空列表应返回空结果"""
        results = batch_analyze([])
        assert results == []
