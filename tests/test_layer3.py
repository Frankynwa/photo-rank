"""Layer 3 审美评分单元测试 — mock pyiqa/torch，不依赖 GPU"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from core.models import Layer3Result

# ---------------------------------------------------------------------------
# 辅助：生成临时测试图片
# ---------------------------------------------------------------------------

def _make_test_image(path: Path, size=(64, 64), color=(128, 128, 128)) -> Path:
    img = Image.new("RGB", size, color)
    img.save(str(path))
    return path


def _setup_mocks(mock_score=0.75, model_side_effect=None):
    """创建 pyiqa / torch 的 mock 对象，返回 (mock_pyiqa, mock_torch)"""
    mock_pyiqa = MagicMock()
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    mock_model = MagicMock()
    if model_side_effect:
        mock_model.return_value.item.side_effect = model_side_effect
    else:
        mock_model.return_value.item.return_value = mock_score
    mock_pyiqa.create_metric.return_value = mock_model
    return mock_pyiqa, mock_torch


# ---------------------------------------------------------------------------
# score_photos 正常调用测试
# ---------------------------------------------------------------------------

class TestScorePhotos:
    def test_score_photos_normal(self, tmp_path: Path):
        """正常图片应返回 Layer3Result 列表，分值在合理范围内"""
        mock_pyiqa, mock_torch = _setup_mocks(mock_score=0.75)

        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            paths = [_make_test_image(tmp_path / f"img{i}.png") for i in range(3)]
            results = score_photos([str(p) for p in paths], device="cpu")

        assert len(results) == 3
        for r in results:
            assert isinstance(r, Layer3Result)
            assert r.error is None
            assert r.device == "cpu"
            # 0.75 * 10 = 7.5
            assert r.aesthetic_score == pytest.approx(7.5, abs=0.01)
            assert r.final_score == pytest.approx(7.5, abs=0.01)

    def test_score_photos_empty列表(self):
        """空列表应返回空结果"""
        mock_pyiqa, mock_torch = _setup_mocks()

        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            results = score_photos([], device="cpu")

        assert results == []

    def test_score_photos_invalid_path(self, tmp_path: Path):
        """无效路径应产生 error 字段"""
        mock_pyiqa, mock_torch = _setup_mocks()

        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            results = score_photos([str(tmp_path / "nonexistent.png")], device="cpu")

        assert len(results) == 1
        assert results[0].error is not None
        assert results[0].filename == "nonexistent.png"

    def test_score_range_max(self, tmp_path: Path):
        """模型返回 1.0 时，映射后分值应为 10.0"""
        mock_pyiqa, mock_torch = _setup_mocks(mock_score=1.0)

        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            path = _make_test_image(tmp_path / "max.png")
            results = score_photos([str(path)], device="cpu")

        assert results[0].aesthetic_score == pytest.approx(10.0, abs=0.01)

    def test_score_range_min(self, tmp_path: Path):
        """模型返回 0.0 时，分值应为 0.0"""
        mock_pyiqa, mock_torch = _setup_mocks(mock_score=0.0)

        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            path = _make_test_image(tmp_path / "min.png")
            results = score_photos([str(path)], device="cpu")

        assert results[0].aesthetic_score == pytest.approx(0.0, abs=0.01)

    def test_model_load_failure_cpu_fallback(self, tmp_path: Path):
        """GPU 模型加载失败时应 fallback 到 CPU"""
        mock_pyiqa = MagicMock()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        mock_model = MagicMock()
        mock_model.return_value.item.return_value = 0.5
        # 第一次调用（GPU）抛异常，第二次（CPU fallback）成功
        mock_pyiqa.create_metric.side_effect = [
            RuntimeError("CUDA out of memory"),
            mock_model,
        ]

        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            path = _make_test_image(tmp_path / "fallback.png")
            results = score_photos([str(path)], device="auto")

        assert len(results) == 1
        assert results[0].device == "cpu"
        assert results[0].error is None

    def test_model_load_total_failure(self, tmp_path: Path):
        """GPU 和 CPU 模型都加载失败时，所有结果应带 error"""
        mock_pyiqa = MagicMock()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_pyiqa.create_metric.side_effect = RuntimeError("total failure")

        with patch.dict(sys.modules, {"pyiqa": mock_pyiqa, "torch": mock_torch}):
            from core.layer3_aesthetic import score_photos
            path = _make_test_image(tmp_path / "fail.png")
            results = score_photos([str(path)], device="auto")

        assert len(results) == 1
        assert results[0].error is not None
