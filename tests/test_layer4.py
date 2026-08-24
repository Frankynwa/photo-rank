"""Layer 4 人脸质量单元测试 — mock mediapipe / cv2，不依赖真实模型"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from core.models import Layer4Result

# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

def _make_solid_image(path: Path, color=(128, 128, 128), size=(200, 200)):
    img = Image.new("RGB", size, color)
    img.save(str(path))
    return path


def _make_random_image(path: Path, size=(200, 200)):
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(str(path))
    return path


class _FakeLandmark:
    """模拟 MediaPipe 人脸关键点"""
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakeBlendshape:
    """模拟 MediaPipe blendshape"""
    def __init__(self, category_name: str, score: float):
        self.category_name = category_name
        self.score = score


def _build_mock_detection(
    face_count: int = 0,
    blink: bool = False,
    smile: float = 0.0,
):
    """构建模拟的 MediaPipe 检测结果

    face_count=0 → 无人脸
    face_count>0 → 生成对应数量的 landmarks 和 blendshapes
    """
    # 构建 mock mediapipe 模块树
    mock_mp = MagicMock()
    mock_mp.Image.return_value = MagicMock()
    mock_mp.ImageFormat.SRGB = 1

    # mediapipe.tasks.python.vision
    mock_vision = MagicMock()
    mock_detector = MagicMock()
    mock_vision.FaceLandmarker.create_from_options.return_value = mock_detector

    # 构建检测结果
    if face_count == 0:
        result = MagicMock()
        result.face_landmarks = []
        result.face_blendshapes = []
    else:
        # 为每张人脸生成 landmarks（归一化坐标 0.3~0.7 模拟面部区域）
        landmarks = [
            _FakeLandmark(0.3, 0.2), _FakeLandmark(0.7, 0.2),
            _FakeLandmark(0.3, 0.8), _FakeLandmark(0.7, 0.8),
            _FakeLandmark(0.5, 0.5),
        ]
        face_landmarks = [landmarks for _ in range(face_count)]

        # blendshapes
        blendshapes = []
        for _ in range(face_count):
            bs = [
                _FakeBlendshape("eyeBlinkLeft", 0.8 if blink else 0.1),
                _FakeBlendshape("eyeBlinkRight", 0.8 if blink else 0.1),
                _FakeBlendshape("mouthSmileLeft", smile),
                _FakeBlendshape("mouthSmileRight", smile),
            ]
            blendshapes.append(bs)

        result = MagicMock()
        result.face_landmarks = face_landmarks
        result.face_blendshapes = blendshapes

    mock_detector.detect.return_value = result

    # 组装模块树
    mock_tasks_python = MagicMock()
    mock_tasks_python.vision = mock_vision

    mock_tasks = MagicMock()
    mock_tasks.python = mock_tasks_python

    mock_mp.tasks = mock_tasks

    return mock_mp, mock_detector


def _install_mediapipe_mock(mock_mp):
    """将 mock mediapipe 模块树注入 sys.modules"""
    modules = {
        "mediapipe": mock_mp,
        "mediapipe.tasks": mock_mp.tasks,
        "mediapipe.tasks.python": mock_mp.tasks.python,
        "mediapipe.tasks.python.vision": mock_mp.tasks.python.vision,
    }
    return patch.dict(sys.modules, modules)


def _install_cv2_mock():
    """注入 mock cv2"""
    mock_cv2 = MagicMock()
    mock_cv2.COLOR_RGB2GRAY = 7
    mock_cv2.CV_64F = 6
    # cv2.cvtColor 返回灰度图
    mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
    # cv2.Laplacian 返回一个 array，.var() 返回方差
    lap_result = MagicMock()
    lap_result.var.return_value = 25000.0  # 25000/500 = 50 → 截断到 1.0
    mock_cv2.Laplacian.return_value = lap_result
    return patch.dict(sys.modules, {"cv2": mock_cv2})


# ---------------------------------------------------------------------------
# FaceQuality dataclass 测试
# ---------------------------------------------------------------------------

class TestFaceQuality:
    def test_default_values(self):
        """默认值应全部为零/False"""
        from core.layer4_face import FaceQuality
        fq = FaceQuality()
        assert fq.face_count == 0
        assert fq.has_face is False
        assert fq.blink_detected is False
        assert fq.expression_score == 0.0
        assert fq.face_clarity == 0.0
        assert fq.overall_face_score == 0.0
        assert fq.face_ratio == 0.0
        assert fq.second_face_ratio == 0.0
        assert fq.main_face_blink is False

    def test_custom_values(self):
        """自定义值应正确赋值"""
        from core.layer4_face import FaceQuality
        fq = FaceQuality(
            face_count=2, has_face=True, blink_detected=True,
            expression_score=0.8, face_clarity=0.9, overall_face_score=0.7,
            face_ratio=0.24, second_face_ratio=1.0, main_face_blink=True,
        )
        assert fq.face_count == 2
        assert fq.has_face is True
        assert fq.overall_face_score == 0.7
        assert fq.face_ratio == 0.24
        assert fq.second_face_ratio == 1.0
        assert fq.main_face_blink is True


# ---------------------------------------------------------------------------
# FaceAnalyzer 初始化测试
# ---------------------------------------------------------------------------

class TestFaceAnalyzerInit:
    def test_initial_state(self):
        """新实例应处于未初始化状态"""
        from core.layer4_face import FaceAnalyzer
        fa = FaceAnalyzer()
        assert fa.detector is None
        assert fa.initialized is False
        assert fa._init_attempts == 0
        assert fa._max_attempts == 3

    def test_init_with_mock_mediapipe(self, tmp_path):
        """mock mediapipe 可用时应成功初始化"""
        mock_mp, _ = _build_mock_detection(face_count=0)
        with _install_mediapipe_mock(mock_mp), \
             patch.object(
                 __import__("core.layer4_face", fromlist=["FaceAnalyzer"]).FaceAnalyzer,
                 "_get_model_path", return_value="/fake/model.task",
             ):
            from core.layer4_face import FaceAnalyzer
            fa = FaceAnalyzer()
            fa._init_mediapipe()
            assert fa.initialized is True
            assert fa.detector is not None

    def test_init_failure_increments_attempts(self):
        """初始化失败应递增 _init_attempts"""
        from core.layer4_face import FaceAnalyzer
        fa = FaceAnalyzer()
        with patch.object(fa, "_get_model_path", return_value=""):
            fa._init_mediapipe()
        assert fa._init_attempts == 1
        assert fa.initialized is False

    def test_max_attempts_exceeded(self):
        """超过最大尝试次数后应停止重试"""
        from core.layer4_face import FaceAnalyzer
        fa = FaceAnalyzer()
        fa._init_attempts = 3  # 已达上限
        fa._init_mediapipe()
        assert fa._init_attempts == 3  # 不应再增加
        assert fa.initialized is False


# ---------------------------------------------------------------------------
# analyze_face 测试
# ---------------------------------------------------------------------------

class TestAnalyzeFace:
    def test_no_face_detected(self, tmp_path):
        """无人脸图片应返回默认 FaceQuality"""
        mock_mp, _ = _build_mock_detection(face_count=0)
        with _install_mediapipe_mock(mock_mp), \
             patch.object(
                 __import__("core.layer4_face", fromlist=["FaceAnalyzer"]).FaceAnalyzer,
                 "_get_model_path", return_value="/fake/model.task",
             ):
            from core.layer4_face import FaceAnalyzer
            fa = FaceAnalyzer()
            p = _make_solid_image(tmp_path / "no_face.png")
            quality = fa.analyze_face(p)

        assert quality.has_face is False
        assert quality.face_count == 0

    def test_single_face_detected(self, tmp_path):
        """检测到单张人脸应返回有效评分"""
        mock_mp, _ = _build_mock_detection(face_count=1, smile=0.6, blink=False)
        with _install_mediapipe_mock(mock_mp), \
             _install_cv2_mock(), \
             patch.object(
                 __import__("core.layer4_face", fromlist=["FaceAnalyzer"]).FaceAnalyzer,
                 "_get_model_path", return_value="/fake/model.task",
             ):
            from core.layer4_face import FaceAnalyzer
            fa = FaceAnalyzer()
            p = _make_random_image(tmp_path / "face.png")
            quality = fa.analyze_face(p)

        assert quality.has_face is True
        assert quality.face_count == 1
        assert quality.expression_score == pytest.approx(0.6, abs=0.01)
        assert quality.blink_detected is False
        # mock landmarks: x 0.3-0.7, y 0.2-0.8 → 面积 80*120=9600, 图片 200*200
        assert quality.face_ratio == pytest.approx(0.24, abs=0.01)
        assert quality.second_face_ratio == 0.0  # 仅一张脸
        assert quality.main_face_blink is False

    def test_blink_detected(self, tmp_path):
        """眨眼应被检测到"""
        mock_mp, _ = _build_mock_detection(face_count=1, blink=True, smile=0.0)
        with _install_mediapipe_mock(mock_mp), \
             _install_cv2_mock(), \
             patch.object(
                 __import__("core.layer4_face", fromlist=["FaceAnalyzer"]).FaceAnalyzer,
                 "_get_model_path", return_value="/fake/model.task",
             ):
            from core.layer4_face import FaceAnalyzer
            fa = FaceAnalyzer()
            p = _make_random_image(tmp_path / "blink.png")
            quality = fa.analyze_face(p)

        assert quality.blink_detected is True

    def test_multiple_faces(self, tmp_path):
        """多人脸检测应正确计数"""
        mock_mp, _ = _build_mock_detection(face_count=3, smile=0.3)
        with _install_mediapipe_mock(mock_mp), \
             _install_cv2_mock(), \
             patch.object(
                 __import__("core.layer4_face", fromlist=["FaceAnalyzer"]).FaceAnalyzer,
                 "_get_model_path", return_value="/fake/model.task",
             ):
            from core.layer4_face import FaceAnalyzer
            fa = FaceAnalyzer()
            p = _make_random_image(tmp_path / "multi.png")
            quality = fa.analyze_face(p)

        assert quality.face_count == 3
        assert quality.has_face is True
        # 所有 mock 脸面积相同 → 面积均匀（疑似合影特征）
        assert quality.second_face_ratio == pytest.approx(1.0, abs=0.01)

    def test_quality_scores_range(self, tmp_path):
        """质量评分应在 [0, 1] 范围内"""
        mock_mp, _ = _build_mock_detection(face_count=1, smile=0.5)
        with _install_mediapipe_mock(mock_mp), \
             _install_cv2_mock(), \
             patch.object(
                 __import__("core.layer4_face", fromlist=["FaceAnalyzer"]).FaceAnalyzer,
                 "_get_model_path", return_value="/fake/model.task",
             ):
            from core.layer4_face import FaceAnalyzer
            fa = FaceAnalyzer()
            p = _make_random_image(tmp_path / "range.png")
            quality = fa.analyze_face(p)

        assert 0.0 <= quality.expression_score <= 1.0
        assert 0.0 <= quality.face_clarity <= 1.0
        assert 0.0 <= quality.overall_face_score <= 1.0

    def test_nonexistent_file(self, tmp_path):
        """不存在的文件应返回默认 FaceQuality"""
        mock_mp, _ = _build_mock_detection(face_count=0)
        with _install_mediapipe_mock(mock_mp), \
             patch.object(
                 __import__("core.layer4_face", fromlist=["FaceAnalyzer"]).FaceAnalyzer,
                 "_get_model_path", return_value="/fake/model.task",
             ):
            from core.layer4_face import FaceAnalyzer
            fa = FaceAnalyzer()
            quality = fa.analyze_face(tmp_path / "missing.png")

        assert quality.has_face is False

    def test_detector_unavailable(self, tmp_path):
        """detector 为 None 时应返回默认 FaceQuality"""
        from core.layer4_face import FaceAnalyzer
        fa = FaceAnalyzer()
        # 不初始化 → detector 为 None
        p = _make_solid_image(tmp_path / "no_init.png")
        quality = fa.analyze_face(p)
        assert quality.has_face is False
        assert quality.face_count == 0


# ---------------------------------------------------------------------------
# batch_analyze 测试
# ---------------------------------------------------------------------------

class TestBatchAnalyze:
    def test_batch_multiple_images(self, tmp_path):
        """批量分析多张图片，结果数量与输入一致"""
        mock_mp, _ = _build_mock_detection(face_count=0)
        with _install_mediapipe_mock(mock_mp), \
             patch.object(
                 __import__("core.layer4_face", fromlist=["FaceAnalyzer"]).FaceAnalyzer,
                 "_get_model_path", return_value="/fake/model.task",
             ):
            from core.layer4_face import FaceAnalyzer
            fa = FaceAnalyzer()
            paths = [_make_solid_image(tmp_path / f"b{i}.png") for i in range(3)]
            results = fa.batch_analyze(paths)

        assert len(results) == 3
        for r in results:
            assert isinstance(r, Layer4Result)
            assert r.path
            assert r.filename

    def test_batch_empty_list(self):
        """空列表应返回空结果"""
        from core.layer4_face import FaceAnalyzer
        fa = FaceAnalyzer()
        results = fa.batch_analyze([])
        assert results == []

    def test_batch_result_fields(self, tmp_path):
        """结果应包含正确的 path 和 filename"""
        mock_mp, _ = _build_mock_detection(face_count=1, smile=0.4)
        with _install_mediapipe_mock(mock_mp), \
             _install_cv2_mock(), \
             patch.object(
                 __import__("core.layer4_face", fromlist=["FaceAnalyzer"]).FaceAnalyzer,
                 "_get_model_path", return_value="/fake/model.task",
             ):
            from core.layer4_face import FaceAnalyzer
            fa = FaceAnalyzer()
            p = _make_random_image(tmp_path / "fields.png")
            results = fa.batch_analyze([p])

        r = results[0]
        assert r.path == str(p)
        assert r.filename == "fields.png"
        assert r.has_face is True
        assert r.face_count == 1


# ---------------------------------------------------------------------------
# to_face_prompt_summary 测试（六档分档文案）
# ---------------------------------------------------------------------------

class TestToFacePromptSummary:
    def test_no_face(self):
        """无人脸 → 聚焦场景文案"""
        from core.layer4_face import to_face_prompt_summary
        r = Layer4Result(path="/a.jpg", filename="a.jpg")
        assert to_face_prompt_summary(r) == "【人脸检测信息】未检测到人脸，请聚焦场景本身。"

    def test_single_face(self):
        """单张人脸 → 主脸信息"""
        from core.layer4_face import to_face_prompt_summary
        r = Layer4Result(
            path="/a.jpg", filename="a.jpg", has_face=True, face_count=1,
            main_face_blink=False, face_clarity=0.9, face_ratio=0.24,
        )
        text = to_face_prompt_summary(r)
        assert "检测到 1 张人脸" in text
        assert "主脸闭眼：否" in text
        assert "主脸清晰度：清晰" in text
        assert "人脸占比：半身" in text

    def test_group_uniform_9(self):
        """多人均匀 → 疑似合影 + 存在闭眼者"""
        from core.layer4_face import to_face_prompt_summary
        r = Layer4Result(
            path="/a.jpg", filename="a.jpg", has_face=True, face_count=5,
            blink_detected=True, face_clarity=0.9, face_ratio=0.24,
            second_face_ratio=1.0,
        )
        text = to_face_prompt_summary(r)
        assert "面积均匀分布（疑似合影）" in text
        assert "存在闭眼者：是" in text

    def test_group_uneven_9(self):
        """多人主次分明 → 疑似单主体+背景人群 + 主脸闭眼"""
        from core.layer4_face import to_face_prompt_summary
        r = Layer4Result(
            path="/a.jpg", filename="a.jpg", has_face=True, face_count=4,
            main_face_blink=True, face_clarity=0.8, face_ratio=0.18,
            second_face_ratio=0.3,
        )
        text = to_face_prompt_summary(r)
        assert "主次分明（疑似单主体+背景人群）" in text
        assert "主脸闭眼：是" in text

    def test_crowd_uniform_10plus(self):
        """10+ 人脸均匀 → 多人场景疑似合影"""
        from core.layer4_face import to_face_prompt_summary
        r = Layer4Result(
            path="/a.jpg", filename="a.jpg", has_face=True, face_count=12,
            blink_detected=False, face_clarity=0.7, face_ratio=0.20,
            second_face_ratio=1.0,
        )
        text = to_face_prompt_summary(r)
        assert "多人场景，面积均匀，疑似合影" in text

    def test_crowd_uneven_10plus(self):
        """10+ 人脸主次分明 → 疑似单主体+环境人群"""
        from core.layer4_face import to_face_prompt_summary
        r = Layer4Result(
            path="/a.jpg", filename="a.jpg", has_face=True, face_count=12,
            main_face_blink=False, face_clarity=0.9, face_ratio=0.05,
            second_face_ratio=0.2,
        )
        text = to_face_prompt_summary(r)
        assert "多人场景，主次分明，疑似单主体+环境人群" in text
        assert "主脸占比：环境人像" in text

    def test_mid_distribution_goes_group(self):
        """中间态 0.5-0.7 归入合影档（保守处理）"""
        from core.layer4_face import to_face_prompt_summary
        r = Layer4Result(
            path="/a.jpg", filename="a.jpg", has_face=True, face_count=3,
            face_clarity=0.9, face_ratio=0.24, second_face_ratio=0.6,
        )
        text = to_face_prompt_summary(r)
        assert "疑似合影" in text

    def test_low_clarity_neutral_wording(self):
        """低清晰度不再输出负面定性（墨镜/侧脸/误检天然失真），改中性提示"""
        from core.layer4_face import to_face_prompt_summary
        for clarity_val in (0.12, 0.06):  # 原"轻微模糊"/"明显模糊"档
            r = Layer4Result(
                path="/a.jpg", filename="a.jpg", has_face=True, face_count=1,
                face_clarity=clarity_val, face_ratio=0.24,
            )
            text = to_face_prompt_summary(r)
            assert "清晰度指标偏低，请依据画面自行判断" in text
            assert "模糊" not in text


# ---------------------------------------------------------------------------
# analyze_faces 模块级函数测试（全局实例复用）
# ---------------------------------------------------------------------------

class TestAnalyzeFaces:
    def test_analyze_faces_basic(self, tmp_path):
        """模块级 analyze_faces 应正常工作"""
        mock_mp, _ = _build_mock_detection(face_count=0)
        # 重置全局实例
        import core.layer4_face as face_module
        old_analyzer = face_module._global_analyzer
        face_module._global_analyzer = None

        try:
            with _install_mediapipe_mock(mock_mp), \
                 patch.object(face_module.FaceAnalyzer, "_get_model_path", return_value="/fake/model.task"):
                p = _make_solid_image(tmp_path / "module_func.png")
                results = face_module.analyze_faces([p])
            assert len(results) == 1
            assert isinstance(results[0], Layer4Result)
        finally:
            face_module._global_analyzer = old_analyzer

    def test_analyze_faces_empty(self):
        """空列表应返回空结果"""
        import core.layer4_face as face_module
        old_analyzer = face_module._global_analyzer
        face_module._global_analyzer = None
        try:
            results = face_module.analyze_faces([])
            assert results == []
        finally:
            face_module._global_analyzer = old_analyzer

    def test_reuses_global_instance(self, tmp_path):
        """多次调用应复用全局实例"""
        mock_mp, _ = _build_mock_detection(face_count=0)
        import core.layer4_face as face_module
        old_analyzer = face_module._global_analyzer
        face_module._global_analyzer = None

        try:
            with _install_mediapipe_mock(mock_mp), \
                 patch.object(face_module.FaceAnalyzer, "_get_model_path", return_value="/fake/model.task"):
                p = _make_solid_image(tmp_path / "reuse.png")
                face_module.analyze_faces([p])
                first = face_module._global_analyzer
                face_module.analyze_faces([p])
                second = face_module._global_analyzer
            assert first is second, "应复用同一个全局 FaceAnalyzer 实例"
        finally:
            face_module._global_analyzer = old_analyzer
