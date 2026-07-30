"""Layer 4 人脸质量单元测试"""
from unittest.mock import patch

from core.layer4_face import FaceAnalyzer, FaceQuality


class TestFaceQuality:
    """FaceQuality 数据类测试"""

    def test_default_values(self):
        fq = FaceQuality()
        assert fq.face_count == 0
        assert fq.has_face is False
        assert fq.overall_face_score == 0.0

    def test_custom_values(self):
        fq = FaceQuality(face_count=2, has_face=True, face_clarity=0.8)
        assert fq.face_count == 2
        assert fq.has_face is True
        assert fq.face_clarity == 0.8


class TestFaceAnalyzer:
    """FaceAnalyzer 初始化测试"""

    def test_init_state(self):
        analyzer = FaceAnalyzer()
        assert analyzer.detector is None
        assert analyzer.initialized is False
        assert analyzer._init_attempts == 0

    @patch("core.layer4_face.FaceAnalyzer._init_mediapipe")
    def test_analyze_face_no_detector(self, mock_init):
        """detector 为 None 时返回默认 FaceQuality"""
        analyzer = FaceAnalyzer()
        result = analyzer.analyze_face("dummy.jpg")
        assert result.face_count == 0
        assert result.has_face is False

    def test_max_attempts_guard(self):
        """超过最大尝试次数后不再重试"""
        analyzer = FaceAnalyzer()
        analyzer._init_attempts = 3
        analyzer._init_mediapipe()
        assert analyzer.detector is None
        assert analyzer.initialized is False
