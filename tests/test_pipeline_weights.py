"""融合评分权重单元测试 — 验证无脸照片的权重重分配逻辑"""
from core.pipeline import _effective_weights


class TestEffectiveWeights:
    def test_has_face_keeps_standard_weights(self):
        """有人脸照片应使用标准 FUSION_WEIGHTS"""
        w = _effective_weights(has_face=True)
        assert w["face"] > 0
        assert sum(w.values()) == 1.0

    def test_no_face_redistributes(self):
        """无脸照片人脸权重应为 0，其余维度按比例放大，总和仍为 1.0"""
        w = _effective_weights(has_face=False)
        assert w.get("face", 0.0) == 0.0  # face 键被移除，取值恒为 0
        assert sum(w.values()) == 1.0
        assert w["vlm"] > 0.45  # 有人脸时的 0.45 被放大
        assert w["topiq"] > 0.30
        assert w["objective"] > 0.10

    def test_no_face_preserves_relative_order(self):
        """重分配后各维度的相对大小关系保持不变"""
        w = _effective_weights(has_face=False)
        assert w["vlm"] > w["topiq"] > w["objective"]

    def test_no_face_landscape_not_penalized(self):
        """无脸照片的分数上限应与人像照片一致（权重总和都为 1.0）"""
        w_face = _effective_weights(has_face=True)
        w_no_face = _effective_weights(has_face=False)
        assert sum(w_face.values()) == sum(w_no_face.values())
