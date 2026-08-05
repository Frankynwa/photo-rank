"""融合评分权重与离群诊断单元测试 — 权重重分配 + VLM 有效分/TOPIQ 离群诊断"""
import pytest

from core.models import Layer3Result
from core.pipeline import _effective_weights, _vlm_effective, _vlm_topiq_divergence

# 与 pipeline 中平台默认维度权重一致（face 不参与维度加权和）
_WEIGHTS = {"composition": 0.35, "color": 0.30, "lighting": 0.20, "face": 0.15}
_DIM_W_SUM = _WEIGHTS["composition"] + _WEIGHTS["color"] + _WEIGHTS["lighting"]


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

    def test_vlm_scored_removes_face_weight(self):
        """VLM 已评分时人脸权重归零，重分配到其余维度（防双重计数）"""
        w = _effective_weights(has_face=True, has_vlm=True)
        assert w.get("face", 0.0) == 0.0
        assert sum(w.values()) == 1.0
        assert w["vlm"] > 0.45
        assert w["topiq"] > 0.30
        assert w["objective"] > 0.10

    def test_vlm_scored_no_face_no_extra_redistribution(self):
        """VLM 已评分且无脸：face 已在第一分支移除，重分配分支不重复执行，总和仍为 1.0"""
        w = _effective_weights(has_face=False, has_vlm=True)
        assert w.get("face", 0.0) == 0.0
        assert sum(w.values()) == 1.0
        assert w["vlm"] > 0.45


class TestVlmEffective:
    """VLM 有效分（0.7×维度加权 + 0.3×综合）——离群诊断的输入"""

    def test_vlm_scored_returns_effective_score(self):
        """VLM 已评分应返回 0.7×维度加权分 + 0.3×综合分"""
        r = Layer3Result(
            path="/a.jpg", filename="a.jpg",
            composition_score=8.0, color_score=7.5, lighting_score=7.0,
            vlm_overall_score=6.8,
        )
        dim_score = (8.0 * 0.35 + 7.5 * 0.30 + 7.0 * 0.20) / _DIM_W_SUM
        expected = 0.7 * dim_score + 0.3 * 6.8
        assert _vlm_effective(r, _WEIGHTS, _DIM_W_SUM) == pytest.approx(expected, abs=1e-6)

    def test_vlm_unscored_returns_none(self):
        """VLM 未评分（vlm_overall_score=0）应返回 None"""
        r = Layer3Result(path="/a.jpg", filename="a.jpg", vlm_overall_score=0.0)
        assert _vlm_effective(r, _WEIGHTS, _DIM_W_SUM) is None

    def test_uses_compute_vlm_dim_score(self, monkeypatch):
        """回归：维度分必须走 _compute_vlm_dim_score（防引用未定义符号导致 NameError）"""
        from core.pipeline import _compute_vlm_dim_score

        calls = []
        original = _compute_vlm_dim_score

        def spy(r, weights, dim_w_sum):
            calls.append(1)
            return original(r, weights, dim_w_sum)

        monkeypatch.setattr("core.pipeline._compute_vlm_dim_score", spy)
        r = Layer3Result(
            path="/a.jpg", filename="a.jpg",
            composition_score=8.0, color_score=7.5, lighting_score=7.0,
            vlm_overall_score=6.8,
        )
        result = _vlm_effective(r, _WEIGHTS, _DIM_W_SUM)
        assert calls  # _compute_vlm_dim_score 被调用 → 未引用未定义的 _vlm_dim_score
        assert result is not None

    def test_dim_w_sum_zero_falls_back_to_overall(self):
        """维度权重和为零（异常配置）时应回退到 TOPIQ 分，不崩溃"""
        r = Layer3Result(
            path="/a.jpg", filename="a.jpg",
            composition_score=8.0, color_score=7.5, lighting_score=7.0,
            vlm_overall_score=6.8, overall_score=7.2,
        )
        # dim_w_sum=0 → _compute_vlm_dim_score 回退 r.overall_score
        q = _vlm_effective(r, _WEIGHTS, 0.0)
        assert q == pytest.approx(0.7 * 7.2 + 0.3 * 6.8, abs=1e-6)


class TestVlmTopiqDivergence:
    """VLM 有效分与 TOPIQ 分之差（离群诊断，只诊断不改分）"""

    def test_vlm_scored_returns_abs_diff(self):
        """VLM 已评分应返回 |VLM有效分 - TOPIQ分| 并保留 4 位小数"""
        r = Layer3Result(
            path="/a.jpg", filename="a.jpg",
            overall_score=7.0,
            composition_score=8.0, color_score=7.5, lighting_score=7.0,
            vlm_overall_score=6.8,
        )
        q = _vlm_effective(r, _WEIGHTS, _DIM_W_SUM)
        expected = round(abs(q - 7.0), 4)
        assert _vlm_topiq_divergence(r, _WEIGHTS, _DIM_W_SUM) == pytest.approx(expected, abs=1e-6)

    def test_vlm_unscored_returns_none(self):
        """VLM 未评分应返回 None（无诊断数据）"""
        r = Layer3Result(path="/a.jpg", filename="a.jpg", vlm_overall_score=0.0)
        assert _vlm_topiq_divergence(r, _WEIGHTS, _DIM_W_SUM) is None

    def test_zero_when_agreement(self):
        """VLM 有效分与 TOPIQ 分相同时返回 0.0（无分歧）"""
        r = Layer3Result(
            path="/a.jpg", filename="a.jpg",
            overall_score=7.0,
            composition_score=7.0, color_score=7.0, lighting_score=7.0,
            vlm_overall_score=7.0,
        )
        assert _vlm_topiq_divergence(r, _WEIGHTS, _DIM_W_SUM) == 0.0
