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

    def test_all_zero_dim_weights_falls_back_to_topiq(self):
        """维度权重全为 0（异常配置）时分母为 0，应回退 r.overall_score 不崩溃

        注意：dim_w_sum 入参已不再生效（改为内部按题材重算），必须用全 0 权重才触发回退。
        """
        r = Layer3Result(
            path="/a.jpg", filename="a.jpg",
            composition_score=8.0, color_score=7.5, lighting_score=7.0,
            vlm_overall_score=6.8, overall_score=7.2,
        )
        zero_w = {"composition": 0.0, "color": 0.0, "lighting": 0.0, "face": 1.0}
        q = _vlm_effective(r, zero_w, 0.0)
        assert q == pytest.approx(0.7 * 7.2 + 0.3 * 6.8, abs=1e-6)

    def test_dim_w_sum_arg_ignored(self):
        """回归：dim_w_sum 入参仅为签名兼容，传垃圾值不应影响结果（内部按题材重算）"""
        r = Layer3Result(
            path="/a.jpg", filename="a.jpg",
            composition_score=8.0, color_score=7.5, lighting_score=7.0,
            vlm_overall_score=6.8,
        )
        normal = _vlm_effective(r, _WEIGHTS, _DIM_W_SUM)
        garbage = _vlm_effective(r, _WEIGHTS, 999.0)
        assert normal == pytest.approx(garbage, abs=1e-9)


class TestCategoryWeightOverride:
    """题材化维度权重：命中 CATEGORY_WEIGHT_OVERRIDES 的题材覆盖平台权重

    依据：用同一把尺子量不同题材——夜景的光影、美食的色彩、人像的人脸
    才是决定性维度，平台统一权重会让"夜色氛围好但构图普通"输给"白天构图工整但无光影"。
    """

    def test_override_applied_when_category_hit(self):
        """命中题材时权重被覆盖（四项齐全）"""
        from core.pipeline import _weights_for

        r = Layer3Result(path="/a.jpg", filename="a.jpg", category="夜景")
        w = _weights_for(r, _WEIGHTS)
        assert w["lighting"] == 0.40  # 夜景光影权重最高
        assert w["composition"] == 0.20
        assert w["color"] == 0.30
        assert sum(w[k] for k in ("composition", "color", "lighting", "face")) == pytest.approx(1.0)

    def test_base_weights_kept_when_category_miss(self):
        """未命中题材（空/"其他"/未列出的题材）应原样返回平台权重"""
        from core.pipeline import _weights_for

        for cat in ("", "其他", "未分类", "航拍"):
            r = Layer3Result(path="/a.jpg", filename="a.jpg", category=cat)
            assert _weights_for(r, _WEIGHTS) == _WEIGHTS

    def test_base_dict_not_mutated(self):
        """覆盖不应就地修改传入的 base 权重（否则会影响后续照片）"""
        from core.pipeline import _weights_for

        base = dict(_WEIGHTS)
        r = Layer3Result(path="/a.jpg", filename="a.jpg", category="人像")
        _weights_for(r, base)
        assert base == _WEIGHTS

    def test_override_changes_ranking_of_same_dimensions(self):
        """同一组维度分，不同题材应得出不同有效分（权重重构排序）"""
        dims = dict(composition_score=9.0, color_score=6.0, lighting_score=9.0,
                    vlm_overall_score=7.0)
        portrait = Layer3Result(path="/p.jpg", filename="p.jpg", category="人像", **dims)
        building = Layer3Result(path="/b.jpg", filename="b.jpg", category="建筑", **dims)
        # 人像：色彩权重 0.25 高于建筑 0.20，而 color=6 是低分 → 人像被拉低
        # 建筑：构图 0.45 最高，构图=9 的高分被放大 → 有效分更高
        assert _vlm_effective(building, _WEIGHTS) > _vlm_effective(portrait, _WEIGHTS)


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
