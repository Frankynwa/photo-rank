"""三桶分级（精选/可用/淘汰）单元测试 — 门槛取严、入选提级、百分位计算

设计依据（docs/research/p0-3）：
- 门槛 = max(批次分位数, 固定底线)，两道闸都要生效；
- 分位数吸收批次难度，固定底线保住绝对语义；
- 已入选照片永不标淘汰（类别保底会特意选低分照片冲配额）。
"""
import pytest

from core.pipeline import _assign_buckets, _percentile_of


class TestPercentileOf:
    def test_empty_returns_zero(self):
        assert _percentile_of([], 0.5) == 0.0

    def test_single_value(self):
        assert _percentile_of([7.3], 0.85) == 7.3

    def test_linear_interpolation(self):
        """10 个等距值：p85 = 8 + 0.65，p20 = 2 + 0.8（线性插值）"""
        vals = [float(i) for i in range(1, 11)]
        assert _percentile_of(vals, 0.85) == pytest.approx(8.65)
        assert _percentile_of(vals, 0.20) == pytest.approx(2.8)

    def test_extremes(self):
        vals = [1.0, 5.0, 9.0]
        assert _percentile_of(vals, 0.0) == 1.0
        assert _percentile_of(vals, 1.0) == 9.0

    def test_unsorted_input(self):
        """入参顺序不影响结果"""
        assert _percentile_of([5.0, 1.0, 9.0], 0.5) == _percentile_of([1.0, 5.0, 9.0], 0.5)


class TestAssignBuckets:
    def _scores(self, vals):
        return {f"/p{i}.jpg": v for i, v in enumerate(vals)}

    def test_three_buckets_split(self):
        """10 张 1-10：精选 ≥8.65 两张，淘汰 <4.0 三张，其余可用"""
        info = _assign_buckets(self._scores([float(i) for i in range(1, 11)]), set())
        assert info["thresholds"]["select"] == pytest.approx(8.65)
        assert info["thresholds"]["reject"] == pytest.approx(4.0)
        assert info["counts"]["select"] == 2   # 9, 10
        assert info["counts"]["reject"] == 3   # 1, 2, 3
        assert info["counts"]["maybe"] == 5    # 4~8

    def test_select_threshold_uses_percentile_when_above_floor(self):
        """P85 > 底线 7.0 时，实际门槛取 P85"""
        info = _assign_buckets(self._scores([float(i) for i in range(1, 11)]), set())
        assert info["thresholds"]["select_p85"] == pytest.approx(8.65)
        assert info["thresholds"]["select"] == pytest.approx(8.65)

    def test_reject_threshold_uses_floor_when_above_percentile(self):
        """P20 (2.8) < 底线 4.0 时，实际门槛取底线 —— 固定底线守住绝对语义"""
        info = _assign_buckets(self._scores([float(i) for i in range(1, 11)]), set())
        assert info["thresholds"]["reject_p20"] == pytest.approx(2.8)
        assert info["thresholds"]["reject"] == pytest.approx(4.0)

    def test_all_bad_batch_nothing_selected(self):
        """整批都差（最高 3 分）：P85 只有 2.7，但底线 7.0 生效 → 零精选"""
        info = _assign_buckets(self._scores([1.0, 1.5, 2.0, 2.5, 3.0]), set())
        assert info["counts"]["select"] == 0
        assert info["thresholds"]["select"] == pytest.approx(7.0)

    def test_all_good_batch_still_rejects_tail(self):
        """整批都好（最低 8 分）：P20 升到 8.4，淘汰门槛跟着升 → 仍有 1 张进淘汰

        分位数的意义：整批全好时不至于零淘汰、失去区分度。
        """
        info = _assign_buckets(self._scores([8.0, 8.5, 9.0, 9.5, 10.0]), set())
        assert info["thresholds"]["reject"] == pytest.approx(8.4)
        assert info["counts"]["reject"] == 1

    def test_selected_photo_never_rejected(self):
        """已入选的照片即使低于淘汰门槛也只降级为可用（带 promoted 标记）"""
        scores = self._scores([1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5, 10.0])
        selected = {"/p0.jpg", "/p1.jpg"}  # 最低分的两张被类别保底选中
        info = _assign_buckets(scores, selected)
        assert info["map"]["/p0.jpg"]["bucket"] == "maybe"
        assert info["map"]["/p0.jpg"]["promoted"] is True
        assert info["map"]["/p1.jpg"]["bucket"] == "maybe"
        # 未被选中的低分照片正常淘汰
        assert info["map"]["/p2.jpg"]["bucket"] == "reject"
        assert info["counts"]["reject"] == 1
        assert info["counts"]["promoted"] == 2

    def test_percentile_midpoint_for_ties(self):
        """并列分数取中位位置，避免名次抖动"""
        s = self._scores([5.0] * 10)
        info = _assign_buckets(s, set())
        # 全并列：(0+10)/2/10 = 50%
        assert info["map"]["/p0.jpg"]["percentile"] == pytest.approx(50.0)

    def test_percentile_ordering(self):
        """百分位随分数单调不减"""
        info = _assign_buckets(self._scores([float(i) for i in range(1, 11)]), set())
        pcts = [info["map"][f"/p{i}.jpg"]["percentile"] for i in range(10)]
        assert pcts == sorted(pcts)
        assert pcts[0] < 10.0 and pcts[-1] > 90.0

    def test_empty_scores(self):
        info = _assign_buckets({}, set())
        assert info == {"map": {}, "thresholds": {}, "counts": {}}

    def test_single_photo(self):
        """单张：P85=P20=该分，精选门槛被底线抬到 7.0，淘汰门槛被该分本身抬到 5.0

        淘汰用严格小于，因此 5.0 分的照片不会把自己淘汰（归入可用）。
        """
        info = _assign_buckets({"/a.jpg": 5.0}, set())
        assert info["map"]["/a.jpg"]["bucket"] == "maybe"
        assert info["thresholds"]["select"] == pytest.approx(7.0)
        assert info["thresholds"]["reject"] == pytest.approx(5.0)

    def test_single_high_score_goes_select(self):
        """单张高分：P85 = 该分，与分同值也算入选（阈值用 >=）"""
        info = _assign_buckets({"/a.jpg": 8.0}, set())
        assert info["map"]["/a.jpg"]["bucket"] == "select"
        assert info["thresholds"]["select"] == pytest.approx(8.0)

    def test_reject_never_exceeds_select(self):
        """异常配置保护：淘汰门槛不得反超精选门槛"""
        info = _assign_buckets(self._scores([float(i) for i in range(1, 11)]), set())
        assert info["thresholds"]["reject"] < info["thresholds"]["select"]

    def test_thresholds_expose_raw_and_floor(self):
        """审计口径：原始分位数与固定底线都要落盘，便于回溯“哪道闸生效”"""
        info = _assign_buckets(self._scores([float(i) for i in range(1, 11)]), set())
        t = info["thresholds"]
        assert set(t) == {
            "select", "reject", "select_p85", "reject_p20",
            "select_floor", "reject_floor",
        }
        assert t["select_floor"] == 7.0
        assert t["reject_floor"] == 4.0
