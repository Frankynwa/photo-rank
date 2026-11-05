"""选片配额回归测试：场景桶硬配额、类别保底、质量地板、前段保底

固化 2026-08 P0 多样性优化的两个历史 bug：
1. 类别保底置换时借用 SCENE_BUCKET_MAX_RELAXED，导致审计报告 relax_level=0
   但实际单桶 3 张——硬配额被保底悄悄绕过；
2. 前段保底只尝试池中第一个候选，且把"榜内移动"误判为新增占位，导致
   Top100 有人像、Top12（用户实际发帖量）永远 0 张人像。
"""
from collections import Counter

from config import (
    CATEGORY_MIN_RATIO,
    CATEGORY_QUOTA_MIN_SCORE_RATIO,
    SCENE_BUCKET_MAX,
)
from core.models import Layer3Result
from core.pipeline import (
    _apply_category_quota,
    _ensure_top_segment,
    _scene_bucket,
    _select_with_quota,
)


def _name(day: int, hhmm: str, tag: str = "") -> str:
    """生成可被 _scene_bucket 解析的文件名：IMG_202607{day}_{hhmm}{tag}.jpg"""
    return f"IMG_202607{day:02d}_{hhmm}{tag}.jpg"


def _distinct(i: int, base_hh: int = 8) -> str:
    """第 i 个互不相同的场景桶（28 天循环，超过则换小时）"""
    return _name(i % 28 + 1, f"{base_hh + i // 28:02d}00")


def _photo(filename: str, score: float, category: str) -> Layer3Result:
    return Layer3Result(path=filename, filename=filename, category=category)


def _rank(rows):
    score_map = {r.path: s for r, s in rows}
    ranked = sorted((r for r, _ in rows), key=lambda r: score_map[r.path], reverse=True)
    return ranked, score_map


class TestSceneBucketCap:
    """场景桶硬配额（同一次旅行里 30 分钟内的照片不应霸榜）"""

    def test_spread_candidates_keep_strict_cap(self):
        """候选分散在足够多的桶时，严格配额即可凑满，不放宽"""
        rows = [(_photo(_distinct(i), 9.0 - i * 0.01, "风景"), 9.0 - i * 0.01)
                for i in range(20)]
        ranked, score_map = _rank(rows)
        chosen, audit = _select_with_quota(ranked, score_map, 10)

        assert len(chosen) == 10
        assert audit["relax_level"] == 0
        assert audit["scene_bucket_peak"] <= SCENE_BUCKET_MAX

    def test_same_bucket_relaxes_only_when_short(self):
        """同桶候选过多时必须逐级放宽才能凑满 k，审计如实记录放宽级别"""
        rows = [(_photo(_name(1, f"09{i:02d}"), 9.0 - i * 0.01, "风景"), 9.0 - i * 0.01)
                for i in range(6)]
        ranked, score_map = _rank(rows)
        chosen, audit = _select_with_quota(ranked, score_map, 5)

        assert len(chosen) == 5
        assert audit["relax_level"] > 0, "同一桶候选超标时必须放宽才能凑满"
        assert audit["scene_bucket_peak"] == 5


class TestCategoryQuota:
    """类别保底（人像/人文纪实不得被风景挤出榜单）"""

    def test_fills_portrait_gap(self):
        """人像缺口从榜尾最低分的非保底类置换，总量不变"""
        rows = [(_photo(_distinct(i), 9.0 - i * 0.01, "风景"), 9.0 - i * 0.01)
                for i in range(40)]
        rows += [(_photo(_distinct(i, base_hh=15), 8.5 - i * 0.01, "人像"),
                  8.5 - i * 0.01) for i in range(10)]
        ranked, score_map = _rank(rows)
        chosen, audit = _select_with_quota(ranked, score_map, 20)

        need = int(round(20 * CATEGORY_MIN_RATIO["人像"]))
        assert need >= 1
        assert audit["category_counts"].get("人像", 0) == need
        assert len(chosen) == 20, "保底置换不得改变榜单总量"

    def test_rejects_below_floor(self):
        """候选分低于入选最低分 × 系数时不补：宁可缺，不塞烂片"""
        rows = [(_photo(_distinct(i), 9.0 - i * 0.01, "风景"), 9.0 - i * 0.01)
                for i in range(40)]
        rows += [(_photo(_distinct(i, base_hh=15), 3.0, "人像"), 3.0) for i in range(10)]
        ranked, score_map = _rank(rows)
        chosen, audit = _select_with_quota(ranked, score_map, 20)

        assert audit["category_counts"].get("人像", 0) == 0, "低分人像不应被塞进榜"
        assert len(chosen) == 20

    def test_quota_replacement_respects_scene_cap(self):
        """回归：保底置换不得借用放宽上限——已满的桶必须跳过，改试下一个候选"""
        # 桶 Q（20260701, 30）内已有 2 张风景，达到硬上限
        base = [
            (_photo(_name(1, "1500"), 8.999, "风景"), 8.999),
            (_photo(_name(1, "1510"), 8.998, "风景"), 8.998),
        ]
        base += [(_photo(_distinct(i + 1, base_hh=10), 8.99 - i * 0.001, "风景"),
                  8.99 - i * 0.001) for i in range(19)]
        # 已入选的人像 A（桶 P）
        base.append((_photo(_name(2, "0900"), 8.90, "人像"), 8.90))
        ranked, score_map = _rank(base)
        k = 22
        need = int(round(k * CATEGORY_MIN_RATIO["人像"]))
        assert need == 2, "本用例依赖人像配额为 2"

        # 榜外人像（均低于 A，保证不在前 k 内）：B 在已满的桶 Q，C 在空桶 R
        extra_b = (_photo(_name(1, "1520"), 8.85, "人像"), 8.85)
        extra_c = (_photo(_name(3, "0900"), 8.84, "人像"), 8.84)
        for r, s in (extra_b, extra_c):
            score_map[r.path] = s
        ranked = sorted(list(ranked) + [extra_b[0], extra_c[0]],
                        key=lambda r: score_map[r.path], reverse=True)

        chosen = _apply_category_quota(ranked[:k], ranked, score_map, k)

        assert sum(1 for r in chosen if r.category == "人像") == need
        buckets = Counter(b for b in (_scene_bucket(r.path) for r in chosen) if b is not None)
        assert buckets[_scene_bucket(_name(1, "1500"))] <= SCENE_BUCKET_MAX, \
            "保底不得把已满的场景桶堆到上限之上"
        assert extra_b[0].path not in {r.path for r in chosen}, "满桶候选应被跳过"
        assert extra_c[0].path in {r.path for r in chosen}, "有空位的桶应被采用"


class TestTopSegment:
    """前段保底（用户实际只发 12 张，全文配额对第一屏无效）"""

    def test_portrait_enters_top_segment(self):
        """前十全是风景时，人像必须被置入前段"""
        rows = [(_photo(_distinct(i), 9.5 - i * 0.005, "风景"), 9.5 - i * 0.005)
                for i in range(30)]
        rows += [(_photo(_distinct(i, base_hh=17), 9.2 - i * 0.01, "人像"),
                  9.2 - i * 0.01) for i in range(10)]
        ranked, score_map = _rank(rows)
        chosen, audit = _select_with_quota(ranked, score_map, 40, top_segment=12)

        head_cats = Counter(r.category for r in chosen[:12])
        assert head_cats.get("人像", 0) >= 1, "前 12 张必须出现人像"
        assert audit["top_segment_changes"] >= 1
        assert len(chosen) == 40, "前段保底不得改变榜单总量"

    def test_in_list_move_does_not_need_bucket_slot(self):
        """回归：已入选的候选移入前段不占新桶名额，不能被场景配额误拒"""
        # 前段 12 张风景，其中 2 张占住桶 X
        head = [
            _photo(_name(1, "1100"), 9.90, "风景"),
            _photo(_name(1, "1110"), 9.89, "风景"),
        ]
        head += [_photo(_distinct(i, base_hh=8), 9.88 - i * 0.001, "风景")
                 for i in range(10)]
        # 榜内：人像 B 在空桶 Y
        tail_portrait = _photo(_name(2, "0900"), 9.40, "人像")
        # 榜外：人像 A 分更高，但所在桶 X 已满
        off_portrait = _photo(_name(1, "1120"), 9.50, "人像")
        chosen = head + [tail_portrait]
        score_map = {r.path: s for r, s in zip(chosen, [9.90, 9.89] + [9.87 - i * 0.001 for i in range(10)] + [9.40])}
        score_map[off_portrait.path] = 9.50
        ranked = sorted(chosen + [off_portrait], key=lambda r: score_map[r.path], reverse=True)

        new_chosen, changes = _ensure_top_segment(chosen, ranked, score_map, 12)

        head_paths = {r.path for r in new_chosen[:12]}
        assert changes == 1
        assert tail_portrait.path in head_paths, "榜内人像应被移到前段"
        assert off_portrait.path not in head_paths, "满桶候选即使分更高也不得破配额"
        buckets = Counter(b for b in (_scene_bucket(r.path) for r in new_chosen) if b is not None)
        assert buckets[_scene_bucket(_name(1, "1100"))] <= SCENE_BUCKET_MAX

    def test_floor_blocks_low_score_portrait(self):
        """人像分低于前段门槛 × 系数时不做置换：前段质量优先于形式多样性"""
        rows = [(_photo(_distinct(i), 9.5 - i * 0.005, "风景"), 9.5 - i * 0.005)
                for i in range(30)]
        rows += [(_photo(_distinct(i, base_hh=17), 5.0, "人像"), 5.0) for i in range(10)]
        ranked, score_map = _rank(rows)

        head = ranked[:12]
        min_head = min(score_map[r.path] for r in head)
        assert 5.0 < min_head * CATEGORY_QUOTA_MIN_SCORE_RATIO, "用例前提：人像低于质量地板"

        chosen, audit = _select_with_quota(ranked, score_map, 40, top_segment=12)
        assert Counter(r.category for r in chosen[:12]).get("人像", 0) == 0
        assert audit["top_segment_changes"] == 0
