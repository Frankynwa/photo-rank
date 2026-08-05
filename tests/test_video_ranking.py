"""视频精选榜测试 — 时长比例名额分配 + run_video_ranking 流程"""
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core.models import Layer3Result
from core.pipeline import _allocate_quotas, run_video_ranking


# ---------------------------------------------------------------------------
# 名额分配（时长比例）测试
# ---------------------------------------------------------------------------

class TestAllocateQuotas:
    def test_equal_lengths(self):
        """等长视频应均分名额"""
        q = _allocate_quotas({"a.mp4": 10.0, "b.mp4": 10.0}, 20)
        assert q == {"a.mp4": 10, "b.mp4": 10}

    def test_proportional_lengths(self):
        """3:7 时长 → 名额 6:14（总额度 20）"""
        q = _allocate_quotas({"short.mp4": 3.0, "long.mp4": 7.0}, 20)
        assert q["short.mp4"] == 6
        assert q["long.mp4"] == 14
        assert sum(q.values()) == 20

    def test_extreme_ratio_keeps_min_one(self):
        """极端时长比（0.1s vs 99.9s）仍满足短视频至少 1、总和用满"""
        q = _allocate_quotas({"tiny.mp4": 0.1, "huge.mp4": 99.9}, 20)
        assert q["tiny.mp4"] >= 1
        assert q["huge.mp4"] > q["tiny.mp4"]
        assert sum(q.values()) == 20

    def test_missing_duration_degrades_to_equal(self):
        """时长缺失（全 0）时退化为均分"""
        q = _allocate_quotas({"a.mp4": 0.0, "b.mp4": 0.0, "c.mp4": 0.0}, 20)
        assert sum(q.values()) == 20
        assert min(q.values()) >= 6  # 20/3 = 6 余 2

    def test_every_video_at_least_one(self):
        """短视频也至少 1 个名额"""
        q = _allocate_quotas({"big.mp4": 100.0, "tiny.mp4": 0.01}, 20)
        assert q["tiny.mp4"] >= 1
        assert sum(q.values()) == 20


# ---------------------------------------------------------------------------
# run_video_ranking 流程测试（mock 深度评分层）
# ---------------------------------------------------------------------------

def _make_frame(path: Path, seed: int):
    """生成高纹理帧并落盘"""
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))


def _no_faces(paths):
    return []


def _fake_topiq(paths, device="auto"):
    return [
        Layer3Result(
            path=str(p), filename=Path(p).name,
            overall_score=round((seed % 5) / 2.0, 2),  # 区分分数便于排序断言
        )
        for seed, p in enumerate(paths)
    ]


def _fake_vlm(results, **kwargs):
    """透传 TOPIQ 结果（VLM 未评分路径，融合分回退 TOPIQ）"""
    return list(results)


class TestRunVideoRanking:
    def test_basic_ranking_with_manifest(self, tmp_path: Path):
        """构造 2 视频帧 + manifest，验证视频榜输出与出处/时间标注"""
        frames_dir = tmp_path / "video_frames"
        frames_dir.mkdir(parents=True)

        # 两个视频各 2 帧，时长 5s / 15s
        entries = []
        for vname, dur, tsecs in [("a.mp4", 5.0, [1, 3]), ("b.mp4", 15.0, [2, 8])]:
            for t in tsecs:
                p = frames_dir / f"{vname.split('.')[0]}_{'sig'}_t{t:03d}.jpg"
                _make_frame(p, seed=t)
                entries.append({
                    "filename": p.name,
                    "source_video": vname,
                    "timestamp": float(t),
                    "sharpness": 50.0,
                    "overall": 0.7,
                    "phash": "",
                    "duration": dur,
                })
        (frames_dir / "manifest.json").write_text(
            json.dumps(entries, ensure_ascii=False), encoding="utf-8"
        )

        with patch("core.layer3_aesthetic.apply_vlm_dimensions", side_effect=_fake_vlm), \
                patch("core.layer4_face.analyze_faces", side_effect=_no_faces), \
                patch("core.layer4_face.to_face_prompt_summary", return_value=""):
            summary = run_video_ranking(
                frames_dir, platform="test_platform", max_total=20,
            )

        assert summary["total"] == 4
        assert summary["output_count"] == 4  # 4 帧 < 20 额度，全进
        # 出处/时间标注
        for r in summary["results"]:
            assert r["source_video"] in ("a.mp4", "b.mp4")
            assert r["timestamp"] > 0
        # 4 帧 < 20 额度，两视频帧全部入选（时长比例在配额单测中覆盖）
        by_src = {}
        for r in summary["results"]:
            by_src.setdefault(r["source_video"], 0)
            by_src[r["source_video"]] += 1
        assert by_src == {"a.mp4": 2, "b.mp4": 2}

        # 输出文件存在（展示文件名 v01_xxx.jpg）
        out = Path(summary["output_dir"])
        ranking_file = out / "video_ranking.json"
        assert ranking_file.exists()
        saved = json.loads(ranking_file.read_text(encoding="utf-8"))
        assert len(saved) == 4
        assert all(s["filename"].startswith("v") for s in saved), "展示文件名应为 vNN_ 序号名"
        for s in saved:
            assert (out / s["filename"]).exists(), "入选帧应复制到 output 目录"

    def test_no_frames_returns_empty(self, tmp_path: Path):
        """无视频帧目录时应安全返回空结果"""
        empty = tmp_path / "empty"
        empty.mkdir(parents=True)
        summary = run_video_ranking(empty, platform="test_platform")
        assert summary["total"] == 0
        assert summary["results"] == []
