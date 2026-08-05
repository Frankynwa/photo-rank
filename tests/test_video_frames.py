"""视频帧提取与本地快筛测试 — OpenCV 合成视频，不依赖真实视频文件"""
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core.layer2_similarity import compute_phash
from core.video_frames import (
    VideoExtractError,
    _apply_relative_blur_filter,
    _filter_batch,
    extract_quality_frames,
)

# ---------------------------------------------------------------------------
# 辅助工具：合成视频
# ---------------------------------------------------------------------------

def _make_video(path: Path, frames: list[np.ndarray], fps: float = 10.0):
    """把帧列表写入 mp4 视频"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    assert path.exists() and path.stat().st_size > 0, "视频生成失败"


def _clear_frame(size=(128, 128), seed: int = 1) -> np.ndarray:
    """高纹理清晰帧（Laplacian 方差大，不会被 L1 判模糊）"""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    return arr


def _blank_frame(color=(128, 128, 128), size=(128, 128)) -> np.ndarray:
    """纯色帧（Laplacian 方差为 0，必判模糊）"""
    return np.full((*size, 3), color, dtype=np.uint8)


# 人脸分析 mock：测试环境不下载 MediaPipe 模型
def _no_faces(paths):
    return []


# ---------------------------------------------------------------------------
# 正常抽帧筛选测试
# ---------------------------------------------------------------------------

class TestExtractQualityFrames:
    def test_clear_frames_kept(self, tmp_path: Path):
        """清晰纹理帧应被保留（L1 通过，仅相对淘汰最低 30%）"""
        video = tmp_path / "clear.mp4"
        frames = [_clear_frame(seed=i) for i in range(10)]
        _make_video(video, frames)

        out = tmp_path / "frames"
        with patch("core.video_frames.analyze_faces", side_effect=_no_faces):
            stats = extract_quality_frames(video, out, interval=0.1)

        assert stats["candidates"] == 10
        assert stats["kept"] >= 1, "清晰帧应至少保留 1 帧"
        assert stats["blurry"] <= 4, "相对淘汰只淘汰最低 30% 左右"
        assert len(stats["saved"]) == stats["kept"]
        # 落盘文件存在，命名含时间戳（t{秒}）
        for name in stats["saved"]:
            assert (out / name).exists()
            assert "_t" in name, f"帧命名应含时间戳: {name}"
        # manifest.json 存在且字段完整
        mf = out / "manifest.json"
        assert mf.exists(), "应输出 manifest.json"
        import json
        manifest = json.loads(mf.read_text(encoding="utf-8"))
        assert len(manifest) == stats["kept"]
        for entry in manifest:
            assert entry["filename"] in stats["saved"]
            assert entry["source_video"] == "clear.mp4"
            assert entry["timestamp"] >= 0
            assert "sharpness" in entry and "overall" in entry
            assert "phash" in entry and "duration" in entry

    def test_blurry_frames_rejected(self, tmp_path: Path):
        """纯色（模糊）帧应全部被 L1 淘汰"""
        video = tmp_path / "blurry.mp4"
        frames = [_blank_frame() for _ in range(10)]
        _make_video(video, frames)

        out = tmp_path / "frames"
        with patch("core.video_frames.analyze_faces", side_effect=_no_faces):
            stats = extract_quality_frames(video, out, interval=0.1)

        assert stats["candidates"] == 10
        assert stats["blurry"] >= 10
        assert stats["kept"] == 0, "纯色模糊帧不应保留"

    def test_identical_frames_dedup(self, tmp_path: Path):
        """完全相同的帧应被去重，只保留第一帧"""
        # 直接测试 _filter_batch 的去重逻辑（避免 mp4 有损压缩导致解码帧不完全一致）
        frame = _clear_frame(seed=7)
        p1 = tmp_path / "a.jpg"
        p2 = tmp_path / "b.jpg"
        cv2.imwrite(str(p1), frame)
        cv2.imwrite(str(p2), frame)  # 内容完全一致

        stats = {"blurry": 0, "dup": 0, "blink": 0, "failed": 0}
        kept = [(p1, compute_phash(p1), 100.0, 0.8)]
        with patch("core.video_frames.analyze_faces", side_effect=_no_faces):
            _filter_batch([p2], kept, stats)

        assert stats["dup"] == 1, "相同帧应被去重"
        assert len(kept) == 1, "去重后 kept 不应增加"

    def test_mixed_blurry_and_clear(self, tmp_path: Path):
        """模糊帧淘汰、清晰帧保留，互不干扰"""
        video = tmp_path / "mixed.mp4"
        frames = []
        for i in range(5):
            frames.append(_clear_frame(seed=i))
            frames.append(_blank_frame())
        _make_video(video, frames)

        out = tmp_path / "frames"
        with patch("core.video_frames.analyze_faces", side_effect=_no_faces):
            stats = extract_quality_frames(video, out, interval=0.1)

        assert stats["candidates"] == 10
        assert stats["blurry"] >= 5
        assert stats["kept"] >= 1


# ---------------------------------------------------------------------------
# 相对模糊淘汰测试
# ---------------------------------------------------------------------------

class TestRelativeBlurFilter:
    def _mk_kept(self, sharpness_list):
        return [
            (Path(f"/tmp/{i}.jpg"), object(), float(s), 0.8)
            for i, s in enumerate(sharpness_list)
        ]

    def test_bottom_30_percent_removed(self):
        """Laplacian 最低的 30% 帧应被淘汰"""
        kept = self._mk_kept([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        stats = {"blurry": 0}
        _apply_relative_blur_filter(kept, stats)
        # p30 = 37 → 淘汰 10/20/30，保留 7 帧
        assert len(kept) == 7
        assert stats["blurry"] == 3

    def test_few_frames_no_relative_removal(self):
        """少于 4 帧不做相对淘汰（避免噪声放大）"""
        kept = self._mk_kept([5, 6, 7])
        stats = {"blurry": 0}
        _apply_relative_blur_filter(kept, stats)
        assert len(kept) == 3
        assert stats["blurry"] == 0

    def test_all_above_floor_kept_when_similar(self):
        """分数接近时淘汰数量应接近 30% 比例"""
        kept = self._mk_kept([50.0] * 6 + [49.0, 49.5, 49.8, 49.9])
        stats = {"blurry": 0}
        _apply_relative_blur_filter(kept, stats)
        # p30 落在 49.x 附近，只淘汰极少数
        assert 0 <= stats["blurry"] <= 4


# ---------------------------------------------------------------------------
# 错误处理测试
# ---------------------------------------------------------------------------

class TestVideoErrors:
    def test_missing_video_raises(self, tmp_path: Path):
        """不存在的视频应抛 VideoExtractError"""
        with pytest.raises(VideoExtractError, match="不存在"):
            extract_quality_frames(tmp_path / "missing.mp4", tmp_path)

    def test_corrupted_video_raises(self, tmp_path: Path):
        """损坏文件（非视频）应抛 VideoExtractError"""
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"this is not a video file")
        with pytest.raises(VideoExtractError):
            extract_quality_frames(bad, tmp_path)

    def test_empty_frames_no_crash(self, tmp_path: Path):
        """空输出目录不应报错"""
        video = tmp_path / "clear.mp4"
        frames = [_clear_frame(seed=i) for i in range(10)]
        _make_video(video, frames)

        with patch("core.video_frames.analyze_faces", side_effect=_no_faces):
            stats = extract_quality_frames(video, tmp_path / "sub", interval=0.1)

        assert isinstance(stats, dict)
        assert "kept" in stats
