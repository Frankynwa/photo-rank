"""视频帧提取与本地快筛 — 流式逐帧筛选，不合格帧不落盘

流程：OpenCV 逐帧读取 → 按 interval 采样候选帧（临时目录）→ 批量快筛
（L1 客观淘汰 + L2 pHash 去重 + L4 闭眼淘汰，全部本地零 API）→
合格帧复制到 out_dir（uploads/video_frames），废帧随临时目录删除。

设计原则：
- 候选帧只落临时目录，处理后即删，uploads 只保留合格帧
- 完全复用 L1/L2/L4 的现有函数，不复制筛选逻辑
- 长视频保护：候选帧超过 VIDEO_MAX_FRAMES 时自动放大采样间隔
- 落盘帧命名含时间戳（t{秒}），并输出 manifest.json 记录来源视频/时间戳/客观分
"""
import hashlib
import json
import logging
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from config import (
    HAMMING_THRESHOLD,
    VIDEO_FRAME_BLUR_RATIO,
    VIDEO_FRAME_INTERVAL,
    VIDEO_FRAME_MIN_SHARPNESS,
    VIDEO_MAX_FRAMES,
)
from core.layer1_objective import batch_analyze
from core.layer2_similarity import compute_phash
from core.layer4_face import analyze_faces

logger = logging.getLogger(__name__)

# 每批候选帧数量：攒够一批统一筛选，平衡内存与批处理效率
_BATCH_SIZE = 20

# 采样步长保护：视频异常（fps 极小/极大）时步长至少 1 帧
_MIN_STEP = 1


class VideoExtractError(Exception):
    """视频处理错误（打不开 / 无视频流等）"""


def _empty_stats(total_frames: int = 0) -> dict:
    return {
        "total_frames": total_frames,  # 视频总帧数
        "candidates": 0,               # 采样出的候选帧数
        "blurry": 0,                   # 因模糊/曝光被 L1 淘汰
        "dup": 0,                      # 因与已保留帧相似被去重
        "blink": 0,                    # 因闭眼被淘汰
        "failed": 0,                   # 其他失败（读帧失败等）
        "kept": 0,                     # 最终保留帧数
        "saved": [],                   # 落盘文件名列表
    }


def _filter_batch(candidate_paths: list[Path], kept: list[tuple[Path, object, float, float]], stats: dict) -> None:
    """批量快筛：曝光淘汰 → 绝对模糊底线 → pHash 去重 → 累积（含 sharpness + L1 客观分）

    模糊的"批内相对淘汰"不在此处做，而是在全部候选帧读完后统一计算
    （见 _apply_relative_blur_filter），保证阈值基于完整批次而非单个子批。

    Args:
        candidate_paths: 本批候选帧（临时目录内的 JPEG）
        kept: 已保留帧 [(临时路径, phash, sharpness, l1_overall), ...]，去重时与新帧比较
        stats: 统计字典（就地累加）
    """
    # 1. L1 客观分析（多线程，复用现有实现）
    results = batch_analyze(candidate_paths)
    keep_paths: list[tuple[Path, float, float]] = []
    for p, r in zip(candidate_paths, results):
        # 曝光异常（过曝/欠曝）直接淘汰；模糊不用 L1 的绝对阈值（照片阈值对视频帧过严）
        if "过曝" in r.reject_reason or "欠曝" in r.reject_reason:
            stats["failed"] += 1
            continue
        # 绝对模糊底线：Laplacian 过低视为解码异常/纯色帧
        if r.sharpness < VIDEO_FRAME_MIN_SHARPNESS:
            stats["blurry"] += 1
            continue
        keep_paths.append((p, r.sharpness, r.overall))

    # 2. L2 pHash 去重（与已保留帧比较，连续画面高度相似只留一帧）
    dedup: list[tuple[Path, object, float, float]] = []
    for p, sharpness, overall in keep_paths:
        h = compute_phash(p)
        if h is None:
            stats["failed"] += 1
            continue
        if any(abs(int(h - old_h)) < HAMMING_THRESHOLD for _p, old_h, _s, _o in kept):
            stats["dup"] += 1
            continue
        dedup.append((p, h, sharpness, overall))

    # 3. L4 人脸闭眼淘汰（批量）
    if dedup:
        blink_map: dict[str, bool] = {}
        try:
            faces = analyze_faces([str(p) for p, _, _, _ in dedup])
            blink_map = {
                Path(r.path).name: r.blink_detected
                for r in faces if r.has_face
            }
        except Exception as e:
            logger.debug(f"人脸批分析失败: {e}")
        for p, h, sharpness, overall in dedup:
            if blink_map.get(Path(p).name):
                stats["blink"] += 1
                continue
            kept.append((p, h, sharpness, overall))


def _apply_relative_blur_filter(
    kept: list[tuple[Path, object, float]], stats: dict,
) -> None:
    """批内相对模糊淘汰：按 Laplacian 排序淘汰最低 VIDEO_FRAME_BLUR_RATIO 的帧

    视频帧的 Laplacian 受编码/分辨率影响大（4K HEVC 天然偏低），
    绝对阈值不可靠，改为相对：每批淘汰"相对最不清晰"的帧（运动模糊/失焦）。
    候选帧太少（<4）时不做相对淘汰，避免噪声放大。
    """
    if len(kept) < 4:
        return
    laps = [s for _p, _h, s, _o in kept]
    rel_thr = float(np.percentile(laps, VIDEO_FRAME_BLUR_RATIO * 100))
    thr = max(rel_thr, VIDEO_FRAME_MIN_SHARPNESS)
    before = len(kept)
    kept[:] = [(p, h, s, o) for p, h, s, o in kept if s >= thr]
    stats["blurry"] += before - len(kept)
    if before - len(kept):
        logger.debug(f"相对模糊淘汰: {before - len(kept)} 帧（阈值 Laplacian={thr:.1f}）")


def extract_quality_frames(
    video_path: str | Path,
    out_dir: str | Path,
    interval: float = VIDEO_FRAME_INTERVAL,
    max_frames: int = VIDEO_MAX_FRAMES,
) -> dict:
    """从视频流式抽帧并本地快筛，合格帧落盘 out_dir

    Args:
        video_path: 视频文件路径
        out_dir: 合格帧输出目录（应指向 uploads 顶层，进标准流水线）
        interval: 候选帧采样间隔（秒）
        max_frames: 候选帧数量上限（超出自动放大间隔）

    Returns:
        dict: 统计信息（total_frames/candidates/blurry/dup/blink/failed/kept/saved）

    Raises:
        VideoExtractError: 视频不存在 / 打不开 / 无有效画面流
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    if not video_path.exists():
        raise VideoExtractError(f"视频不存在: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoExtractError(f"无法打开视频（可能损坏或编码不支持）: {video_path.name}")

    stats = _empty_stats()
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            raise VideoExtractError(f"视频无有效画面流: {video_path.name}")

        stats["total_frames"] = frame_count
        duration = frame_count / fps

        # 长视频保护：候选帧超上限时自动放大采样间隔
        if max_frames > 0 and duration / interval > max_frames:
            interval = duration / max_frames

        step = max(_MIN_STEP, int(round(interval * fps)))
        logger.info(
            f"视频: {video_path.name} | {duration:.1f}s | {frame_count}帧 | "
            f"采样步长 {step}帧（约 {interval:.1f}s/帧）"
        )

        kept: list[tuple[Path, object, float, float]] = []

        with tempfile.TemporaryDirectory(prefix="pr_frames_") as tmp:
            tmp_dir = Path(tmp)
            batch: list[Path] = []
            frame_idx = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % step != 0:
                    frame_idx += 1
                    continue

                fpath = tmp_dir / f"cand_{frame_idx:06d}.jpg"
                if not cv2.imwrite(str(fpath), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                    stats["failed"] += 1
                else:
                    batch.append(fpath)
                    stats["candidates"] += 1

                    if len(batch) >= _BATCH_SIZE:
                        _filter_batch(batch, kept, stats)
                        batch = []
                frame_idx += 1

            if batch:
                _filter_batch(batch, kept, stats)

            # 相对模糊淘汰：基于全部候选帧统一计算阈值（运动模糊/失焦帧）
            _apply_relative_blur_filter(kept, stats)

            # 空视频：元数据正常但实际读不出帧（损坏/无内容）——明确报错而非假成功
            if stats["candidates"] == 0:
                raise VideoExtractError(
                    f"未能从视频中读取任何帧，文件可能损坏: {video_path.name}"
                )

            # 合格帧复制到 out_dir（文件名带视频名+签名+时间戳，可辨识来源且不冲突）
            # 签名用 size+mtime 哈希：同一文件名的视频内容更新后，帧名变化，避免旧帧被静默跳过
            sig = hashlib.md5(
                f"{video_path.stat().st_size}:{video_path.stat().st_mtime_ns}".encode()
            ).hexdigest()[:6]
            out_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            manifest_entries = []
            for i, (fpath, h, sharpness, overall) in enumerate(kept, 1):
                # 从临时文件名 cand_{frame_idx:06d}.jpg 解析帧号，换算视频内时间戳
                try:
                    frame_idx = int(fpath.stem.split("_")[1])
                except (IndexError, ValueError):
                    frame_idx = (i - 1) * step
                ts = round(frame_idx / fps, 2)
                dst = out_dir / f"{video_path.stem}_{sig}_t{int(round(ts)):03d}.jpg"
                shutil.copy2(fpath, dst)
                saved.append(dst.name)
                manifest_entries.append({
                    "filename": dst.name,
                    "source_video": video_path.name,
                    "timestamp": ts,
                    "sharpness": round(sharpness, 2),
                    "overall": round(overall, 4),
                    "phash": str(h),
                    "duration": round(duration, 2),
                })
            stats["kept"] = len(saved)
            stats["saved"] = saved
            stats["manifest"] = manifest_entries
            # manifest.json：供视频榜排名读取来源/时间戳/客观分/哈希
            # 多视频累积合并，避免后上传的视频覆盖前面的条目
            try:
                all_entries = list(manifest_entries)
                old_mf = out_dir / "manifest.json"
                if old_mf.exists():
                    try:
                        with open(old_mf, "r", encoding="utf-8") as f:
                            old = json.load(f)
                        if isinstance(old, list):
                            all_entries = old + manifest_entries
                    except Exception as e:
                        logger.warning(f"读取旧 manifest 失败，仅保留本次条目: {e}")
                with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
                    json.dump(all_entries, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"manifest.json 写入失败: {e}")

        logger.info(
            f"视频抽帧完成: 候选{stats['candidates']} → 保留{stats['kept']} "
            f"(模糊{stats['blurry']}/重复{stats['dup']}/闭眼{stats['blink']}/失败{stats['failed']})"
        )
        return stats
    finally:
        cap.release()
