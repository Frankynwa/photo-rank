"""Layer 1: 客观指标 — 清晰度、曝光、噪声（多线程 + HEIC 支持）"""
import cv2
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import LAPLACIAN_THRESHOLD, OVEREXPOSURE_THRESHOLD, UNDEREXPOSURE_THRESHOLD

logger = logging.getLogger(__name__)

# 注册 HEIC 支持
_HEIC_SUPPORT = False
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIC_SUPPORT = True
except ImportError:
    logger.warning("pillow-heif 未安装，HEIC 文件可能无法识别。安装: pip install pillow-heif")


@dataclass
class ObjectiveScore:
    sharpness: float = 0
    exposure: float = 0
    noise: float = 0
    overall: float = 0
    is_blurry: bool = False
    is_overexposed: bool = False
    is_underexposed: bool = False
    rejected: bool = False
    reject_reason: str = ""


def load_image(image_path: str | Path) -> np.ndarray | None:
    """加载图片（支持 HEIC），失败返回 None + 日志"""
    path = Path(image_path)
    suffix = path.suffix.lower()

    try:
        if suffix in ('.heic', '.heif'):
            if not _HEIC_SUPPORT:
                logger.error(f"{path.name}: HEIC 不支持，请安装 pillow-heif")
                return None
            img = Image.open(path).convert("RGB")
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(str(path))
            if img is None:
                logger.warning(f"{path.name}: cv2.imread 返回 None，尝试 PIL...")
                img = Image.open(path).convert("RGB")
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return img
    except Exception as e:
        logger.error(f"{path.name}: 加载失败 - {e}")
        return None


def _analyze_metrics(image: np.ndarray) -> tuple:
    """一次性计算所有指标（共享灰度图，避免三次 cvtColor）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 清晰度
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = lap.var()

    # 曝光
    mean_brightness = np.mean(gray)
    over_ratio = np.sum(gray > OVEREXPOSURE_THRESHOLD) / gray.size
    under_ratio = np.sum(gray < UNDEREXPOSURE_THRESHOLD) / gray.size
    exposure_score = 1.0 - abs(mean_brightness - 128) / 128

    # 噪声
    noise_std = np.std(lap)
    signal = mean_brightness
    snr = signal / noise_std if noise_std > 0 else 999
    noise_score = min(snr / 100.0, 1.0)

    return sharpness, exposure_score, over_ratio, under_ratio, noise_score


def analyze_image(image_path: str | Path) -> ObjectiveScore:
    """分析单张照片的客观指标"""
    image = load_image(image_path)
    if image is None:
        return ObjectiveScore(rejected=True, reject_reason="无法读取图片")

    sharpness, exposure, over_ratio, under_ratio, noise = _analyze_metrics(image)

    sharpness_norm = min(sharpness / 500.0, 1.0)
    overall = sharpness_norm * 0.4 + exposure * 0.3 + noise * 0.3

    is_blurry = sharpness < LAPLACIAN_THRESHOLD
    is_over = over_ratio > 0.3
    is_under = under_ratio > 0.5

    rejected = is_blurry or is_over or is_under
    reasons = []
    if is_blurry:
        reasons.append("模糊")
    if is_over:
        reasons.append("过曝")
    if is_under:
        reasons.append("欠曝")

    return ObjectiveScore(
        sharpness=sharpness, exposure=exposure, noise=noise, overall=overall,
        is_blurry=is_blurry, is_overexposed=is_over, is_underexposed=is_under,
        rejected=rejected, reject_reason=", ".join(reasons),
    )


def batch_analyze(image_paths: list[str | Path], max_workers: int = 4) -> list[dict]:
    """批量分析照片（多线程）"""
    paths = [str(p) for p in image_paths]
    results = [None] * len(paths)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(analyze_image, p): i for i, p in enumerate(paths)}

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                score = future.result()
                results[idx] = {
                    "path": paths[idx],
                    "filename": Path(paths[idx]).name,
                    "sharpness": round(score.sharpness, 2),
                    "exposure": round(score.exposure, 4),
                    "noise": round(score.noise, 4),
                    "overall": round(score.overall, 4),
                    "is_blurry": score.is_blurry,
                    "is_overexposed": score.is_overexposed,
                    "is_underexposed": score.is_underexposed,
                    "rejected": score.rejected,
                    "reject_reason": score.reject_reason,
                }
            except Exception as e:
                logger.error(f"分析失败 {Path(paths[idx]).name}: {e}")
                results[idx] = {
                    "path": paths[idx], "filename": Path(paths[idx]).name,
                    "sharpness": 0, "exposure": 0, "noise": 0, "overall": 0,
                    "rejected": False, "reject_reason": f"分析异常: {e}",
                }

    return results
