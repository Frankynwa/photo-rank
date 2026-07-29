"""Layer 1: 客观指标 — 清晰度、曝光、噪声（支持 HEIC）"""
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from PIL import Image

# 注册 HEIC 支持
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


@dataclass
class ObjectiveScore:
    """客观指标评分结果"""
    sharpness: float      # 清晰度（拉普拉斯方差）
    exposure: float       # 曝光质量（0-1，越高越好）
    noise: float          # 噪声（信噪比，越高越好）
    overall: float        # 综合分（0-1）
    is_blurry: bool       # 是否模糊
    is_overexposed: bool  # 是否过曝
    is_underexposed: bool # 是否欠曝
    rejected: bool        # 是否被淘汰
    reject_reason: str    # 淘汰原因


def load_image(image_path: str | Path) -> np.ndarray:
    """加载图片（支持 HEIC）"""
    path = Path(image_path)
    
    if path.suffix.lower() in ('.heic', '.heif'):
        try:
            img = Image.open(path).convert("RGB")
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"HEIC 加载失败 {path.name}: {e}")
            return None
    else:
        return cv2.imread(str(path))


def compute_sharpness(image: np.ndarray) -> float:
    """计算清晰度（拉普拉斯方差）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_exposure(image: np.ndarray) -> tuple[float, bool, bool]:
    """计算曝光质量"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    overexposed_ratio = np.sum(gray > 240) / gray.size
    underexposed_ratio = np.sum(gray < 15) / gray.size
    
    exposure_score = 1.0 - abs(mean_brightness - 128) / 128
    
    is_overexposed = overexposed_ratio > 0.3
    is_underexposed = underexposed_ratio > 0.5
    
    return exposure_score, is_overexposed, is_underexposed


def compute_noise(image: np.ndarray) -> float:
    """计算噪声（信噪比）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_estimate = np.std(laplacian)
    signal = np.mean(gray)
    if noise_estimate == 0:
        return 1.0
    snr = signal / noise_estimate
    return min(snr / 100.0, 1.0)


def analyze_image(image_path: str | Path) -> ObjectiveScore:
    """分析单张照片的客观指标"""
    image = load_image(image_path)
    
    if image is None:
        return ObjectiveScore(
            sharpness=0, exposure=0, noise=0, overall=0,
            is_blurry=True, is_overexposed=False, is_underexposed=False,
            rejected=True, reject_reason="无法读取图片"
        )
    
    sharpness = compute_sharpness(image)
    exposure, is_over, is_under = compute_exposure(image)
    noise = compute_noise(image)
    
    sharpness_normalized = min(sharpness / 500.0, 1.0)
    overall = (sharpness_normalized * 0.4 + exposure * 0.3 + noise * 0.3)
    
    is_blurry = sharpness < 100
    rejected = False
    reject_reason = ""
    
    if is_blurry:
        rejected = True
        reject_reason = "模糊"
    elif is_over:
        rejected = True
        reject_reason = "过曝"
    elif is_under:
        rejected = True
        reject_reason = "欠曝"
    
    return ObjectiveScore(
        sharpness=sharpness,
        exposure=exposure,
        noise=noise,
        overall=overall,
        is_blurry=is_blurry,
        is_overexposed=is_over,
        is_underexposed=is_under,
        rejected=rejected,
        reject_reason=reject_reason,
    )


def batch_analyze(image_paths: list[str | Path]) -> list[dict]:
    """批量分析照片客观指标"""
    results = []
    for path in image_paths:
        score = analyze_image(path)
        results.append({
            "path": str(path),
            "filename": Path(path).name,
            "sharpness": round(score.sharpness, 2),
            "exposure": round(score.exposure, 4),
            "noise": round(score.noise, 4),
            "overall": round(score.overall, 4),
            "is_blurry": score.is_blurry,
            "is_overexposed": score.is_overexposed,
            "is_underexposed": score.is_underexposed,
            "rejected": score.rejected,
            "reject_reason": score.reject_reason,
        })
    return results
