"""Layer 3: 审美评分 — NIMA-VGG16（GPU/CPU 自动切换，自动释放显存）"""
import gc
import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


def score_photos(
    image_paths: list[str | Path],
    device: str = "auto",
) -> list[dict]:
    """评分照片（逐张推理。NIMA 不支持 batch 路径列表）"""
    import pyiqa
    import torch

    # 设备检测
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")

    if device == "cpu":
        logger.warning("CPU 推理，速度较慢")

    # 加载模型（带容错 + CPU fallback）
    try:
        logger.info("加载 NIMA-VGG16...")
        model = pyiqa.create_metric("nima-vgg16-ava", device=device)
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        if device != "cpu":
            try:
                logger.info("尝试 CPU fallback...")
                model = pyiqa.create_metric("nima-vgg16-ava", device="cpu")
                device = "cpu"
            except Exception as e2:
                logger.error(f"CPU fallback 也失败: {e2}")
                return [
                    {"path": str(p), "filename": Path(p).name,
                     "aesthetic_score": 0, "technical_score": 0,
                     "overall_score": 0, "final_score": 0,
                     "error": str(e), "device": "cpu"}
                    for p in image_paths
                ]

    results = []
    total = len(image_paths)

    try:
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                score = model(img).item()
                results.append({
                    "path": str(path), "filename": Path(path).name,
                    "aesthetic_score": round(score, 4),
                    "technical_score": round(score, 4),
                    "overall_score": round(score, 4),
                    "final_score": round(score, 4),
                    "error": None, "device": device,
                })
            except Exception as e:
                logger.warning(f"评分失败 {Path(path).name}: {e}")
                results.append({
                    "path": str(path), "filename": Path(path).name,
                    "aesthetic_score": 0, "technical_score": 0,
                    "overall_score": 0, "final_score": 0,
                    "error": str(e), "device": device,
                })

            if (i + 1) % 10 == 0:
                logger.info(f"评分进度: {i + 1}/{total}")

        logger.info(f"NIMA-VGG16 完成（{device}）")
        if device == "cuda":
            logger.info(f"显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    finally:
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            logger.debug("GPU 显存已释放")

    return results
