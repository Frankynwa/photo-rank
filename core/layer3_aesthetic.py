"""Layer 3: 审美评分 — NIMA-VGG16（支持 GPU/CPU 自动切换）"""
import gc
import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


def score_photos(
    image_paths: list[str | Path],
    device: str = "auto",
    batch_size: int = 4,
) -> list[dict]:
    """批量评分照片

    Args:
        image_paths: 照片路径列表
        device: "auto"（自动检测）, "cuda", 或 "cpu"
        batch_size: 每批处理张数（仅 GPU 时有效，CPU 固定为 1）
    """
    import pyiqa
    import torch

    # 设备自动检测
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")

    # GPU 时启用 batch，CPU 时逐张
    if device == "cpu":
        batch_size = 1
        logger.warning("CPU 推理，逐张处理（速度较慢）")

    # 加载模型（带容错）
    try:
        logger.info("加载 NIMA-VGG16...")
        model = pyiqa.create_metric("nima-vgg16-ava", device=device)
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        # 尝试 CPU fallback
        if device != "cpu":
            logger.info("尝试 CPU fallback...")
            try:
                model = pyiqa.create_metric("nima-vgg16-ava", device="cpu")
                device = "cpu"
                batch_size = 1
            except Exception as e2:
                logger.error(f"CPU fallback 也失败: {e2}")
                # 全部返回 0
                return [
                    {
                        "path": str(p), "filename": Path(p).name,
                        "aesthetic_score": 0, "technical_score": 0,
                        "overall_score": 0, "final_score": 0,
                        "error": str(e), "device": device,
                    }
                    for p in image_paths
                ]

    results = []
    total = len(image_paths)

    try:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_paths = image_paths[batch_start:batch_end]

            # 加载图片
            images = []
            valid_indices = []
            for i, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    valid_indices.append(batch_start + i)
                except Exception as e:
                    logger.warning(f"无法读取 {Path(p).name}: {e}")
                    results.append({
                        "path": str(p), "filename": Path(p).name,
                        "aesthetic_score": 0, "technical_score": 0,
                        "overall_score": 0, "final_score": 0,
                        "error": str(e), "device": device,
                    })

            if images:
                # batch 推理
                scores = model(images).cpu().tolist()
                if not isinstance(scores, list):
                    scores = [scores]

                for j, score in enumerate(scores):
                    idx = batch_start + j
                    p = image_paths[idx]
                    s = round(float(score), 4)
                    results.append({
                        "path": str(p), "filename": Path(p).name,
                        "aesthetic_score": s,
                        "technical_score": s,
                        "overall_score": s,
                        "final_score": s,
                        "error": None,
                        "device": device,
                    })

            if (batch_start + batch_size) % 10 == 0 or batch_end >= total:
                logger.info(f"评分进度: {batch_end}/{total}")

        logger.info(f"NIMA-VGG16 完成（{device}）")
        if device == "cuda":
            logger.info(f"显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    finally:
        # 释放 GPU 显存
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            logger.debug("GPU 显存已释放")

    return results
