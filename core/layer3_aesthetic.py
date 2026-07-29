"""Layer 3: 审美评分 — NIMA-VGG16（RTX 5070 CUDA）"""
import numpy as np
from pathlib import Path
from PIL import Image


def score_photos(image_paths: list[str | Path]) -> list[dict]:
    """批量评分照片（RTX 5070 CUDA）"""
    import pyiqa
    import torch
    
    print("  加载 NIMA-VGG16（CUDA）...")
    model = pyiqa.create_metric("nima-vgg16-ava", device="cuda")
    print("  模型加载成功")
    
    results = []
    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            score = model(img).item()
            
            results.append({
                "path": str(path),
                "filename": Path(path).name,
                "aesthetic_score": round(score, 4),
                "technical_score": round(score, 4),
                "overall_score": round(score, 4),
                "final_score": round(score, 4),
                "error": None,
            })
        except Exception as e:
            results.append({
                "path": str(path),
                "filename": Path(path).name,
                "aesthetic_score": 0,
                "technical_score": 0,
                "overall_score": 0,
                "final_score": 0,
                "error": str(e),
            })
        
        if (i + 1) % 10 == 0:
            print(f"    评分进度: {i + 1}/{len(image_paths)}")
    
    print(f"  NIMA-VGG16 完成，显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    return results
