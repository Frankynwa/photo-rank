"""图像解码缓存 — HEIC/超大图首解转 JPEG，消除流水线中 L2/L4/L3 的重复解码

背景：一张照片在流水线中被解码多次（L1 cv2、L2 pHash、L4 PIL、TOPIQ PIL、
VLM PIL+缩放），HEIC 仅 PIL 支持且解码昂贵。本模块把 HEIC/超大图首次解码后
转存为长边 2048 的 JPEG 缓存，后续层直接读缓存，避免重复解码原图。

设计约束：
- L1 保持读原图：清晰度（Laplacian 方差）等指标对缩放敏感，读缓存会漂移淘汰判定
- 普通小图（长边 ≤ CACHE_MAX_SIDE 且非 HEIC）直接返回原路径，零转换开销
- 原子写（.tmp + os.replace）保证多线程 worker 并发转换同一张图时无损坏
- 缓存目录 cache/images/（.gitignore 已排除 cache/）
"""
import hashlib
import logging
import os
import threading
from pathlib import Path

from PIL import Image

from config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# 缓存目录（.gitignore 已排除 cache/）
CACHE_DIR = PROJECT_ROOT / "cache" / "images"

# 超过此长边才转换。4096 而非 2048：iPhone HEIC 长边 4032 几乎不缩放（仅格式转换），
# 避免 L4 人脸清晰度（Laplacian 绝对阈值）因缩放平滑而系统性偏低误触硬伤钳制；
# TOPIQ/VLM 模型输入均为低分辨率，读 4096 缓存再缩到模型输入与原图差异可忽略
CACHE_MAX_SIDE = 4096
CACHE_QUALITY = 90

# 必须转换的格式（HEIC 解码昂贵且仅 PIL 支持）
_NEEDS_CONVERT = {".heic", ".heif"}

# HEIC 解码支持（与 layer1 一致，独立注册保证本模块单独加载时也可用）
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def _cache_path_for(image_path: Path) -> Path:
    """缓存文件名 = 原路径哈希（跨目录无冲突）"""
    digest = hashlib.md5(str(image_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{digest}_w{CACHE_MAX_SIDE}.jpg"


def get_cached_path(image_path: str | Path) -> Path:
    """返回可复用的解码路径

    - 普通小图（长边 ≤ CACHE_MAX_SIDE 且非 HEIC）：返回原路径，零开销
    - HEIC/超大图：首次转换落盘 JPEG 缓存，后续直接读缓存
    - 转换失败时回退原路径（保证正确性优先）
    """
    path = Path(image_path)
    suffix = path.suffix.lower()
    if suffix not in _NEEDS_CONVERT:
        try:
            with Image.open(path) as img:
                w, h = img.size
            if max(w, h) <= CACHE_MAX_SIDE:
                return path
        except Exception as e:
            logger.warning(f"[解码缓存] 尺寸探测失败 {path.name}: {e}，按原图处理")
            return path
    return _ensure_converted(path)


def _ensure_converted(path: Path) -> Path:
    """转换并缓存（原子写：先写唯一 .tmp 再 os.replace，避免并发写同一文件）"""
    cache_path = _cache_path_for(path)
    if cache_path.exists():
        return cache_path
    tmp = None
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > CACHE_MAX_SIDE:
            ratio = CACHE_MAX_SIDE / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
        # 线程唯一后缀：L4∥TOPIQ 并发时同图可能被两个线程同时转换，避免写同一 tmp
        tmp = cache_path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        img.save(tmp, format="JPEG", quality=CACHE_QUALITY)
        os.replace(tmp, cache_path)
        logger.info(f"[解码缓存] {path.name} → {cache_path.name}（{img.width}x{img.height}）")
        return cache_path
    except Exception as e:
        logger.warning(f"[解码缓存] 转换失败 {path.name}: {e}，回退原图")
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass
        return path


def clean_image_cache(max_age_days: int = 7) -> int:
    """清理超过 max_age_days 的过期缓存文件，返回清理数量（防目录无限增长）"""
    import time

    if not CACHE_DIR.exists():
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for f in CACHE_DIR.glob("*.jpg"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            continue
    if removed:
        logger.info(f"[解码缓存] 清理 {removed} 个过期缓存（> {max_age_days} 天）")
    return removed
