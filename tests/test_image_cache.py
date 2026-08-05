"""图像解码缓存单元测试 — 小图直读原路径，超大图/HEIC 转 JPEG 缓存复用"""
import os
import time
from pathlib import Path

import pytest
from PIL import Image

import core.image_cache as image_cache


def _make_image(path: Path, size=(100, 100), color=(200, 50, 50)):
    img = Image.new("RGB", size, color)
    img.save(str(path))
    return path


class TestGetCachedPath:
    def test_small_image_returns_original(self, tmp_path, monkeypatch):
        """长边 ≤ 2048 的普通图应直接返回原路径（零转换开销）"""
        monkeypatch.setattr(image_cache, "CACHE_DIR", tmp_path / "cache")
        p = _make_image(tmp_path / "small.jpg", size=(1024, 768))
        assert image_cache.get_cached_path(p) == p

    def test_large_image_converted_and_reused(self, tmp_path, monkeypatch):
        """超大图应转 JPEG 缓存；二次调用复用同一缓存路径（幂等）"""
        monkeypatch.setattr(image_cache, "CACHE_DIR", tmp_path / "cache")
        p = _make_image(tmp_path / "large.jpg", size=(5000, 2000))
        c1 = image_cache.get_cached_path(p)
        c2 = image_cache.get_cached_path(p)
        assert c1 != p  # 转换到缓存路径
        assert c1 == c2  # 幂等复用
        assert c1.exists() and c1.suffix.lower() == ".jpg"
        with Image.open(c1) as img:
            assert max(img.size) <= image_cache.CACHE_MAX_SIDE  # 长边钳制

    def test_heic_converted_even_if_small(self, tmp_path, monkeypatch):
        """HEIC 文件（即使尺寸小）也应转 JPEG 缓存（HEIC 解码昂贵且仅 PIL 支持）"""
        pytest.importorskip("pillow_heif")
        from pillow_heif import register_heif_opener
        register_heif_opener()
        monkeypatch.setattr(image_cache, "CACHE_DIR", tmp_path / "cache")
        p = tmp_path / "photo.heic"
        try:
            Image.new("RGB", (800, 600), "blue").save(str(p), format="HEIF")
        except Exception:
            pytest.skip("pillow-heif 无 HEIF 编码器")
        c = image_cache.get_cached_path(p)
        assert c != p
        assert c.suffix.lower() == ".jpg"
        assert c.exists()
        # 二次调用复用同一缓存（不重复转换）
        assert image_cache.get_cached_path(p) == c

    def test_missing_file_falls_back_to_original(self, tmp_path, monkeypatch):
        """不存在的文件应回退原路径（不崩溃）"""
        monkeypatch.setattr(image_cache, "CACHE_DIR", tmp_path / "cache")
        fake = tmp_path / "missing.jpg"
        assert image_cache.get_cached_path(fake) == fake

    def test_clean_image_cache(self, tmp_path, monkeypatch):
        """clean_image_cache 应删除过期缓存，保留新缓存"""
        monkeypatch.setattr(image_cache, "CACHE_DIR", tmp_path / "cache")
        p = _make_image(tmp_path / "large.jpg", size=(5000, 2000))
        c = image_cache.get_cached_path(p)
        assert c != p and c.exists()
        # 把缓存 mtime 改为 10 天前 → 清理应删除（阈值 7 天）
        old = time.time() - 10 * 86400
        os.utime(c, (old, old))
        removed = image_cache.clean_image_cache(max_age_days=7)
        assert removed >= 1
        assert not c.exists()
        # 再次清理：无过期文件，返回 0
        assert image_cache.clean_image_cache(max_age_days=7) == 0
