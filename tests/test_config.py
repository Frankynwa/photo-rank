"""config.py 配置单元测试"""
import importlib
import os
from unittest.mock import patch

# ---------------------------------------------------------------------------
# 常量值合理性测试
# ---------------------------------------------------------------------------

class TestConfigConstants:
    """测试 config.py 中硬编码常量的合理性"""

    def test_project_root_exists(self):
        from config import PROJECT_ROOT
        assert PROJECT_ROOT.is_dir()

    def test_photos_dir_is_subpath(self):
        from config import PHOTOS_DIR, PROJECT_ROOT
        assert str(PHOTOS_DIR).startswith(str(PROJECT_ROOT))

    def test_output_dir_is_subpath(self):
        from config import OUTPUT_DIR, PROJECT_ROOT
        assert str(OUTPUT_DIR).startswith(str(PROJECT_ROOT))

    def test_categories_non_empty(self):
        from config import CATEGORIES
        assert len(CATEGORIES) >= 5
        assert "其他" in CATEGORIES

    def test_platforms_have_required_keys(self):
        from config import PLATFORMS
        for key, val in PLATFORMS.items():
            assert "name" in val, f"平台 {key} 缺少 name"
            assert "style" in val, f"平台 {key} 缺少 style"
            assert "output_count" in val, f"平台 {key} 缺少 output_count"

    def test_laplacian_threshold_positive(self):
        from config import LAPLACIAN_THRESHOLD
        assert LAPLACIAN_THRESHOLD > 0

    def test_exposure_thresholds_valid_range(self):
        from config import OVEREXPOSURE_THRESHOLD, UNDEREXPOSURE_THRESHOLD
        assert 0 <= UNDEREXPOSURE_THRESHOLD < OVEREXPOSURE_THRESHOLD <= 255

    def test_hamming_threshold_positive(self):
        from config import HAMMING_THRESHOLD
        assert HAMMING_THRESHOLD > 0

    def test_final_output_count_positive(self):
        from config import FINAL_OUTPUT_COUNT
        assert FINAL_OUTPUT_COUNT > 0

    def test_top_k_per_category_positive(self):
        from config import TOP_K_PER_CATEGORY
        assert TOP_K_PER_CATEGORY > 0


# ---------------------------------------------------------------------------
# 环境变量覆盖测试
# ---------------------------------------------------------------------------

class TestEnvOverrides:
    """测试通过环境变量覆盖配置值"""

    def _reload_config(self):
        """重新加载 config 模块以获取新的环境变量"""
        import config
        return importlib.reload(config)

    def test_host_override(self):
        with patch.dict(os.environ, {"HOST": "0.0.0.0"}):
            cfg = self._reload_config()
            assert cfg.HOST == "0.0.0.0"

    def test_port_override(self):
        with patch.dict(os.environ, {"PORT": "9999"}):
            cfg = self._reload_config()
            assert cfg.PORT == 9999

    def test_qwen_model_override(self):
        with patch.dict(os.environ, {"QWEN_MODEL": "qwen-vl-plus"}):
            cfg = self._reload_config()
            assert cfg.QWEN_MODEL == "qwen-vl-plus"

    def test_default_port(self):
        """移除 PORT 环境变量后应回退到默认值 8080"""
        env = os.environ.copy()
        env.pop("PORT", None)
        with patch.dict(os.environ, env, clear=True):
            cfg = self._reload_config()
            assert cfg.PORT == 8080
