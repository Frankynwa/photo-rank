"""PhotoRank 配置 — 所有配置通过环境变量覆盖"""
import os
from pathlib import Path

# 项目路径
PROJECT_ROOT = Path(__file__).parent
PHOTOS_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "output"

# 照片分类
CATEGORIES = ["风景", "人像", "人文", "美食", "夜景", "自拍", "合照", "其他"]

# 平台配置
PLATFORMS = {
    "xiaohongshu": {"name": "小红书", "style": "氛围感、故事感、封面感", "output_count": 12},
    "wechat": {"name": "朋友圈", "style": "九宫格叙事", "output_count": 12},
    "douyin": {"name": "抖音", "style": "冲击力、竖版适配", "output_count": 12},
}

# Layer 1: 客观指标阈值
LAPLACIAN_THRESHOLD = 100
OVEREXPOSURE_THRESHOLD = 240
UNDEREXPOSURE_THRESHOLD = 15

# Layer 2: 相似聚类（汉明距离，对应 config 的语义阈值）
SIMILARITY_THRESHOLD = 0.85  # 余弦相似度（备用，当前未被任何模块引用）
HAMMING_THRESHOLD = 10       # 感知哈希汉明距离（layer2_similarity.py 使用）

# Layer 3/5 共用
TOP_K_PER_CATEGORY = 5
FINAL_OUTPUT_COUNT = 12

# Qwen API
QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL",
    "https://llm-1r8e612iutxlixav.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-vl-max")

# 代理（通过环境变量 HTTPS_PROXY 自动获取）
PROXY = os.environ.get("HTTPS_PROXY", "")

# Web UI
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8080"))
