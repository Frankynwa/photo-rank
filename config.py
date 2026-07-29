"""PhotoRank 配置"""
from pathlib import Path

# 项目路径
PROJECT_ROOT = Path(__file__).parent
PHOTOS_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / "cache"

# 照片分类
CATEGORIES = [
    "风景",    # 自然风光、草原、山、水、天空
    "人像",    # 自拍、他拍（单人）
    "人文",    # 建筑、街景、文化、民俗
    "美食",
    "夜景",
    "自拍",    # 手机前置摄像头
    "合照",    # 多人合影
    "其他",
]

# 平台配置
PLATFORMS = {
    "xiaohongshu": {
        "name": "小红书",
        "style": "氛围感、故事感、真实感、封面感",
        "output_count": 12,
    },
    "wechat": {
        "name": "朋友圈",
        "style": "九宫格叙事、风景:人文:人像 = 2:2:1",
        "output_count": 12,
    },
    "douyin": {
        "name": "抖音",
        "style": "冲击力、视觉张力、封面吸引力、竖版适配",
        "output_count": 12,
    },
}

# Layer 1: 客观指标阈值
LAPLACIAN_THRESHOLD = 100    # 清晰度，低于此值为模糊
OVEREXPOSURE_THRESHOLD = 240 # 过曝阈值
UNDEREXPOSURE_THRESHOLD = 15 # 欠曝阈值

# Layer 2: 相似聚类
SIMILARITY_THRESHOLD = 0.85  # 余弦相似度阈值

# Layer 3: 审美评分
TOP_K_PER_CATEGORY = 5       # 每类选 Top 5
FINAL_OUTPUT_COUNT = 12      # 最终每平台输出

# Layer 5: Qwen-VL API
QWEN_BASE_URL = "https://llm-1r8e612iutxlixav.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-vl-max"

# 代理
PROXY = "http://127.0.0.1:7890"

# Web UI
HOST = "127.0.0.1"
PORT = 8080
