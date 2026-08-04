"""PhotoRank 配置 — 所有配置通过环境变量覆盖"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 统一加载 .env：保证所有模块（L3 的 VLM 评分、L5 的 API 调用等）
# 都能读到密钥，避免只有部分代码能读 .env 的不一致问题
load_dotenv()

# 项目路径
PROJECT_ROOT = Path(__file__).parent
PHOTOS_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "output"

# 照片分类
CATEGORIES = ["风景", "人像", "人文", "美食", "夜景", "自拍", "合照", "其他"]

# 平台配置
PLATFORMS = {
    "xiaohongshu": {
        "name": "小红书",
        "style": "氛围感、故事感、封面感",
        "output_count": 12,
        "weights": {
            "composition": 0.35,  # 构图
            "color": 0.30,        # 色彩
            "lighting": 0.20,     # 光影
            "face": 0.15,         # 人脸
        },
    },
    "wechat": {
        "name": "朋友圈",
        "style": "九宫格叙事",
        "output_count": 12,
        "weights": {
            "composition": 0.25,
            "color": 0.25,
            "lighting": 0.30,
            "face": 0.20,
        },
    },
    "douyin": {
        "name": "抖音",
        "style": "冲击力、竖版适配",
        "output_count": 12,
        "weights": {
            "composition": 0.30,
            "color": 0.25,
            "lighting": 0.25,
            "face": 0.20,
        },
    },
}

# Layer 1: 客观指标阈值
LAPLACIAN_THRESHOLD = 100
OVEREXPOSURE_THRESHOLD = 240
UNDEREXPOSURE_THRESHOLD = 15
# 淘汰比例阈值（超过亮度阈值的像素占比）
# 0.4：避免天空/雪景等大面积亮部误杀；0.8：避免夜景等暗调照片误杀
OVEREXPOSURE_RATIO_THRESHOLD = 0.4
UNDEREXPOSURE_RATIO_THRESHOLD = 0.8
# 曝光评分合理亮度区间：区间内不扣分（夜景/雪景等合法场景不再被惩罚）
EXPOSURE_BAND = (70, 180)

# Layer 2: 相似聚类（汉明距离，对应 config 的语义阈值）
SIMILARITY_THRESHOLD = 0.85  # 余弦相似度（备用，当前未被任何模块引用）
HAMMING_THRESHOLD = 10       # 感知哈希汉明距离（layer2_similarity.py 使用）

# Layer 3/5 共用
TOP_K_PER_CATEGORY = 5
FINAL_OUTPUT_COUNT = 12

# 多样性惩罚（Top N 选择前，对相似照片扣分）
# 15% 扣分幅度：足以让相似照片从 Top 12 边缘掉出（通常分差 > 10%），
# 又不至于让一张好照片因轻微相似就被完全挤出榜单。
DIVERSITY_PENALTY = 0.15
DIVERSITY_DISTANCE_THRESHOLD = 10  # pHash 汉明距离低于此值视为相似

# Layer 3: VLM 评分前将图片长边缩放到此尺寸（减小上传体积、提速）
VLM_IMAGE_MAX_SIDE = 1024

# 最终综合分融合权重（总和 1.0）
# vlm: VLM 多维度审美（按平台权重加权构图/色彩/光影）
# topiq: TOPIQ-IAA 客观审美（批次归一化后）
# face: 人脸质量；objective: Layer 1 客观指标
FUSION_WEIGHTS = {
    "vlm": 0.45,
    "topiq": 0.30,
    "face": 0.15,
    "objective": 0.10,
}

# Qwen API
QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3-vl-plus")

# 代理（通过环境变量 HTTPS_PROXY 自动获取）
PROXY = os.environ.get("HTTPS_PROXY", "")

# Web UI
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8080"))
