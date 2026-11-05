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
VIDEOS_DIR = PROJECT_ROOT / "uploads" / "videos"  # 视频文件存放目录（抽帧后保留）
VIDEO_FRAMES_DIR = PHOTOS_DIR / "video_frames"  # 视频合格帧独立目录（与照片隔离，供视频榜）

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
# 清晰度阈值：实测 100 附近无天然分界（95~105 连续分布 29 张），
# sharpness 单指标在灰色地带区分不了好坏（同区间 VLM 0.1~9.4 并存），
# 且低纹理场景（天空/水面）天然偏低，100 一刀切会误杀好图。
# 降到 70：只拦截明显糊片（0-70 共 151 张），70-100 灰色地带交给后续层判断。
LAPLACIAN_THRESHOLD = 70
# 人脸区域清晰度参考阈值（与 LAPLACIAN_THRESHOLD 同源）：人脸 bbox 达到此 Laplacian
# 方差视为清晰（clarity=1.0）。历史实现用固定除 500，而人脸区域通常仅占画面 2-15%、
# 像素数远低于整图，导致 53 张检出脸的照片 clarity 全部 <0.3（p90=0.000），指标失效。
FACE_CLARITY_REF_LAPLACIAN = 140.0
# 人脸清晰度尺度补偿的参考短边（像素）：bbox 小于该尺寸时按面积比放大 Laplacian 方差，
# 抵消"区域越小、高频细节越少、方差越低"的尺度依赖
FACE_CLARITY_REF_SHORT_SIDE = 200
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
# 感知哈希汉明距离（layer2_similarity.py 使用）
# 实测 20 过宽导致跨日期/跨场景照片雪崩合并（409 张→30 组、最大组 377）；
# 16 在悬崖前（16: 最大组 4，18: 最大组 186），配合照片榜同日期约束根治误合并
HAMMING_THRESHOLD = 16

# Layer 3/5 共用
TOP_K_PER_CATEGORY = 5
FINAL_OUTPUT_COUNT = 100

# 多样性惩罚（照片榜已由下方场景配额取代，仅视频榜仍在用；保留供回退）
# 15% 扣分幅度：足以让相似照片从 Top 12 边缘掉出（通常分差 > 10%），
# 又不至于让一张好照片因轻微相似就被完全挤出榜单。
# 历史缺陷：DIVERSITY_DISTANCE_THRESHOLD(10) < HAMMING_THRESHOLD(16)，
# 候选对在 L2 阶段已被合并，惩罚循环形同空转；且惩罚发生在排序前，
# 注释所称"j 排名靠后"不成立。照片榜现改由 SCENE_BUCKET_MAX 硬配额约束。
DIVERSITY_PENALTY = 0.15
DIVERSITY_DISTANCE_THRESHOLD = 10  # pHash 汉明距离低于此值视为相似

# ── 选片多样性（排序后贪心 + 硬配额）──
# 场景桶 =（拍摄日期, 30 分钟窗口）。依据：实测 Top100 中 13 个 30 分钟桶 ≥3 张、
# 最大单桶 11 张，而标准 64-bit pHash 在 Top100 内只识别出 1 对相似，
# 说明 pHash 无法表达"同场景不同构图"，只能靠拍摄时间约束。
SCENE_BUCKET_MINUTES = 30     # 时间桶粒度（分钟）
SCENE_BUCKET_MAX = 2          # 同一时间桶最多入选张数（配额过严导致选不满时自动放宽）
SCENE_BUCKET_MAX_RELAXED = 3  # 首选配额选不满时的放宽值（仍不满则不限）

# 类别保底配额（占输出总量的比例）：旅行照里"人"最可能被发出来，
# 但实测人像入榜率仅 7.2%、动物 3.8%，远低于风景 14.1%，属系统性失衡。
# 缺口从已选中最低分的非保底类照片置换，不增加总量；候选不足时按可用量给。
CATEGORY_MIN_RATIO = {
    "人像": 0.10,
    "人文纪实": 0.15,
}
# 类别保底的质量地板：置换候选的融合分需 ≥ 当前入选最低分 × 该系数。
# 防止为了凑配额把人像照片硬塞进榜（用户要的是"好的人像"，不是"任意人像"）。
CATEGORY_QUOTA_MIN_SCORE_RATIO = 0.85
# 前段保底：在 PLATFORMS[*].output_count（用户实际发帖量，如小红书 12 张）内，
# 每个保底类别至少 1 张。实测 Top100 已含 10 张人像，但前 12 张仍是 0 张人像——
# 用户只看前 12 张，全文配额对主观感受无效。
TOP_SEGMENT_ENSURE_CATEGORY = True

# ── 硬伤封顶（代码侧强制，不依赖 VLM 遵守 prompt）──
# 实测：VLM 报"画面杂乱"295 张中 94.2% 的构图分 >4、报"歪斜"100% >4，
# prompt 里"存在硬伤 → 对应维度上限 4 分"从未被执行。
# 关闭后 VLM 自报硬伤不影响分数（不推荐，仅用于回退对比）。
HARD_FLAW_CAP_ENABLED = True
HARD_FLAW_DIM_CAP = 4.0  # 命中硬伤时对应维度分的上限

# 视频抽帧：流式逐帧筛选，不合格帧不落盘
VIDEO_FRAME_INTERVAL = 1.0  # 候选帧采样间隔（秒）
VIDEO_MAX_FRAMES = 200  # 单视频候选帧上限（超出自动放大间隔）
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
VIDEO_OUTPUT_COUNT = 20  # 视频精选榜总名额（按各视频时长比例分配）
# 视频帧模糊判定（照片阈值不适用：HEVC 压缩 + 4K 稀释使 Laplacian 天然偏低）
# 绝对底线：低于此值视为解码异常/纯色帧，直接淘汰
VIDEO_FRAME_MIN_SHARPNESS = 3.0
# 批内相对淘汰比例：按 Laplacian 排序淘汰最低的 30%（运动模糊/失焦帧）
VIDEO_FRAME_BLUR_RATIO = 0.3

# Layer 3: VLM 评分前将图片长边缩放到此尺寸（减小上传体积、提速）
VLM_IMAGE_MAX_SIDE = 1024
# VLM 上传 JPEG 质量（85→80：体积约 -15%，视觉差异对 VLM 判断可忽略；体积直接影响 API 延迟）
VLM_JPEG_QUALITY = 80
# VLM 饱和重采样：原始分饱和（综合 ≥9.8 或三项维度全 ≥9）的照片追加一次评分取均值。
# 默认关闭：每饱和一张多一次 API 调用；历史数据饱和占比 ~11%，全开成本不可忽视
VLM_SATURATION_RESAMPLE = False
# VLM 批次归一化（p5-p95 拉伸到 0-10）
# 默认关闭：VLM 已按 prompt 的 1-10 绝对标准打分（5=随手拍、8=封面级），
# 批次拉伸会把绝对质量改写成"批次内相对位次"——实测 Top100 全距被压到 0.96，
# 而 VLM 与 TOPIQ 的平均分歧达 2.32，排序退化为噪声内洗牌。
# 关闭后 vlm_overall_score 直接采用原始绝对分（vlm_raw_score 仍固化保存，便于幂等切换）。
VLM_BATCH_NORMALIZE = False

# ── TOPIQ 通道尺度处理 ──
# 关闭批次归一化：TOPIQ-IAA 的 score_range 本来就是 '1,10'（原始 MOS），与 VLM 同尺度，
# p5-p95 拉伸会把"本批最差"压成 0 分、"本批最好"拉到 10 分（实测 overall_score 全距
# 被铺满 0-10，p5=0.036/p95=10.0），分数含义随批次漂移。
# 关闭后 overall_score = topiq_raw_score（原始 MOS），跨批次可比。
TOPIQ_BATCH_NORMALIZE = False
# 固定标定锚点（原始 MOS → 0-10）：标定一次全批次共用，避免批次依赖。
# None = 不标定直接用原始 MOS。
#
# 实测 801 张旅行照（RTX 5070, pyiqa 0.1.16）：
#   min 2.855 / p05 4.035 / p50 4.850 / p95 5.476 / max 6.180 / span 3.325
# 取 (4.0, 6.0)：TOPIQ 在旅行摄影上的实际判别区间被映射到 0-10。
# 为何不用 p5-p95 锚：那是批次依赖的，换一批照片就漂移；4.0/6.0 是固定值，
# 同一张照片在任何批次都得同一分。
# 为何不直接用原始 MOS（None）：0.30 权重下 TOPIQ 只贡献 stdev 0.45 的方差，
# 不标定时 top-30 张的融合分被挤在 7.375~7.756（span 0.38），
# 且全榜无一张能到 8 分（天花板被压死）。
TOPIQ_CALIBRATION = (4.0, 6.0)

# 最终综合分融合权重（总和 1.0）
# vlm: VLM 多维度审美（按平台权重加权构图/色彩/光影）
# topiq: TOPIQ-IAA 客观审美（原始 MOS 1-10 绝对分，见 TOPIQ_BATCH_NORMALIZE）
# face: 人脸质量；objective: Layer 1 客观指标
FUSION_WEIGHTS = {
    "vlm": 0.45,
    "topiq": 0.30,
    "face": 0.15,
    "objective": 0.10,
}

# ── 三桶分级（精选 / 可用 / 淘汰）──
# 门槛 = max(批次分位数, 固定底线)，取严一侧：
#   分位数吸收批次难度（整批都好片时不至于全部进精选）；
#   固定底线保住绝对语义（整批都废片时不会因“相对较好”而进精选）。
# 精选/淘汰门槛取 max 而非 min：两边都是“数字越大越严”，
# 取 max 才能让两道闸都生效。
# 实测依据（801 张旅行照，TOPIQ 已改绝对分）：P85≈7.45、P20≈4.48，
# 在固定底线 7.0/4.0 之上，因而实际生效的是分位数一侧；
# 底线只在“整批极差”或“整批极好”时兜住语义。
BUCKET_SELECT_PERCENTILE = 0.85   # 精选：批次 P85
BUCKET_SELECT_FLOOR = 7.0         # 精选固定底线（VLM prompt 语义：6-7 值得分享，8+ 专业级）
BUCKET_REJECT_PERCENTILE = 0.20   # 淘汰：批次 P20
BUCKET_REJECT_FLOOR = 4.0         # 淘汰固定底线（VLM prompt 语义：4 分以下一般不值得保留）

# ── 题材化维度权重（覆盖平台权重；四项必须齐全且和为 1.0）──
# 依据：平台统一权重会让"夜色氛围好但构图普通"输给"白天构图工整但无光影"。
# 仅在 VLM 输出 category 命中时生效，未列出的题材（含"其他"）沿用平台权重。
# face 项只在 VLM 未评分的降级路径生效（VLM 已评分时 face 通道归零重分配）。
CATEGORY_WEIGHT_OVERRIDES = {
    "人像":     {"composition": 0.30, "color": 0.25, "lighting": 0.15, "face": 0.30},
    "夜景":     {"composition": 0.20, "color": 0.30, "lighting": 0.40, "face": 0.10},
    "美食":     {"composition": 0.25, "color": 0.45, "lighting": 0.25, "face": 0.05},
    "建筑":     {"composition": 0.45, "color": 0.20, "lighting": 0.28, "face": 0.07},
    "动物":     {"composition": 0.35, "color": 0.25, "lighting": 0.20, "face": 0.20},
    "风景":     {"composition": 0.40, "color": 0.25, "lighting": 0.28, "face": 0.07},
    "人文纪实": {"composition": 0.35, "color": 0.25, "lighting": 0.25, "face": 0.15},
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
