"""PhotoRank 数据模型 — Pydantic-based type-safe data classes

替代原来的裸 dict 传递，提供：
- IDE 自动补全
- 运行时类型校验
- JSON 序列化兼容
"""
from typing import Optional

from pydantic import BaseModel


class Layer1Result(BaseModel):
    """Layer 1 客观指标输出"""
    path: str
    filename: str
    rejected: bool = False
    reject_reason: str = ""
    sharpness: float = 0.0
    exposure: float = 0.0
    noise: float = 0.0
    overall: float = 0.0


class SimilarityCluster(BaseModel):
    """Layer 2 相似聚类组"""
    cluster_id: int = 0
    member_count: int = 1
    members: list[str] = []
    representative: str = ""  # 清晰度最高的一张（兼容旧字段）
    representatives: list[str] = []  # 按清晰度取前 2 张进入审美层
    filename: str = ""
    member_scores: dict[str, float] = {}


class Layer3Result(BaseModel):
    """Layer 3 审美评分输出"""
    path: str
    filename: str
    aesthetic_score: float = 0.0
    technical_score: float = 0.0
    overall_score: float = 0.0
    final_score: float = 0.0
    composition_score: float = 0
    color_score: float = 0
    lighting_score: float = 0
    vlm_overall_score: float = 0.0  # VLM 综合审美分（>0 表示真实评分，否则为未评分）
    score_reason: str = ""  # VLM 评分理由
    category: str = ""  # VLM 分类标签（风景/建筑/人像/人文纪实/美食/动物/夜景/其他）
    error: Optional[str] = None
    device: str = "cpu"


class Layer4Result(BaseModel):
    """Layer 4 人脸质量输出"""
    path: str
    filename: str
    face_count: int = 0
    has_face: bool = False
    blink_detected: bool = False
    expression_score: float = 0.0
    face_clarity: float = 0.0
    overall_face_score: float = 0.0


class FinalResult(BaseModel):
    """最终输出：评分 + 分类标签"""
    path: str
    filename: str
    aesthetic_score: float
    overall_score: float
    final_score: float
    category: str = ""


class PipelineResult(BaseModel):
    """流水线完整输出"""
    total: int = 0
    rejected: int = 0
    merged: int = 0
    ranked: int = 0
    output_count: int = 0
    output_dir: str = ""
    skipped_layers: list[str] = []
    results: list[FinalResult] = []


class ProgressEvent(BaseModel):
    """进度事件"""
    event: str  # "layer_start" | "layer_end" | "layer_progress" | "error"
    layer: Optional[int] = None
    layer_name: Optional[str] = None
    current: Optional[int] = None
    total: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None
