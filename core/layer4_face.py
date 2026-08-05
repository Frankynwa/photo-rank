"""Layer 4: 人脸质量 — MediaPipe Face Landmarker"""
import logging
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from core.models import Layer4Result

logger = logging.getLogger(__name__)

# 全局缓存：避免每次调用都创建新实例
_global_analyzer = None
_global_analyzer_lock = threading.Lock()
# 全局调用锁：MediaPipe FaceLandmarker 官方非线程安全，
# 视频抽帧的 L4 与流水线 L4 可能经 asyncio.to_thread 并发调用，串行化防 C++ 层崩溃
_analyze_call_lock = threading.Lock()

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
EXPECTED_SIZE = 3 * 1024 * 1024  # ~3MB


@dataclass
class FaceQuality:
    face_count: int = 0
    has_face: bool = False
    blink_detected: bool = False  # 任一检测到的人脸闭眼（合影硬伤用）
    expression_score: float = 0.0
    face_clarity: float = 0.0
    overall_face_score: float = 0.0
    face_ratio: float = 0.0          # 主脸 bbox 面积 / 画面面积
    second_face_ratio: float = 0.0   # 次脸面积 / 主脸面积（0 = 仅一张脸）
    main_face_blink: bool = False    # 主脸（面积最大者）是否闭眼


class FaceAnalyzer:
    def __init__(self):
        self.detector = None
        self.initialized = False
        self._init_attempts = 0
        self._max_attempts = 3

    def _download_model(self, model_path: str) -> bool:
        """下载模型（带超时和完整性校验）"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        try:
            logger.info("下载 face_landmarker 模型 (~3MB)...")
            urllib.request.urlretrieve(MODEL_URL, model_path)
            time.sleep(0.5)

            size = os.path.getsize(model_path)
            if size < EXPECTED_SIZE * 0.5:
                logger.error(f"模型文件异常小 ({size} bytes)，可能下载不完整")
                os.remove(model_path)
                return False

            logger.info(f"模型下载完成 ({size / 1024 / 1024:.1f} MB)")
            return True
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)
            return False

    def _get_model_path(self) -> str:
        """获取模型路径（自动下载）"""
        model_dir = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")
        model_path = os.path.join(model_dir, "face_landmarker.task")

        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            if size > EXPECTED_SIZE * 0.5:
                return model_path
            logger.warning("模型文件损坏，重新下载...")
            os.remove(model_path)

        if self._download_model(model_path):
            return model_path
        return ""

    def _init_mediapipe(self):
        if self.initialized:
            return
        if self._init_attempts >= self._max_attempts:
            logger.warning(f"MediaPipe 已失败 {self._max_attempts} 次，不再重试")
            return

        self._init_attempts += 1

        try:
            model_path = self._get_model_path()
            if not model_path:
                raise RuntimeError("模型下载失败")

            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                num_faces=10,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            self.initialized = True
            logger.info("MediaPipe Face Landmarker 初始化成功")

        except Exception as e:
            logger.error(f"MediaPipe 初始化失败 (第 {self._init_attempts} 次): {e}")

    def analyze_face(self, image_path: str | Path) -> FaceQuality:
        self._init_mediapipe()
        if self.detector is None:
            logger.debug(f"MediaPipe 不可用，跳过: {Path(image_path).name}")
            return FaceQuality()

        try:
            image = Image.open(image_path).convert("RGB")
            arr = np.array(image)

            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
            result = self.detector.detect(mp_image)

            if not result.face_landmarks:
                return FaceQuality()

            face_count = len(result.face_landmarks)
            blink = False  # 任一闭眼（合影硬伤）
            main_blink = False  # 主脸闭眼
            max_smile = 0.0

            if result.face_blendshapes:
                for blendshapes in result.face_blendshapes:
                    blink_left = next((b.score for b in blendshapes if b.category_name == "eyeBlinkLeft"), 0)
                    blink_right = next((b.score for b in blendshapes if b.category_name == "eyeBlinkRight"), 0)
                    if blink_left > 0.5 or blink_right > 0.5:
                        blink = True
                    smile_left = next((b.score for b in blendshapes if b.category_name == "mouthSmileLeft"), 0)
                    smile_right = next((b.score for b in blendshapes if b.category_name == "mouthSmileRight"), 0)
                    max_smile = max(max_smile, (smile_left + smile_right) / 2)

            # 计算每张脸的 bbox 面积，按面积排序定位主脸（L4 只产出客观几何信息，不判定主体）
            import cv2
            h, w = arr.shape[:2]
            face_sizes = []  # (area, index, x_min, y_min, x_max, y_max)
            for i, lm in enumerate(result.face_landmarks):
                x_min = int(min(p.x for p in lm) * w)
                x_max = int(max(p.x for p in lm) * w)
                y_min = int(min(p.y for p in lm) * h)
                y_max = int(max(p.y for p in lm) * h)
                if x_max > x_min and y_max > y_min:
                    area = (x_max - x_min) * (y_max - y_min)
                    face_sizes.append((area, i, x_min, y_min, x_max, y_max))

            face_ratio = 0.0
            second_face_ratio = 0.0
            clarity = 0.0
            if face_sizes:
                face_sizes.sort(key=lambda t: t[0], reverse=True)
                main_area, main_idx, x_min, y_min, x_max, y_max = face_sizes[0]
                face_ratio = main_area / (w * h)
                if len(face_sizes) >= 2:
                    second_face_ratio = face_sizes[1][0] / main_area
                # 主脸闭眼：face_landmarks 与 face_blendshapes 顺序一致
                if result.face_blendshapes and main_idx < len(result.face_blendshapes):
                    bs = result.face_blendshapes[main_idx]
                    bl = next((b.score for b in bs if b.category_name == "eyeBlinkLeft"), 0)
                    br = next((b.score for b in bs if b.category_name == "eyeBlinkRight"), 0)
                    main_blink = bl > 0.5 or br > 0.5
                # 主脸清晰度
                face_region = arr[y_min:y_max, x_min:x_max]
                gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
                clarity = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0)

            # 综合分：仅主脸闭眼扣分（背景人群闭眼不影响主体）
            overall = clarity * 0.5 + max_smile * 0.3 + (0.2 if not main_blink else 0)

            return FaceQuality(
                face_count=face_count, has_face=True, blink_detected=blink,
                main_face_blink=main_blink,
                expression_score=round(max_smile, 4), face_clarity=round(clarity, 4),
                face_ratio=round(face_ratio, 4), second_face_ratio=round(second_face_ratio, 4),
                overall_face_score=round(overall, 4),
            )

        except Exception as e:
            logger.debug(f"人脸分析失败 {Path(image_path).name}: {e}")
            return FaceQuality()

    def batch_analyze(self, image_paths: list[str | Path]) -> list[Layer4Result]:
        results = []
        for path in image_paths:
            quality = self.analyze_face(path)
            results.append(Layer4Result(
                path=str(path), filename=Path(path).name,
                face_count=quality.face_count, has_face=quality.has_face,
                blink_detected=quality.blink_detected,
                expression_score=quality.expression_score,
                face_clarity=quality.face_clarity,
                overall_face_score=quality.overall_face_score,
                face_ratio=quality.face_ratio,
                second_face_ratio=quality.second_face_ratio,
                main_face_blink=quality.main_face_blink,
            ))
        return results


def analyze_faces(image_paths: list[str | Path]) -> list[Layer4Result]:
    """批量人脸分析（复用全局实例，串行化调用保证线程安全）"""
    global _global_analyzer
    with _global_analyzer_lock:
        if _global_analyzer is None:
            _global_analyzer = FaceAnalyzer()
    with _analyze_call_lock:
        return _global_analyzer.batch_analyze(image_paths)


def to_face_prompt_summary(r: Layer4Result) -> str:
    """将 L4 人脸检测结果转为中性分档提示文案（供 L3 VLM prompt 注入）

    只输出客观事实 + 中性可能性提示（"疑似合影"/"疑似单主体+背景人群"），
    主体判定一律交给 VLM 结合画面内容自行裁断（见 L4与VLM职责划分规范）。
    """
    if not r.has_face or r.face_count == 0:
        return "【人脸检测信息】未检测到人脸，请聚焦场景本身。"

    n = r.face_count

    # 主脸清晰度定性
    if r.face_clarity >= 0.5:
        clarity = "清晰"
    elif r.face_clarity >= 0.25:
        clarity = "轻微模糊"
    else:
        clarity = "明显模糊"

    # 主脸占比定性
    if r.face_ratio >= 0.30:
        size_desc = "特写"
    elif r.face_ratio >= 0.10:
        size_desc = "半身"
    else:
        size_desc = "环境人像"

    # 面积分布：≥0.7 均匀 / 0.5-0.7 略异（归入合影档）/ <0.5 主次分明
    dist = "均匀" if r.second_face_ratio >= 0.7 else ("主次分明" if r.second_face_ratio < 0.5 else "大小略有差异")

    main_blink = "是" if r.main_face_blink else "否"
    any_blink = "是" if r.blink_detected else "否"

    if n == 1:
        return (f"【人脸检测信息】检测到 1 张人脸；主脸闭眼：{main_blink}；"
                f"主脸清晰度：{clarity}；人脸占比：{size_desc}。")

    if n <= 9:
        if dist == "主次分明":
            return (f"【人脸检测信息】检测到 {n} 张人脸，主次分明（疑似单主体+背景人群）；"
                    f"主脸闭眼：{main_blink}；主脸清晰度：{clarity}；主脸占比：{size_desc}。")
        if dist == "大小略有差异":
            return (f"【人脸检测信息】检测到 {n} 张人脸，大小略有差异（疑似合影）；"
                    f"存在闭眼者：{any_blink}；主脸清晰度：{clarity}；人脸占比：{size_desc}。")
        return (f"【人脸检测信息】检测到 {n} 张人脸，面积均匀分布（疑似合影）；"
                f"存在闭眼者：{any_blink}；主脸清晰度：{clarity}；人脸占比：{size_desc}。")

    # 10 张以上
    if dist == "主次分明":
        return (f"【人脸检测信息】检测到 {n} 张人脸（多人场景，主次分明，疑似单主体+环境人群）；"
                f"主脸闭眼：{main_blink}；主脸清晰度：{clarity}；主脸占比：{size_desc}。")
    if dist == "大小略有差异":
        return (f"【人脸检测信息】检测到 {n} 张人脸（多人场景，大小略有差异，疑似合影）；"
                f"存在闭眼者：{any_blink}；人脸占比：{size_desc}。")
    return (f"【人脸检测信息】检测到 {n} 张人脸（多人场景，面积均匀，疑似合影）；"
            f"存在闭眼者：{any_blink}；人脸占比：{size_desc}。")
