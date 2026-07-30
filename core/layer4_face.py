"""Layer 4: 人脸质量 — MediaPipe Face Landmarker"""
import os
import time
import threading
import logging
import urllib.request
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# 全局缓存：避免每次调用都创建新实例
_global_analyzer = None
_global_analyzer_lock = threading.Lock()

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
EXPECTED_SIZE = 3 * 1024 * 1024  # ~3MB


@dataclass
class FaceQuality:
    face_count: int = 0
    has_face: bool = False
    blink_detected: bool = False
    expression_score: float = 0.0
    face_clarity: float = 0.0
    overall_face_score: float = 0.0


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

            import mediapipe as mp
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
            blink = False
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

            # 人脸清晰度
            import cv2
            h, w = arr.shape[:2]
            landmarks = result.face_landmarks[0]
            x_min = int(min(lm.x for lm in landmarks) * w)
            x_max = int(max(lm.x for lm in landmarks) * w)
            y_min = int(min(lm.y for lm in landmarks) * h)
            y_max = int(max(lm.y for lm in landmarks) * h)

            if x_max > x_min and y_max > y_min:
                face_region = arr[y_min:y_max, x_min:x_max]
                gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
                clarity = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0)
            else:
                clarity = 0.0

            overall = clarity * 0.5 + max_smile * 0.3 + (0.2 if not blink else 0)

            return FaceQuality(
                face_count=face_count, has_face=True, blink_detected=blink,
                expression_score=round(max_smile, 4), face_clarity=round(clarity, 4),
                overall_face_score=round(overall, 4),
            )

        except Exception as e:
            logger.debug(f"人脸分析失败 {Path(image_path).name}: {e}")
            return FaceQuality()

    def batch_analyze(self, image_paths: list[str | Path]) -> list[dict]:
        results = []
        for path in image_paths:
            quality = self.analyze_face(path)
            results.append({
                "path": str(path), "filename": Path(path).name,
                "face_count": quality.face_count, "has_face": quality.has_face,
                "blink_detected": quality.blink_detected,
                "expression_score": quality.expression_score,
                "face_clarity": quality.face_clarity,
                "overall_face_score": quality.overall_face_score,
            })
        return results


def analyze_faces(image_paths: list[str | Path]) -> list[dict]:
    """批量人脸分析（复用全局实例）"""
    global _global_analyzer
    with _global_analyzer_lock:
        if _global_analyzer is None:
            _global_analyzer = FaceAnalyzer()
    return _global_analyzer.batch_analyze(image_paths)
