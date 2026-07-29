"""Layer 4: 人脸质量 — MediaPipe Face Landmarker（v1.0 API）"""
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from PIL import Image


@dataclass
class FaceQuality:
    """人脸质量评估结果"""
    face_count: int
    has_face: bool
    blink_detected: bool
    expression_score: float
    face_clarity: float
    overall_face_score: float


class FaceAnalyzer:
    """人脸质量分析器（MediaPipe v1.0）"""
    
    def __init__(self):
        self.detector = None
        self.initialized = False
    
    def _init_mediapipe(self):
        """初始化 MediaPipe Face Landmarker"""
        if self.initialized:
            return
        
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # 下载模型
            model_path = self._get_model_path()
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                num_faces=10,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            self.initialized = True
            print("  MediaPipe Face Landmarker 初始化成功")
        except Exception as e:
            print(f"  MediaPipe 初始化失败: {e}")
            self.initialized = True
    
    def _get_model_path(self) -> str:
        """获取模型路径（自动下载）"""
        import urllib.request
        import os
        
        model_dir = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "face_landmarker.task")
        
        if not os.path.exists(model_path):
            print("  下载 face_landmarker 模型...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("  模型下载完成")
        
        return model_path
    
    def analyze_face(self, image_path: str | Path) -> FaceQuality:
        """分析单张照片的人脸质量"""
        self._init_mediapipe()
        
        if self.detector is None:
            return FaceQuality(0, False, False, 0, 0, 0)
        
        try:
            # 读取图片
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)
            
            # 创建 MediaPipe Image
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
            
            # 检测
            result = self.detector.detect(mp_image)
            
            if not result.face_landmarks:
                return FaceQuality(0, False, False, 0, 0, 0)
            
            face_count = len(result.face_landmarks)
            
            # 检测闭眼（通过 blendshapes）
            blink_detected = False
            expression_score = 0.0
            
            if result.face_blendshapes:
                for blendshapes in result.face_blendshapes:
                    # 闭眼检测
                    blink_left = next((b.score for b in blendshapes if b.category_name == "eyeBlinkLeft"), 0)
                    blink_right = next((b.score for b in blendshapes if b.category_name == "eyeBlinkRight"), 0)
                    if blink_left > 0.5 or blink_right > 0.5:
                        blink_detected = True
                    
                    # 微笑检测
                    smile_left = next((b.score for b in blendshapes if b.category_name == "mouthSmileLeft"), 0)
                    smile_right = next((b.score for b in blendshapes if b.category_name == "mouthSmileRight"), 0)
                    smile_score = (smile_left + smile_right) / 2
                    expression_score = max(expression_score, smile_score)
            
            # 计算人脸清晰度（基于关键点分布）
            # result.face_landmarks[0] 是 NormalizedLandmark 列表
            first_face = result.face_landmarks[0]
            face_clarity = self._compute_face_clarity(image_array, first_face)
            
            # 综合人脸质量分
            overall = face_clarity * 0.5 + expression_score * 0.3 + (0.2 if not blink_detected else 0)
            
            return FaceQuality(
                face_count=face_count,
                has_face=True,
                blink_detected=blink_detected,
                expression_score=round(expression_score, 4),
                face_clarity=round(face_clarity, 4),
                overall_face_score=round(overall, 4),
            )
        except Exception as e:
            print(f"  人脸分析失败 {Path(image_path).name}: {e}")
            return FaceQuality(0, False, False, 0, 0, 0)
    
    def _compute_face_clarity(self, image: np.ndarray, face_landmarks) -> float:
        """计算人脸区域清晰度"""
        import cv2
        h, w = image.shape[:2]
        
        # face_landmarks 是 NormalizedLandmark 列表
        x_min = int(min(lm.x for lm in face_landmarks) * w)
        x_max = int(max(lm.x for lm in face_landmarks) * w)
        y_min = int(min(lm.y for lm in face_landmarks) * h)
        y_max = int(max(lm.y for lm in face_landmarks) * h)
        
        # 裁剪人脸区域
        face_region = image[y_min:y_max, x_min:x_max]
        if face_region.size == 0:
            return 0
        
        # 计算清晰度（拉普拉斯方差）
        gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
        clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return min(clarity / 500.0, 1.0)
    
    def batch_analyze(self, image_paths: list[str | Path]) -> list[dict]:
        """批量分析人脸质量"""
        results = []
        for path in image_paths:
            quality = self.analyze_face(path)
            results.append({
                "path": str(path),
                "filename": Path(path).name,
                "face_count": quality.face_count,
                "has_face": quality.has_face,
                "blink_detected": quality.blink_detected,
                "expression_score": quality.expression_score,
                "face_clarity": quality.face_clarity,
                "overall_face_score": quality.overall_face_score,
            })
        return results


def analyze_faces(image_paths: list[str | Path]) -> list[dict]:
    """批量分析人脸质量"""
    analyzer = FaceAnalyzer()
    return analyzer.batch_analyze(image_paths)
