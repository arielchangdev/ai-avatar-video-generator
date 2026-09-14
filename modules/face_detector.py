"""AI Avatar Video Generator - Face detection and analysis module.

Based on InsightFace for face detection, alignment, and embedding extraction.
Handles graceful degradation when InsightFace is not installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from core.exceptions import PipelineError
from models.schemas import FaceInfo, TaskStage

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FaceDetectionError(PipelineError):
    """人臉偵測相關錯誤"""

    def __init__(self, reason: str, recoverable: bool = True):
        super().__init__(
            stage=TaskStage.FACE_MODELING,
            reason=reason,
            recoverable=recoverable,
        )


class FaceDetector:
    """基於 InsightFace 的人臉偵測與分析模組"""

    def __init__(self, model_name: str = "buffalo_l"):
        """載入 InsightFace 模型。

        Args:
            model_name: InsightFace model pack name (default: buffalo_l).

        Raises:
            FaceDetectionError: If InsightFace is not installed or model fails to load.
        """
        self._model_name = model_name
        self._app = None

        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise FaceDetectionError(
                reason=(
                    "InsightFace is not installed. "
                    "Install with: pip install insightface onnxruntime-gpu"
                ),
                recoverable=False,
            ) from e

        try:
            self._app = FaceAnalysis(name=model_name)
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace model '%s' loaded successfully.", model_name)
        except Exception as e:
            raise FaceDetectionError(
                reason=f"Failed to load InsightFace model '{model_name}': {e}",
                recoverable=False,
            ) from e

    def detect_faces(self, image: np.ndarray) -> list[FaceInfo]:
        """偵測圖片中的所有人臉。

        Args:
            image: Input image as numpy array (BGR format, as used by OpenCV).

        Returns:
            List of FaceInfo objects for each detected face.

        Raises:
            FaceDetectionError: If no faces are detected in the image.
        """
        if self._app is None:
            raise FaceDetectionError(
                reason="Face detection model is not initialized.",
                recoverable=False,
            )

        faces = self._app.get(image)

        if not faces:
            raise FaceDetectionError(
                reason="素材需包含至少一張清晰正面人臉",
                recoverable=True,
            )

        face_infos: list[FaceInfo] = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            area = (x2 - x1) * (y2 - y1)

            # Extract landmarks (5-point or 106-point depending on model)
            landmarks: list[tuple[float, float]] = []
            if face.landmark_2d_106 is not None:
                landmarks = [
                    (float(pt[0]), float(pt[1])) for pt in face.landmark_2d_106
                ]
            elif face.kps is not None:
                landmarks = [(float(pt[0]), float(pt[1])) for pt in face.kps]

            # Extract embedding
            embedding: list[float] = []
            if face.embedding is not None:
                embedding = face.embedding.tolist()

            face_info = FaceInfo(
                bbox=(x1, y1, x2, y2),
                area=area,
                confidence=float(face.det_score),
                embedding=embedding,
                landmarks=landmarks,
            )
            face_infos.append(face_info)

        return face_infos

    def select_primary_face(self, faces: list[FaceInfo]) -> FaceInfo:
        """選取面積最大的人臉。

        Args:
            faces: List of detected FaceInfo objects.

        Returns:
            The FaceInfo with the largest area.

        Raises:
            FaceDetectionError: If the face list is empty.
        """
        if not faces:
            raise FaceDetectionError(
                reason="素材需包含至少一張清晰正面人臉",
                recoverable=True,
            )

        return max(faces, key=lambda f: f.area)

    def extract_embedding(self, face: FaceInfo) -> np.ndarray:
        """提取人臉特徵向量（用於相似度比對）。

        Args:
            face: A FaceInfo object containing the embedding.

        Returns:
            Numpy array of the face embedding vector.

        Raises:
            FaceDetectionError: If the face has no embedding data.
        """
        if not face.embedding:
            raise FaceDetectionError(
                reason="Face embedding is not available. Ensure the model supports embedding extraction.",
                recoverable=False,
            )

        return np.array(face.embedding, dtype=np.float32)
