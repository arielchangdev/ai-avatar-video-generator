"""Unit tests for FaceDetector - face detection, selection, and embedding extraction.

Validates Requirements 2.3 and 2.4:
- 2.3: When multiple faces detected, select the largest face area
- 2.4: When no faces detected, raise error stating素材需包含至少一張清晰正面人臉
"""

import sys

sys.path.insert(0, ".")

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models.schemas import FaceInfo


def _make_face_info(
    bbox=(0, 0, 100, 100),
    area=10000,
    confidence=0.95,
    embedding=None,
    landmarks=None,
):
    """Helper to create FaceInfo instances for testing."""
    return FaceInfo(
        bbox=bbox,
        area=area,
        confidence=confidence,
        embedding=embedding if embedding is not None else [0.1] * 512,
        landmarks=landmarks if landmarks is not None else [(0.0, 0.0)] * 5,
    )


class TestSelectPrimaryFaceEmptyList:
    """Test that select_primary_face raises FaceDetectionError for empty list."""

    def test_empty_face_list_raises_error(self):
        """Empty face list should raise FaceDetectionError with appropriate message."""
        from modules.face_detector import FaceDetectionError, FaceDetector

        detector = FaceDetector.__new__(FaceDetector)
        detector._model_name = "buffalo_l"
        detector._app = MagicMock()

        with pytest.raises(FaceDetectionError) as exc_info:
            detector.select_primary_face([])

        assert "素材需包含至少一張清晰正面人臉" in str(exc_info.value)
        assert exc_info.value.recoverable is True


class TestSelectPrimaryFaceSingleFace:
    """Test that select_primary_face returns the single face when only one is provided."""

    def test_single_face_returns_that_face(self):
        """A list with one face should return that exact face."""
        from modules.face_detector import FaceDetector

        detector = FaceDetector.__new__(FaceDetector)
        detector._model_name = "buffalo_l"
        detector._app = MagicMock()

        face = _make_face_info(bbox=(10, 20, 110, 120), area=10000)
        result = detector.select_primary_face([face])

        assert result is face
        assert result.area == 10000
        assert result.bbox == (10, 20, 110, 120)


class TestSelectPrimaryFaceMultipleFaces:
    """Test that select_primary_face returns the face with the largest area."""

    def test_multiple_faces_returns_largest_area(self):
        """When multiple faces exist, the one with largest area should be returned."""
        from modules.face_detector import FaceDetector

        detector = FaceDetector.__new__(FaceDetector)
        detector._model_name = "buffalo_l"
        detector._app = MagicMock()

        small_face = _make_face_info(bbox=(0, 0, 50, 50), area=2500)
        large_face = _make_face_info(bbox=(0, 0, 200, 200), area=40000)
        medium_face = _make_face_info(bbox=(0, 0, 100, 100), area=10000)

        result = detector.select_primary_face([small_face, large_face, medium_face])

        assert result is large_face
        assert result.area == 40000


class TestModelUnavailable:
    """Test that FaceDetector raises FaceDetectionError when insightface is not installed."""

    def test_import_error_raises_face_detection_error(self):
        """When insightface is not importable, FaceDetectionError should be raised."""
        from modules.face_detector import FaceDetectionError, FaceDetector

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "insightface":
                raise ImportError("No module named 'insightface'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(FaceDetectionError) as exc_info:
                FaceDetector(model_name="buffalo_l")

            assert exc_info.value.recoverable is False
            assert "InsightFace" in str(exc_info.value) or "insightface" in str(exc_info.value).lower()


class TestExtractEmbeddingEmpty:
    """Test that extract_embedding raises FaceDetectionError when embedding is empty."""

    def test_empty_embedding_raises_error(self):
        """A face with no embedding data should raise FaceDetectionError."""
        from modules.face_detector import FaceDetectionError, FaceDetector

        detector = FaceDetector.__new__(FaceDetector)
        detector._model_name = "buffalo_l"
        detector._app = MagicMock()

        face = _make_face_info(embedding=[])

        with pytest.raises(FaceDetectionError) as exc_info:
            detector.extract_embedding(face)

        assert exc_info.value.recoverable is False
        assert "embedding" in str(exc_info.value).lower()

    def test_valid_embedding_returns_numpy_array(self):
        """A face with valid embedding should return a numpy array."""
        from modules.face_detector import FaceDetector

        detector = FaceDetector.__new__(FaceDetector)
        detector._model_name = "buffalo_l"
        detector._app = MagicMock()

        embedding_data = [0.5] * 512
        face = _make_face_info(embedding=embedding_data)

        result = detector.extract_embedding(face)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 512


class TestDetectFacesNoFaces:
    """Test that detect_faces raises FaceDetectionError when no faces are found."""

    def test_no_faces_detected_raises_error(self):
        """When the model detects no faces in the image, raise FaceDetectionError."""
        from modules.face_detector import FaceDetectionError, FaceDetector

        mock_app = MagicMock()
        mock_app.get.return_value = []  # No faces found

        detector = FaceDetector.__new__(FaceDetector)
        detector._model_name = "buffalo_l"
        detector._app = mock_app

        fake_image = np.zeros((640, 640, 3), dtype=np.uint8)

        with pytest.raises(FaceDetectionError) as exc_info:
            detector.detect_faces(fake_image)

        assert "素材需包含至少一張清晰正面人臉" in str(exc_info.value)
        assert exc_info.value.recoverable is True

    def test_model_not_initialized_raises_error(self):
        """When the internal model is None, raise FaceDetectionError."""
        from modules.face_detector import FaceDetectionError, FaceDetector

        detector = FaceDetector.__new__(FaceDetector)
        detector._model_name = "buffalo_l"
        detector._app = None

        fake_image = np.zeros((640, 640, 3), dtype=np.uint8)

        with pytest.raises(FaceDetectionError) as exc_info:
            detector.detect_faces(fake_image)

        assert exc_info.value.recoverable is False
