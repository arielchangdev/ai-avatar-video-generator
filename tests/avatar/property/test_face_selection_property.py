"""Property-based test for FaceDetector.select_primary_face.

Property 4: Primary face selection by maximum area
For any non-empty list of FaceInfo objects with distinct areas,
select_primary_face SHALL always return the FaceInfo with the largest area.

**Validates: Requirements 2.3**
"""

import sys

sys.path.insert(0, ".")

from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis.strategies import (
    composite,
    integers,
    floats,
    lists,
)

from models.schemas import FaceInfo
from modules.face_detector import FaceDetector


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@composite
def face_info_strategy(draw, area=None):
    """Generate a random FaceInfo object with a specified or random area."""
    if area is None:
        area = draw(integers(min_value=1, max_value=1_000_000))

    # Generate a bbox that matches the area (approximate)
    width = draw(integers(min_value=1, max_value=1000))
    height = max(1, area // width) if width > 0 else area
    x1 = draw(integers(min_value=0, max_value=500))
    y1 = draw(integers(min_value=0, max_value=500))
    x2 = x1 + width
    y2 = y1 + height

    confidence = draw(floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))

    # Generate a small embedding vector (just needs to be a list of floats)
    embedding = [float(i) for i in range(128)]

    # Generate simple landmarks (5 points)
    landmarks = [(float(x1 + i * 10), float(y1 + i * 10)) for i in range(5)]

    return FaceInfo(
        bbox=(x1, y1, x2, y2),
        area=area,
        confidence=confidence,
        embedding=embedding,
        landmarks=landmarks,
    )


@composite
def distinct_area_face_list(draw):
    """Generate a non-empty list of FaceInfo objects with distinct areas."""
    num_faces = draw(integers(min_value=1, max_value=20))
    areas = draw(
        lists(
            integers(min_value=1, max_value=1_000_000),
            min_size=num_faces,
            max_size=num_faces,
            unique=True,
        )
    )

    faces = []
    for area in areas:
        face = draw(face_info_strategy(area=area))
        faces.append(face)

    return faces


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------


@pytest.mark.property
class TestPrimaryFaceSelectionByMaxArea:
    """Property 4: Primary face selection by maximum area.

    **Validates: Requirements 2.3**
    """

    @given(faces=distinct_area_face_list())
    @settings(max_examples=100)
    def test_select_primary_face_returns_largest_area(self, faces):
        """For any non-empty list of FaceInfo with distinct areas,
        select_primary_face always returns the one with the largest area."""
        expected = max(faces, key=lambda f: f.area)

        with patch.object(FaceDetector, "__init__", lambda self, *args, **kwargs: None):
            detector = FaceDetector()
            result = detector.select_primary_face(faces)

        assert result.area == expected.area, (
            f"Expected face with area {expected.area}, got face with area {result.area}"
        )
        assert result == expected, (
            "Expected the exact FaceInfo with largest area to be returned"
        )

    @given(faces=distinct_area_face_list())
    @settings(max_examples=100)
    def test_select_primary_face_invariant_to_order(self, faces):
        """The result of select_primary_face does not depend on list ordering.
        Shuffling the list should always yield the same face (largest area)."""
        expected = max(faces, key=lambda f: f.area)

        with patch.object(FaceDetector, "__init__", lambda self, *args, **kwargs: None):
            detector = FaceDetector()

            # Test with original order
            result = detector.select_primary_face(faces)
            assert result.area == expected.area

            # Test with reversed order
            result_reversed = detector.select_primary_face(list(reversed(faces)))
            assert result_reversed.area == expected.area

    @given(face=face_info_strategy())
    @settings(max_examples=100)
    def test_single_face_always_selected(self, face):
        """A single-element list always returns that element."""
        with patch.object(FaceDetector, "__init__", lambda self, *args, **kwargs: None):
            detector = FaceDetector()
            result = detector.select_primary_face([face])

        assert result == face, "Single face should always be selected as primary"
