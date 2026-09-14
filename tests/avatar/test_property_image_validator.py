"""Property-based test for AppearanceAssetValidator - Image validation correctness.

**Validates: Requirements 2.1, 2.5, 2.6, 2.7**

Property 2: Image asset validation correctness
For any image file metadata consisting of format, width, height, file_size,
the AppearanceAssetValidator SHALL accept the file if and only if:
  format in {jpg, jpeg, png, webp} AND width >= 512 AND height >= 512
  AND file_size <= 20MB.
Otherwise it SHALL reject with an error message listing the specific violations.
"""

import sys
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis.strategies import (
    sampled_from,
    integers,
    booleans,
    composite,
)
from PIL import Image

sys.path.insert(0, ".")

from core.validators.appearance_validator import AppearanceAssetValidator


# ---------------------------------------------------------------------------
# Constants matching the validator
# ---------------------------------------------------------------------------
ALLOWED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "webp"]
DISALLOWED_FORMATS = ["bmp", "gif", "tiff"]
ALL_FORMATS = ALLOWED_IMAGE_FORMATS + DISALLOWED_FORMATS
MIN_RESOLUTION = 512
MAX_IMAGE_SIZE_MB = 20
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
@composite
def image_metadata(draw):
    """Generate random image metadata: format, width, height, oversized flag."""
    fmt = draw(sampled_from(ALL_FORMATS))
    width = draw(integers(min_value=100, max_value=2000))
    height = draw(integers(min_value=100, max_value=2000))
    oversized = draw(booleans())
    return fmt, width, height, oversized


def _create_test_image(tmp_dir, fmt, width, height, oversized):
    ext = fmt
    if fmt in ("jpg", "jpeg"):
        save_format = "JPEG"
        ext = "jpg" if fmt == "jpg" else "jpeg"
    elif fmt == "png":
        save_format = "PNG"
    elif fmt == "webp":
        save_format = "WEBP"
    elif fmt == "bmp":
        save_format = "BMP"
    elif fmt == "gif":
        save_format = "GIF"
    elif fmt == "tiff":
        save_format = "TIFF"
    else:
        save_format = fmt.upper()

    filepath = tmp_dir / f"test_image.{ext}"
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    img.save(filepath, format=save_format)

    if oversized:
        current_size = filepath.stat().st_size
        target_size = MAX_IMAGE_SIZE_BYTES + 1024
        if target_size > current_size:
            with open(filepath, "ab") as f:
                f.write(b"\x00" * (target_size - current_size))

    return filepath


def _expected_valid(fmt, width, height, oversized):
    format_ok = fmt in ALLOWED_IMAGE_FORMATS
    width_ok = width >= MIN_RESOLUTION
    height_ok = height >= MIN_RESOLUTION
    size_ok = not oversized
    return format_ok and width_ok and height_ok and size_ok


@pytest.mark.property
class TestImageValidationProperty:
    """Property 2: Image asset validation correctness.

    **Validates: Requirements 2.1, 2.5, 2.6, 2.7**
    """

    @given(data=image_metadata())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_validator_accepts_iff_all_constraints_satisfied(self, data):
        """Validates: Requirements 2.1, 2.5, 2.6, 2.7"""
        fmt, width, height, oversized = data
        validator = AppearanceAssetValidator()

        with tempfile.TemporaryDirectory(prefix="pbt_img_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            filepath = _create_test_image(tmp_path, fmt, width, height, oversized)
            result = validator.validate(str(filepath))
            expected_valid = _expected_valid(fmt, width, height, oversized)

            assert result.is_valid == expected_valid, (
                f"Expected is_valid={expected_valid} but got {result.is_valid} "
                f"for format={fmt}, width={width}, height={height}, oversized={oversized}. "
                f"Errors: {result.errors}"
            )

            if not expected_valid:
                assert len(result.errors) > 0

                if fmt not in ALLOWED_IMAGE_FORMATS:
                    assert any("\u683c\u5f0f" in e for e in result.errors)
                else:
                    if width < MIN_RESOLUTION or height < MIN_RESOLUTION:
                        assert any("\u89e3\u6790\u5ea6" in e for e in result.errors)
                    if oversized:
                        assert any("\u5927\u5c0f" in e for e in result.errors)

    @given(
        fmt=sampled_from(ALLOWED_IMAGE_FORMATS),
        width=integers(min_value=512, max_value=2000),
        height=integers(min_value=512, max_value=2000),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_valid_images_always_accepted(self, fmt, width, height):
        """Validates: Requirements 2.1, 2.5, 2.6, 2.7"""
        validator = AppearanceAssetValidator()

        with tempfile.TemporaryDirectory(prefix="pbt_img_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            filepath = _create_test_image(tmp_path, fmt, width, height, oversized=False)
            result = validator.validate(str(filepath))
            assert result.is_valid is True, f"Got errors: {result.errors}"
            assert result.errors == []


@pytest.mark.property
class TestImageFileSizeBoundary:
    """Boundary tests for the 20MB file size limit.

    **Validates: Requirements 2.7**
    """

    def test_file_just_under_20mb_accepted(self, tmp_path):
        """A file at exactly 20MB should be accepted."""
        validator = AppearanceAssetValidator()
        filepath = tmp_path / "boundary.png"
        img = Image.new("RGB", (512, 512), color=(128, 128, 128))
        img.save(filepath, format="PNG")
        current_size = filepath.stat().st_size
        target_size = MAX_IMAGE_SIZE_BYTES
        if target_size > current_size:
            with open(filepath, "ab") as f:
                f.write(b"\x00" * (target_size - current_size))
        result = validator.validate(str(filepath))
        assert result.is_valid is True

    def test_file_just_over_20mb_rejected(self, tmp_path):
        """A file at 20MB + 1 byte should be rejected."""
        validator = AppearanceAssetValidator()
        filepath = tmp_path / "over_boundary.png"
        img = Image.new("RGB", (512, 512), color=(128, 128, 128))
        img.save(filepath, format="PNG")
        current_size = filepath.stat().st_size
        target_size = MAX_IMAGE_SIZE_BYTES + 1
        if target_size > current_size:
            with open(filepath, "ab") as f:
                f.write(b"\x00" * (target_size - current_size))
        result = validator.validate(str(filepath))
        assert result.is_valid is False
        assert any("\u5927\u5c0f" in e for e in result.errors)

    @given(fmt=sampled_from(ALLOWED_IMAGE_FORMATS))
    @settings(
        max_examples=4,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        deadline=None,
    )
    def test_oversized_files_rejected_for_all_formats(self, fmt, tmp_path):
        """Validates: Requirements 2.7"""
        validator = AppearanceAssetValidator()
        save_format = "JPEG" if fmt in ("jpg", "jpeg") else fmt.upper()
        filepath = tmp_path / f"oversized_{fmt}.{fmt}"
        img = Image.new("RGB", (512, 512), color=(128, 128, 128))
        img.save(filepath, format=save_format)
        current_size = filepath.stat().st_size
        target_size = MAX_IMAGE_SIZE_BYTES + 2048
        with open(filepath, "ab") as f:
            f.write(b"\x00" * (target_size - current_size))
        result = validator.validate(str(filepath))
        assert result.is_valid is False
        assert any("\u5927\u5c0f" in e for e in result.errors)
