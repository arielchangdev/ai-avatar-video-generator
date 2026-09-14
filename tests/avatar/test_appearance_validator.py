"""Unit tests for AppearanceAssetValidator - image validation path."""

import os
from pathlib import Path

import pytest
from PIL import Image

from core.validators.appearance_validator import AppearanceAssetValidator


@pytest.fixture
def validator():
    """Create an AppearanceAssetValidator instance."""
    return AppearanceAssetValidator()


@pytest.fixture
def create_test_image(tmp_path):
    """Factory fixture to create test images with specific properties."""

    def _create(
        *,
        width: int = 1024,
        height: int = 1024,
        format: str = "png",
        filename: str | None = None,
        file_size_override_mb: float | None = None,
    ) -> Path:
        if filename is None:
            filename = f"test_image.{format}"
        filepath = tmp_path / filename

        # Create a real image with PIL
        img = Image.new("RGB", (width, height), color=(128, 128, 128))
        save_format = "JPEG" if format in ("jpg", "jpeg") else format.upper()
        img.save(filepath, format=save_format)

        # If we need to override file size (simulate oversized files)
        if file_size_override_mb is not None:
            target_bytes = int(file_size_override_mb * 1024 * 1024)
            current_size = filepath.stat().st_size
            if target_bytes > current_size:
                with open(filepath, "ab") as f:
                    f.write(b"\x00" * (target_bytes - current_size))

        return filepath

    return _create


class TestImageFormatValidation:
    """Tests for image format acceptance and rejection."""

    @pytest.mark.parametrize("ext", ["jpg", "jpeg", "png", "webp"])
    def test_accepts_valid_image_formats(self, validator, create_test_image, ext):
        """Valid image formats should be accepted."""
        fmt = "jpeg" if ext in ("jpg", "jpeg") else ext
        img_path = create_test_image(format=fmt, filename=f"test.{ext}")
        result = validator.validate(str(img_path))
        assert result.is_valid is True
        assert result.errors == []

    @pytest.mark.parametrize("ext", ["bmp", "gif", "tiff", "svg", "pdf"])
    def test_rejects_invalid_image_formats(self, validator, tmp_path, ext):
        """Unsupported formats should be rejected with error message."""
        filepath = tmp_path / f"test.{ext}"
        filepath.write_bytes(b"\x00" * 1024)
        result = validator.validate(str(filepath))
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "不支援的檔案格式" in result.errors[0]


class TestImageResolutionValidation:
    """Tests for image resolution checks."""

    def test_accepts_exact_minimum_resolution(self, validator, create_test_image):
        """512x512 is the exact minimum and should be accepted."""
        img_path = create_test_image(width=512, height=512)
        result = validator.validate(str(img_path))
        assert result.is_valid is True

    def test_accepts_above_minimum_resolution(self, validator, create_test_image):
        """Resolution above minimum should be accepted."""
        img_path = create_test_image(width=1920, height=1080)
        result = validator.validate(str(img_path))
        assert result.is_valid is True

    def test_rejects_width_below_minimum(self, validator, create_test_image):
        """Width below 512 should be rejected."""
        img_path = create_test_image(width=256, height=1024)
        result = validator.validate(str(img_path))
        assert result.is_valid is False
        assert any("解析度" in e and "256x1024" in e for e in result.errors)

    def test_rejects_height_below_minimum(self, validator, create_test_image):
        """Height below 512 should be rejected."""
        img_path = create_test_image(width=1024, height=256)
        result = validator.validate(str(img_path))
        assert result.is_valid is False
        assert any("解析度" in e and "1024x256" in e for e in result.errors)

    def test_rejects_both_dimensions_below_minimum(self, validator, create_test_image):
        """Both dimensions below minimum should be rejected."""
        img_path = create_test_image(width=100, height=100)
        result = validator.validate(str(img_path))
        assert result.is_valid is False
        assert any("解析度" in e for e in result.errors)


class TestImageFileSizeValidation:
    """Tests for image file size checks."""

    def test_accepts_file_within_size_limit(self, validator, create_test_image):
        """File within 20MB should be accepted."""
        img_path = create_test_image(width=512, height=512)
        result = validator.validate(str(img_path))
        assert result.is_valid is True

    def test_rejects_file_exceeding_size_limit(self, validator, create_test_image):
        """File exceeding 20MB should be rejected."""
        img_path = create_test_image(
            width=512, height=512, file_size_override_mb=21.0
        )
        result = validator.validate(str(img_path))
        assert result.is_valid is False
        assert any("大小" in e and "超過上限" in e for e in result.errors)


class TestCorruptedFileHandling:
    """Tests for graceful handling of corrupted/unreadable files."""

    def test_handles_corrupted_image_file(self, validator, tmp_path):
        """Corrupted image files should return an error, not crash."""
        filepath = tmp_path / "corrupted.png"
        filepath.write_bytes(b"not a valid image at all")
        result = validator.validate(str(filepath))
        assert result.is_valid is False
        assert any("損壞" in e for e in result.errors)

    def test_handles_truncated_image_file(self, validator, tmp_path):
        """Truncated image headers should be handled gracefully."""
        filepath = tmp_path / "truncated.jpg"
        # Write just a JPEG magic number but nothing else
        filepath.write_bytes(b"\xff\xd8\xff\xe0")
        result = validator.validate(str(filepath))
        assert result.is_valid is False
        assert any("損壞" in e for e in result.errors)

    def test_handles_nonexistent_file(self, validator, tmp_path):
        """Non-existent files should return a clear error."""
        filepath = tmp_path / "nonexistent.png"
        result = validator.validate(str(filepath))
        assert result.is_valid is False
        assert any("不存在" in e for e in result.errors)


class TestMultipleViolations:
    """Tests that multiple violations are reported simultaneously."""

    def test_reports_both_size_and_resolution_violations(
        self, validator, create_test_image
    ):
        """Both violations should be listed when file is too large and too small."""
        img_path = create_test_image(
            width=100, height=100, file_size_override_mb=21.0
        )
        result = validator.validate(str(img_path))
        assert result.is_valid is False
        assert len(result.errors) == 2
        assert any("大小" in e for e in result.errors)
        assert any("解析度" in e for e in result.errors)
