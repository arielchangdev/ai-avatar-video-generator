"""Property-Based Test: Video asset validation correctness.

Feature: ai-avatar-video-generator, Property 3: Video asset validation correctness

**Validates: Requirements 2.2, 2.6, 2.7, 2.8**

For any video file metadata consisting of (format, width, height, duration, file_size),
the AppearanceAssetValidator SHALL accept the file if and only if:
format in {mp4, mov} AND width >= 512 AND height >= 512 AND duration <= 60 AND
file_size <= 200MB. Otherwise it SHALL reject with an error message listing the
specific violation(s).
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

sys.path.insert(0, ".")

from core.validators.appearance_validator import AppearanceAssetValidator


# --- Strategies ---

# All possible video formats to test (valid + invalid)
video_format_st = st.sampled_from(["mp4", "mov", "avi", "mkv", "wmv", "flv"])

# Resolution dimensions
width_st = st.integers(min_value=100, max_value=4000)
height_st = st.integers(min_value=100, max_value=4000)

# Duration in seconds
duration_st = st.floats(min_value=0.1, max_value=120.0, allow_nan=False, allow_infinity=False)

# File size in MB
file_size_mb_st = st.floats(min_value=0.001, max_value=300.0, allow_nan=False, allow_infinity=False)


# --- Constants matching validator ---

VALID_VIDEO_FORMATS = {"mp4", "mov"}
MIN_RESOLUTION = 512
MAX_DURATION_SEC = 60
MAX_FILE_SIZE_MB = 200


def _should_accept(fmt: str, width: int, height: int, duration: float, file_size_mb: float) -> bool:
    """Determine if the validator should accept based on all constraints."""
    return (
        fmt in VALID_VIDEO_FORMATS
        and width >= MIN_RESOLUTION
        and height >= MIN_RESOLUTION
        and duration <= MAX_DURATION_SEC
        and file_size_mb <= MAX_FILE_SIZE_MB
    )


def _build_ffprobe_response(width: int, height: int, duration: float) -> str:
    """Build a mock ffprobe JSON response."""
    data = {
        "streams": [{"codec_type": "video", "width": width, "height": height}],
        "format": {"duration": str(duration)},
    }
    return json.dumps(data)


# --- Property Tests ---


@pytest.mark.property
class TestVideoAssetValidationProperty:
    """Property 3: Video asset validation correctness.

    **Validates: Requirements 2.2, 2.6, 2.7, 2.8**
    """

    @given(
        fmt=video_format_st,
        width=width_st,
        height=height_st,
        duration=duration_st,
        file_size_mb=file_size_mb_st,
    )
    @settings(max_examples=100, deadline=None)
    def test_validator_accepts_iff_all_constraints_satisfied(
        self,
        fmt: str,
        width: int,
        height: int,
        duration: float,
        file_size_mb: float,
    ) -> None:
        """AppearanceAssetValidator accepts video iff format in {mp4, mov} AND
        width >= 512 AND height >= 512 AND duration <= 60 AND file_size <= 200MB."""
        validator = AppearanceAssetValidator()
        expected_valid = _should_accept(fmt, width, height, duration, file_size_mb)
        file_size_bytes = int(file_size_mb * 1024 * 1024)

        # For invalid formats, the validator rejects at the format check stage
        # (never calls ffprobe), so we only need to mock ffprobe for valid formats.
        if fmt not in VALID_VIDEO_FORMATS:
            # Create a temporary file with the given extension
            with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
                # Write minimal content to make the file exist
                f.write(b"\x00" * 64)
                tmp_path = f.name

            try:
                result = validator.validate(tmp_path)
                assert result.is_valid is False, (
                    f"Expected rejection for unsupported format '{fmt}', "
                    f"but validator accepted"
                )
                assert len(result.errors) >= 1
                assert any("不支援" in e or "格式" in e for e in result.errors), (
                    f"Expected format error message, got: {result.errors}"
                )
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        else:
            # Valid format: create a small temp file and mock file size via Path.stat
            with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
                # Write minimal content - actual size is controlled by mocking stat
                f.write(b"\x00" * 64)
                tmp_path = f.name

            try:
                # Mock subprocess.run to simulate ffprobe response
                mock_run_result = MagicMock()
                mock_run_result.returncode = 0
                mock_run_result.stdout = _build_ffprobe_response(width, height, duration)

                # Mock Path.stat() to return desired file size
                original_stat = Path.stat

                def _mock_stat(self_path, *args, **kwargs):
                    real_stat = original_stat(self_path, *args, **kwargs)
                    if str(self_path) == tmp_path:
                        # Return a mock stat result with our desired file size
                        mock_stat = MagicMock()
                        mock_stat.st_size = file_size_bytes
                        return mock_stat
                    return real_stat

                with patch("subprocess.run", return_value=mock_run_result), \
                     patch.object(Path, "stat", _mock_stat):
                    result = validator.validate(tmp_path)

                assert result.is_valid is expected_valid, (
                    f"Expected is_valid={expected_valid} for "
                    f"fmt={fmt}, width={width}, height={height}, "
                    f"duration={duration:.1f}s, file_size={file_size_mb:.1f}MB. "
                    f"Got is_valid={result.is_valid}, errors={result.errors}"
                )

                if not expected_valid:
                    # Verify specific violations are reported
                    assert len(result.errors) >= 1, (
                        f"Expected at least one error message for invalid input"
                    )
                    self._verify_violation_messages(
                        result.errors, width, height, duration, file_size_mb
                    )
                else:
                    assert result.errors == [], (
                        f"Expected no errors for valid input, got: {result.errors}"
                    )
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    def _verify_violation_messages(
        self,
        errors: list[str],
        width: int,
        height: int,
        duration: float,
        file_size_mb: float,
    ) -> None:
        """Verify that each constraint violation produces a corresponding error."""
        expected_count = 0

        if width < MIN_RESOLUTION or height < MIN_RESOLUTION:
            expected_count += 1
            assert any("解析度" in e for e in errors), (
                f"Expected resolution error for {width}x{height}, got: {errors}"
            )

        if duration > MAX_DURATION_SEC:
            expected_count += 1
            assert any("時長" in e for e in errors), (
                f"Expected duration error for {duration:.1f}s, got: {errors}"
            )

        if file_size_mb > MAX_FILE_SIZE_MB:
            expected_count += 1
            assert any("大小" in e for e in errors), (
                f"Expected file size error for {file_size_mb:.1f}MB, got: {errors}"
            )

        assert len(errors) == expected_count, (
            f"Expected {expected_count} errors but got {len(errors)}: {errors}. "
            f"Constraints: width={width}, height={height}, "
            f"duration={duration:.1f}s, file_size={file_size_mb:.1f}MB"
        )
