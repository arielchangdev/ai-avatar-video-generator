"""Property-based test for VoiceSampleValidator.

Property 1: Voice sample validation correctness
For any audio file metadata consisting of (format, sample_rate, duration, file_size),
the VoiceSampleValidator SHALL accept the file if and only if:
  format in {wav, mp3, flac} AND sample_rate >= 16000 AND 3 <= duration <= 300 AND file_size <= 50MB.
Otherwise it SHALL reject with an error message listing the specific violation(s).

**Validates: Requirements 1.1, 1.2, 1.3, 1.5**
"""

import sys
import os
import struct
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

import pytest
from hypothesis import given, assume, settings, HealthCheck
from hypothesis.strategies import (
    sampled_from,
    integers,
    floats,
    composite,
)

from core.validators.voice_validator import VoiceSampleValidator


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

VALID_FORMATS = ["wav"]
INVALID_FORMATS = ["ogg", "aac", "m4a"]
ALL_FORMATS = VALID_FORMATS + INVALID_FORMATS

# We focus property testing on WAV format since we can generate real files.
# Non-WAV valid formats (mp3, flac) require encoded files from real encoders.


@composite
def wav_file_params(draw):
    """Generate random WAV file parameters for property testing."""
    sample_rate = draw(integers(min_value=8000, max_value=96000))
    # Duration in seconds - use values that generate manageable file sizes
    # Cap at 10 seconds for test performance (property still holds)
    duration = draw(floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False))
    return sample_rate, duration


def _create_wav_file(tmp_dir, sample_rate, duration_sec, filename="test.wav"):
    """Create a real WAV file with specified parameters."""
    filepath = os.path.join(tmp_dir, filename)
    channels = 1
    bit_depth = 16
    num_samples = int(duration_sec * sample_rate)
    bytes_per_sample = bit_depth // 8
    data_size = num_samples * channels * bytes_per_sample

    with open(filepath, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))  # PCM
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * channels * bytes_per_sample))
        f.write(struct.pack("<H", channels * bytes_per_sample))
        f.write(struct.pack("<H", bit_depth))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        # Write silence
        f.write(b"\x00" * data_size)

    return filepath


def _create_invalid_format_file(tmp_dir, extension):
    """Create a file with an unsupported extension."""
    filepath = os.path.join(tmp_dir, f"test.{extension}")
    with open(filepath, "wb") as f:
        f.write(b"\x00" * 1024)
    return filepath


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------


@pytest.mark.property
class TestVoiceSampleValidationCorrectness:
    """Property 1: Voice sample validation correctness.

    **Validates: Requirements 1.1, 1.2, 1.3, 1.5**
    """

    @given(params=wav_file_params())
    def test_wav_validation_accepts_iff_all_constraints_satisfied(self, params):
        """For any WAV file with (sample_rate, duration), validator accepts iff
        sample_rate >= 16000 AND 3 <= duration <= 300 AND file_size <= 50MB.

        Since we generate small test files, file_size is always <= 50MB.
        """
        sample_rate, duration = params

        with tempfile.TemporaryDirectory(prefix="pbt_voice_") as tmp_dir:
            filepath = _create_wav_file(tmp_dir, sample_rate, duration)

            validator = VoiceSampleValidator()
            result = validator.validate(filepath)

            # Determine expected validity
            format_valid = True  # WAV is always valid format
            rate_valid = sample_rate >= 16000
            duration_valid = 3 <= duration <= 300
            size_valid = True  # test files always < 50MB

            all_valid = format_valid and rate_valid and duration_valid and size_valid

            # Core property: accepts if and only if all constraints satisfied
            if all_valid:
                assert result.is_valid, (
                    f"Expected valid for sample_rate={sample_rate}, duration={duration:.2f}, "
                    f"but got errors: {result.errors}"
                )
                assert result.errors == []
            else:
                assert not result.is_valid, (
                    f"Expected invalid for sample_rate={sample_rate}, duration={duration:.2f}, "
                    f"but validator accepted"
                )
                assert len(result.errors) > 0, "Rejection must list at least one violation"

                if not rate_valid:
                    assert any("取樣率" in e or "sample" in e.lower() or str(sample_rate) in e for e in result.errors), (
                        f"Expected sample rate violation for rate={sample_rate}, got: {result.errors}"
                    )
                if not duration_valid:
                    assert any("時長" in e or "duration" in e.lower() for e in result.errors), (
                        f"Expected duration violation for duration={duration:.2f}, got: {result.errors}"
                    )

    @given(ext=sampled_from(INVALID_FORMATS))
    def test_unsupported_format_always_rejected(self, ext):
        """Files with unsupported extensions are always rejected with format error."""
        with tempfile.TemporaryDirectory(prefix="pbt_voice_") as tmp_dir:
            filepath = _create_invalid_format_file(tmp_dir, ext)

            validator = VoiceSampleValidator()
            result = validator.validate(filepath)

            assert not result.is_valid, f"Expected rejection for .{ext} format"
            assert len(result.errors) > 0
            assert any(
                "格式" in e or "format" in e.lower() or ext in e
                for e in result.errors
            ), f"Expected format-related error for .{ext}, got: {result.errors}"

    @given(
        sample_rate=integers(min_value=16000, max_value=96000),
        duration=floats(min_value=3.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    def test_valid_wav_always_accepted(self, sample_rate, duration):
        """A WAV file meeting all constraints is always accepted."""
        assume(duration >= 3.0)

        with tempfile.TemporaryDirectory(prefix="pbt_voice_") as tmp_dir:
            filepath = _create_wav_file(tmp_dir, sample_rate, duration)

            validator = VoiceSampleValidator()
            result = validator.validate(filepath)

            assert result.is_valid, (
                f"Expected acceptance for valid WAV: rate={sample_rate}, duration={duration:.2f}, "
                f"but got errors: {result.errors}"
            )

    @given(
        sample_rate=integers(min_value=8000, max_value=15999),
        duration=floats(min_value=0.5, max_value=2.99, allow_nan=False, allow_infinity=False),
    )
    def test_multiple_violations_all_reported(self, sample_rate, duration):
        """When multiple constraints are violated, all violations are reported."""
        with tempfile.TemporaryDirectory(prefix="pbt_voice_") as tmp_dir:
            filepath = _create_wav_file(tmp_dir, sample_rate, duration)

            validator = VoiceSampleValidator()
            result = validator.validate(filepath)

            assert not result.is_valid
            assert len(result.errors) >= 2, (
                f"Expected at least 2 errors for rate={sample_rate} and duration={duration:.2f}, "
                f"got {len(result.errors)}: {result.errors}"
            )
