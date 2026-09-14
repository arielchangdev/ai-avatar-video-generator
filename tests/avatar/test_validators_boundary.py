"""Boundary-condition unit tests for all validators.

Tests exact boundary values for:
- VoiceSampleValidator: audio duration at 3s, 300s, 2.99s, 300.1s
- AppearanceAssetValidator: image resolution at 512x512, 511x512, 512x511
- ScriptValidator: text length at 1, 10000, 0, 10001 chars; pure whitespace
- Corrupted file handling for audio, images, and zero-byte files

Validates: Requirements 1.2, 1.3, 2.5, 3.1, 3.4
"""

import struct
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, ".")

from core.validators.voice_validator import VoiceSampleValidator
from core.validators.appearance_validator import AppearanceAssetValidator
from core.validators.script_validator import ScriptValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def voice_validator():
    return VoiceSampleValidator()


@pytest.fixture
def appearance_validator():
    return AppearanceAssetValidator()


@pytest.fixture
def script_validator():
    return ScriptValidator()


def _create_wav(filepath: Path, duration_sec: float, sample_rate: int = 44100) -> Path:
    """Create a minimal valid WAV file with exact duration via struct-based header."""
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
        f.write(b"\x00" * data_size)

    return filepath


def _create_png(filepath: Path, width: int, height: int) -> Path:
    """Create a real PNG image with given dimensions using PIL."""
    img = Image.new("RGB", (width, height), color=(100, 100, 100))
    img.save(filepath, format="PNG")
    return filepath


# ---------------------------------------------------------------------------
# Audio Boundary Tests (VoiceSampleValidator)
# Validates: Requirement 1.2 (min 3s) and 1.3 (max 300s)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAudioBoundary:
    """Boundary tests for voice sample duration limits."""

    def test_audio_exactly_3_seconds_passes(self, voice_validator, tmp_path):
        """Audio at exactly 3 seconds (minimum boundary) should PASS."""
        wav_path = _create_wav(tmp_path / "exact_3s.wav", duration_sec=3.0)
        result = voice_validator.validate(str(wav_path))
        assert result.is_valid is True, f"Expected valid but got errors: {result.errors}"

    def test_audio_exactly_300_seconds_passes(self, voice_validator, tmp_path):
        """Audio at exactly 300 seconds (maximum boundary) should PASS."""
        wav_path = _create_wav(tmp_path / "exact_300s.wav", duration_sec=300.0)
        result = voice_validator.validate(str(wav_path))
        assert result.is_valid is True, f"Expected valid but got errors: {result.errors}"

    def test_audio_below_3_seconds_fails(self, voice_validator, tmp_path):
        """Audio at 2.99 seconds (just below minimum) should FAIL."""
        wav_path = _create_wav(tmp_path / "below_min.wav", duration_sec=2.99)
        result = voice_validator.validate(str(wav_path))
        assert result.is_valid is False
        assert any("時長" in e for e in result.errors)

    def test_audio_above_300_seconds_fails(self, voice_validator, tmp_path):
        """Audio at 300.1 seconds (just above maximum) should FAIL."""
        wav_path = _create_wav(tmp_path / "above_max.wav", duration_sec=300.1)
        result = voice_validator.validate(str(wav_path))
        assert result.is_valid is False
        assert any("時長" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Image Boundary Tests (AppearanceAssetValidator)
# Validates: Requirement 2.5 (min 512x512)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImageBoundary:
    """Boundary tests for image resolution limits."""

    def test_image_exactly_512x512_passes(self, appearance_validator, tmp_path):
        """Image at exact minimum resolution 512x512 should PASS."""
        img_path = _create_png(tmp_path / "exact_512x512.png", 512, 512)
        result = appearance_validator.validate(str(img_path))
        assert result.is_valid is True, f"Expected valid but got errors: {result.errors}"

    def test_image_511x512_fails(self, appearance_validator, tmp_path):
        """Image with width 511 (one pixel below min) should FAIL."""
        img_path = _create_png(tmp_path / "511x512.png", 511, 512)
        result = appearance_validator.validate(str(img_path))
        assert result.is_valid is False
        assert any("解析度" in e and "511x512" in e for e in result.errors)

    def test_image_512x511_fails(self, appearance_validator, tmp_path):
        """Image with height 511 (one pixel below min) should FAIL."""
        img_path = _create_png(tmp_path / "512x511.png", 512, 511)
        result = appearance_validator.validate(str(img_path))
        assert result.is_valid is False
        assert any("解析度" in e and "512x511" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Script Boundary Tests (ScriptValidator)
# Validates: Requirement 3.1 (1-10000 chars) and 3.4 (error with char count)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScriptBoundary:
    """Boundary tests for script text length and content."""

    def test_script_exactly_1_char_passes(self, script_validator):
        """Script with exactly 1 character (minimum boundary) should PASS."""
        result = script_validator.validate("A")
        assert result.is_valid is True

    def test_script_exactly_10000_chars_passes(self, script_validator):
        """Script with exactly 10000 characters (maximum boundary) should PASS."""
        result = script_validator.validate("x" * 10000)
        assert result.is_valid is True

    def test_script_0_chars_fails(self, script_validator):
        """Empty script (0 characters) should FAIL."""
        result = script_validator.validate("")
        assert result.is_valid is False
        assert len(result.errors) >= 1

    def test_script_10001_chars_fails_with_count(self, script_validator):
        """Script with 10001 chars should FAIL with character count in error."""
        result = script_validator.validate("a" * 10001)
        assert result.is_valid is False
        assert any("10001" in e for e in result.errors)

    def test_pure_whitespace_spaces_rejected(self, script_validator):
        """Script with only spaces should FAIL."""
        result = script_validator.validate("       ")
        assert result.is_valid is False
        assert any("空白" in e for e in result.errors)

    def test_pure_whitespace_tabs_rejected(self, script_validator):
        """Script with only tabs should FAIL."""
        result = script_validator.validate("\t\t\t")
        assert result.is_valid is False
        assert any("空白" in e for e in result.errors)

    def test_pure_whitespace_newlines_rejected(self, script_validator):
        """Script with only newlines should FAIL."""
        result = script_validator.validate("\n\n\n\n")
        assert result.is_valid is False
        assert any("空白" in e for e in result.errors)

    def test_pure_whitespace_mixed_rejected(self, script_validator):
        """Script with mixed whitespace (spaces, tabs, newlines) should FAIL."""
        result = script_validator.validate("  \t\n  \t\n  ")
        assert result.is_valid is False
        assert any("空白" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Corrupted File Handling Tests
# Validates: Requirement 1.5 (corrupted audio), 2.5 (corrupted images)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCorruptedFileHandling:
    """Tests that validators handle corrupted/garbage files gracefully."""

    def test_corrupted_wav_file(self, voice_validator, tmp_path):
        """WAV file with valid extension but garbage content should fail gracefully."""
        corrupted = tmp_path / "corrupted.wav"
        corrupted.write_bytes(b"RIFF" + b"\x00" * 4 + b"WAVE" + b"garbage content here!!")
        result = voice_validator.validate(str(corrupted))
        assert result.is_valid is False
        # Should not raise, just return error
        assert len(result.errors) >= 1

    def test_corrupted_png_file(self, appearance_validator, tmp_path):
        """PNG file with valid extension but random bytes should fail gracefully."""
        corrupted = tmp_path / "corrupted.png"
        corrupted.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\xff" * 100)
        result = appearance_validator.validate(str(corrupted))
        assert result.is_valid is False
        assert len(result.errors) >= 1

    def test_zero_byte_wav_file(self, voice_validator, tmp_path):
        """Zero-byte WAV file should fail gracefully."""
        empty = tmp_path / "empty.wav"
        empty.write_bytes(b"")
        result = voice_validator.validate(str(empty))
        assert result.is_valid is False
        assert len(result.errors) >= 1

    def test_zero_byte_png_file(self, appearance_validator, tmp_path):
        """Zero-byte PNG file should fail gracefully."""
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")
        result = appearance_validator.validate(str(empty))
        assert result.is_valid is False
        assert len(result.errors) >= 1

    def test_random_bytes_wav(self, voice_validator, tmp_path):
        """WAV file with completely random bytes should fail gracefully."""
        import os
        random_file = tmp_path / "random.wav"
        random_file.write_bytes(os.urandom(1024))
        result = voice_validator.validate(str(random_file))
        assert result.is_valid is False
        assert len(result.errors) >= 1
