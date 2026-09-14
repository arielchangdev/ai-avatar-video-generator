"""Unit tests for SubmissionValidator.

Validates Requirements 8.1, 8.2, 8.3, 8.4, 8.5:
- 8.1: All three inputs must be present and non-zero size
- 8.2: Missing/invalid items listed in error response
- 8.3: Voice duration must be within range (3-300s)
- 8.4: Script length must be within range (1-10000 chars)
- 8.5: On success, produce summary (duration, asset type, char count)

Covers task 11.3 unit tests:
- Each individual input missing/zero-size
- All inputs valid produces correct summary
- Voice duration out of range blocks submission
- Script length out of range blocks submission
"""

import struct
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pytest
from PIL import Image

from core.validators.submission_validator import SubmissionValidator


# ---------------------------------------------------------------------------
# Helpers - create real valid input files
# ---------------------------------------------------------------------------


def _create_wav(filepath: Path, duration_sec: float, sample_rate: int = 44100) -> Path:
    """Create a minimal valid WAV file with exact duration."""
    channels = 1
    bit_depth = 16
    num_samples = int(duration_sec * sample_rate)
    bytes_per_sample = bit_depth // 8
    data_size = num_samples * channels * bytes_per_sample

    with open(filepath, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))  # PCM
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * channels * bytes_per_sample))
        f.write(struct.pack("<H", channels * bytes_per_sample))
        f.write(struct.pack("<H", bit_depth))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)
    return filepath


def _create_png(filepath: Path, width: int = 512, height: int = 512) -> Path:
    """Create a real PNG image with given dimensions."""
    Image.new("RGB", (width, height), color=(120, 120, 120)).save(
        filepath, format="PNG"
    )
    return filepath


@pytest.fixture
def validator() -> SubmissionValidator:
    return SubmissionValidator()


@pytest.fixture
def valid_inputs(tmp_path):
    """Produce a set of fully valid inputs."""
    voice = _create_wav(tmp_path / "voice.wav", duration_sec=10.0)
    image = _create_png(tmp_path / "face.png", 512, 512)
    script = "這是一段有效的講稿。Hello world."
    return str(voice), str(image), script


# ---------------------------------------------------------------------------
# Presence checks — each individual input missing or zero-size
# ---------------------------------------------------------------------------


class TestPresence:
    def test_all_valid_passes(self, validator, valid_inputs):
        voice, image, script = valid_inputs
        result = validator.validate(voice, image, script)
        assert result.is_valid is True
        assert result.errors == []
        assert result.summary is not None

    def test_missing_voice(self, validator, valid_inputs):
        _, image, script = valid_inputs
        result = validator.validate(None, image, script)
        assert result.is_valid is False
        assert any("voice_sample" in e for e in result.errors)

    def test_missing_appearance(self, validator, valid_inputs):
        voice, _, script = valid_inputs
        result = validator.validate(voice, None, script)
        assert result.is_valid is False
        assert any("appearance_asset" in e for e in result.errors)

    def test_missing_script(self, validator, valid_inputs):
        voice, image, _ = valid_inputs
        result = validator.validate(voice, image, None)
        assert result.is_valid is False
        assert any("script" in e for e in result.errors)

    def test_empty_script_string(self, validator, valid_inputs):
        voice, image, _ = valid_inputs
        result = validator.validate(voice, image, "")
        assert result.is_valid is False
        assert any("script" in e for e in result.errors)

    def test_zero_byte_voice_file(self, validator, valid_inputs, tmp_path):
        _, image, script = valid_inputs
        empty = tmp_path / "empty.wav"
        empty.write_bytes(b"")
        result = validator.validate(str(empty), image, script)
        assert result.is_valid is False
        assert any("voice_sample" in e for e in result.errors)

    def test_zero_byte_appearance_file(self, validator, valid_inputs, tmp_path):
        voice, _, script = valid_inputs
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")
        result = validator.validate(voice, str(empty), script)
        assert result.is_valid is False
        assert any("appearance_asset" in e for e in result.errors)

    def test_all_missing_lists_all_three(self, validator):
        result = validator.validate(None, None, None)
        assert result.is_valid is False
        assert any("voice_sample" in e for e in result.errors)
        assert any("appearance_asset" in e for e in result.errors)
        assert any("script" in e for e in result.errors)
        assert len(result.errors) == 3


# ---------------------------------------------------------------------------
# Summary on success (Req 8.5)
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_has_correct_values(self, validator, tmp_path):
        voice = _create_wav(tmp_path / "voice.wav", duration_sec=12.0)
        image = _create_png(tmp_path / "face.png", 640, 640)
        script = "測試腳本內容"
        result = validator.validate(str(voice), str(image), script)

        assert result.is_valid is True
        assert result.summary is not None
        assert abs(result.summary.voice_duration_sec - 12.0) < 0.5
        assert result.summary.appearance_type == "image"
        assert result.summary.script_char_count == len(script)

    def test_summary_detects_video_asset_type(self, validator, tmp_path):
        # Presence + type only: build a file with mp4 extension.
        # Voice + script valid; appearance video type reported in summary.
        voice = _create_wav(tmp_path / "voice.wav", duration_sec=8.0)
        # A real, small valid mp4 is complex to synthesize here; this test
        # focuses on presence/type mapping which uses the extension.
        # We use a valid image-backed check separately, so only assert type mapping helper.
        assert validator._get_appearance_type("clip.mp4") == "video"
        assert validator._get_appearance_type("clip.mov") == "video"
        assert validator._get_appearance_type("pic.jpg") == "image"


# ---------------------------------------------------------------------------
# Individual validator failures block submission (Req 8.3, 8.4)
# ---------------------------------------------------------------------------


class TestIndividualValidation:
    def test_voice_duration_too_short_blocks(self, validator, tmp_path):
        voice = _create_wav(tmp_path / "short.wav", duration_sec=1.0)  # < 3s
        image = _create_png(tmp_path / "face.png", 512, 512)
        result = validator.validate(str(voice), str(image), "有效腳本")
        assert result.is_valid is False
        assert result.summary is None
        assert any("時長" in e for e in result.errors)

    def test_voice_duration_too_long_blocks(self, validator, tmp_path):
        voice = _create_wav(tmp_path / "long.wav", duration_sec=301.0)  # > 300s
        image = _create_png(tmp_path / "face.png", 512, 512)
        result = validator.validate(str(voice), str(image), "有效腳本")
        assert result.is_valid is False
        assert any("時長" in e for e in result.errors)

    def test_script_too_long_blocks(self, validator, tmp_path):
        voice = _create_wav(tmp_path / "voice.wav", duration_sec=10.0)
        image = _create_png(tmp_path / "face.png", 512, 512)
        long_script = "a" * 10001  # > 10000 chars
        result = validator.validate(str(voice), str(image), long_script)
        assert result.is_valid is False
        assert result.summary is None

    def test_whitespace_only_script_blocks(self, validator, tmp_path):
        voice = _create_wav(tmp_path / "voice.wav", duration_sec=10.0)
        image = _create_png(tmp_path / "face.png", 512, 512)
        # Non-empty (passes presence) but whitespace-only (fails script validator)
        result = validator.validate(str(voice), str(image), "    \t  ")
        assert result.is_valid is False
