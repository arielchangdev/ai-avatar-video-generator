"""Property-based test for SubmissionValidator completeness.

Property 9: Task submission validation completeness
For any combination of the three required inputs where K items (0 <= K <= 3)
are missing or zero-size:
- If K > 0, the submission SHALL be rejected with exactly K items reported
  as missing.
- If K == 0 and all individual validations pass, the submission SHALL be
  accepted with a summary containing the correct duration, asset type, and
  script character count.

**Validates: Requirements 8.1, 8.2, 8.5**
"""

import struct
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis.strategies import booleans, integers, text, composite
from PIL import Image

from core.validators.submission_validator import SubmissionValidator


# ---------------------------------------------------------------------------
# File builders
# ---------------------------------------------------------------------------


def _create_wav(filepath: Path, duration_sec: float, sample_rate: int = 44100) -> Path:
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
        f.write(struct.pack("<H", 1))
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
    Image.new("RGB", (width, height), color=(120, 120, 120)).save(
        filepath, format="PNG"
    )
    return filepath


# ---------------------------------------------------------------------------
# Strategy: which inputs are "bad" (missing/zero-size)
# ---------------------------------------------------------------------------


@composite
def input_presence_combo(draw):
    """Return (voice_ok, appearance_ok, script_ok) booleans.

    A False value means that input is missing or zero-size.
    """
    return (
        draw(booleans()),  # voice present & non-zero
        draw(booleans()),  # appearance present & non-zero
        draw(booleans()),  # script present & non-empty
    )


@pytest.mark.property
class TestSubmissionValidationCompleteness:
    """Property 9: Task submission validation completeness.

    **Validates: Requirements 8.1, 8.2, 8.5**
    """

    @given(combo=input_presence_combo())
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
        ],
    )
    def test_rejection_count_matches_missing_count(self, combo, tmp_path):
        """When K inputs are missing/zero-size, rejection lists exactly K items;
        when K == 0, submission is accepted with a correct summary."""
        voice_ok, appearance_ok, script_ok = combo
        validator = SubmissionValidator()

        # Build voice input
        if voice_ok:
            voice_path = str(_create_wav(tmp_path / "voice.wav", 10.0))
        else:
            # Randomly choose None or zero-byte file
            zero = tmp_path / "voice_empty.wav"
            zero.write_bytes(b"")
            voice_path = str(zero)

        # Build appearance input
        if appearance_ok:
            appearance_path = str(_create_png(tmp_path / "face.png", 512, 512))
        else:
            zero = tmp_path / "face_empty.png"
            zero.write_bytes(b"")
            appearance_path = str(zero)

        # Build script input
        script = "有效的講稿內容" if script_ok else ""

        result = validator.validate(voice_path, appearance_path, script)

        missing_count = sum(1 for ok in combo if not ok)

        if missing_count > 0:
            assert result.is_valid is False
            # Exactly K items reported as missing (presence-stage errors)
            missing_errors = [e for e in result.errors if e.startswith("Missing:")]
            assert len(missing_errors) == missing_count, (
                f"Expected {missing_count} missing items, "
                f"got {missing_errors} for combo={combo}"
            )
            assert result.summary is None
        else:
            # All present + individual validators pass -> accepted with summary
            assert result.is_valid is True, f"Unexpected errors: {result.errors}"
            assert result.summary is not None
            assert result.summary.appearance_type == "image"
            assert result.summary.script_char_count == len(script)
            assert result.summary.voice_duration_sec > 0

    @given(
        num_missing=integers(min_value=1, max_value=3),
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
        ],
    )
    def test_all_none_reports_all_missing(self, num_missing, tmp_path):
        """Passing None for the first `num_missing` inputs reports them as missing."""
        validator = SubmissionValidator()

        voice = str(_create_wav(tmp_path / "v.wav", 10.0))
        appearance = str(_create_png(tmp_path / "f.png", 512, 512))
        script = "有效講稿"

        inputs = [voice, appearance, script]
        for i in range(num_missing):
            inputs[i] = None

        result = validator.validate(inputs[0], inputs[1], inputs[2])
        assert result.is_valid is False
        missing_errors = [e for e in result.errors if e.startswith("Missing:")]
        assert len(missing_errors) == num_missing
