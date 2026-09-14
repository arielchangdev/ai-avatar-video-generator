"""Unit tests for VoiceCloner - synthesis failure, silence insertion, sample rejection.

Validates Requirements 4.4, 4.6, 4.7:
- 4.4: Output audio is 44100Hz 16-bit WAV
- 4.6: Insert 0.5-1.5 second silence at paragraph breaks
- 4.7: No incomplete audio file written on synthesis failure
"""

import sys

sys.path.insert(0, ".")

import os
import tempfile
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.exceptions import VoiceSynthesisError
from models.schemas import (
    LanguageSegment,
    LanguageType,
    ScriptResult,
    TaskStage,
    VoiceProfile,
)
from modules.voice_cloner import (
    DEFAULT_SILENCE_DURATION_SEC,
    MAX_SILENCE_DURATION_SEC,
    MIN_SAMPLE_DURATION_SEC,
    MIN_SILENCE_DURATION_SEC,
    TARGET_SAMPLE_RATE,
    VoiceCloner,
)


def _make_voice_profile(sample_path: str = "/tmp/sample.wav") -> VoiceProfile:
    """Helper to create a VoiceProfile for testing."""
    return VoiceProfile(
        sample_path=sample_path,
        duration_sec=5.0,
        sample_rate=44100,
        embedding=[0.1] * 256,
    )


def _make_script_result(
    texts: list[str] | None = None,
    paragraph_breaks: list[int] | None = None,
) -> ScriptResult:
    """Helper to create a ScriptResult for testing."""
    if texts is None:
        texts = ["你好世界", "Hello world"]
    segments = []
    idx = 0
    for text in texts:
        lang = LanguageType.CHINESE if any("\u4e00" <= c <= "\u9fff" for c in text) else LanguageType.ENGLISH
        segments.append(
            LanguageSegment(
                text=text,
                language=lang,
                start_index=idx,
                end_index=idx + len(text),
            )
        )
        idx += len(text)

    return ScriptResult(
        segments=segments,
        paragraph_breaks=paragraph_breaks if paragraph_breaks is not None else [],
        total_chars=idx,
    )


def _create_wav_file(path: str, duration_sec: float, sample_rate: int = 44100) -> None:
    """Create a minimal valid WAV file with specified duration."""
    num_samples = int(sample_rate * duration_sec)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)


def _make_cloner() -> VoiceCloner:
    """Create a VoiceCloner instance bypassing __init__."""
    cloner = VoiceCloner.__new__(VoiceCloner)
    cloner._model_path = "/fake/model/path"
    cloner._device = "cpu"
    cloner._model = MagicMock()
    return cloner


class TestSynthesisFailureNoOutputFile:
    """Test that synthesis failure does not produce an output file (Req 4.7)."""

    def test_model_exception_leaves_no_output(self):
        """When the model raises an exception during synthesis, no file at output_path."""
        cloner = _make_cloner()
        cloner._model.synthesize.side_effect = RuntimeError("GPU OOM")

        script = _make_script_result(texts=["測試語音"])
        profile = _make_voice_profile()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.wav")

            with pytest.raises(VoiceSynthesisError):
                cloner.synthesize(script, profile, output_path)

            assert not os.path.exists(output_path)

    def test_synthesis_error_cleans_temp_file(self):
        """On failure, temp files are cleaned up and output_path has no file."""
        cloner = _make_cloner()
        cloner._model.synthesize.side_effect = Exception("Synthesis crashed")

        script = _make_script_result(texts=["失敗測試"])
        profile = _make_voice_profile()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "result.wav")

            with pytest.raises(VoiceSynthesisError):
                cloner.synthesize(script, profile, output_path)

            remaining = os.listdir(tmpdir)
            assert len(remaining) == 0


class TestParagraphSilenceInsertion:
    """Test that silence is inserted at paragraph break positions (Req 4.6)."""

    def test_silence_inserted_at_paragraph_break(self):
        """Segments at paragraph break positions get silence prepended."""
        cloner = _make_cloner()

        script = _make_script_result(
            texts=["你好世界", "第二段落"],
            paragraph_breaks=[4],  # Break before seg1 (start_index=4)
        )
        profile = _make_voice_profile()

        fake_audio = np.zeros(1000, dtype=np.int16)
        cloner._model.synthesize.return_value = fake_audio

        result_audio = cloner._synthesize_segments(script, profile)

        expected_silence_samples = int(TARGET_SAMPLE_RATE * DEFAULT_SILENCE_DURATION_SEC)
        # seg0(1000) + silence(44100) + seg1(1000)
        expected_total = 1000 + expected_silence_samples + 1000
        assert len(result_audio) == expected_total

    def test_no_silence_without_paragraph_breaks(self):
        """When no paragraph breaks, no silence is inserted."""
        cloner = _make_cloner()

        script = _make_script_result(
            texts=["連續文字", "不分段"],
            paragraph_breaks=[],
        )
        profile = _make_voice_profile()

        fake_audio = np.zeros(500, dtype=np.int16)
        cloner._model.synthesize.return_value = fake_audio

        result_audio = cloner._synthesize_segments(script, profile)

        assert len(result_audio) == 1000

    def test_multiple_paragraph_breaks(self):
        """Multiple paragraph breaks insert silence at each break position."""
        cloner = _make_cloner()

        segments = [
            LanguageSegment(text="AB", language=LanguageType.ENGLISH, start_index=0, end_index=2),
            LanguageSegment(text="CDE", language=LanguageType.ENGLISH, start_index=2, end_index=5),
            LanguageSegment(text="FG", language=LanguageType.ENGLISH, start_index=5, end_index=7),
        ]
        script = ScriptResult(segments=segments, paragraph_breaks=[2, 5], total_chars=7)
        profile = _make_voice_profile()

        fake_audio = np.ones(100, dtype=np.int16)
        cloner._model.synthesize.return_value = fake_audio

        result_audio = cloner._synthesize_segments(script, profile)

        silence_samples = int(TARGET_SAMPLE_RATE * DEFAULT_SILENCE_DURATION_SEC)
        # seg0(100) + silence + seg1(100) + silence + seg2(100)
        expected = 100 + silence_samples + 100 + silence_samples + 100
        assert len(result_audio) == expected


class TestVoiceSampleRejectionInsufficientDuration:
    """Test that voice samples shorter than 3 seconds are rejected (Req 4.4)."""

    def test_short_sample_raises_error(self):
        """Audio shorter than MIN_SAMPLE_DURATION_SEC (3s) raises VoiceSynthesisError."""
        cloner = _make_cloner()

        with tempfile.TemporaryDirectory() as tmpdir:
            short_wav = os.path.join(tmpdir, "short.wav")
            _create_wav_file(short_wav, duration_sec=2.0)

            with pytest.raises(VoiceSynthesisError) as exc_info:
                cloner.analyze_voice_sample(short_wav)

            assert exc_info.value.stage == TaskStage.VOICE_ANALYSIS
            assert "minimum" in str(exc_info.value).lower() or "2.0" in str(exc_info.value)
            assert exc_info.value.recoverable is True

    def test_exactly_below_minimum_raises_error(self):
        """Audio at 2.9s (below 3s threshold) is rejected."""
        cloner = _make_cloner()

        with tempfile.TemporaryDirectory() as tmpdir:
            short_wav = os.path.join(tmpdir, "borderline.wav")
            _create_wav_file(short_wav, duration_sec=2.9)

            with pytest.raises(VoiceSynthesisError) as exc_info:
                cloner.analyze_voice_sample(short_wav)

            assert exc_info.value.stage == TaskStage.VOICE_ANALYSIS

    def test_at_minimum_accepted(self):
        """Audio at exactly 3.0s should pass duration check."""
        cloner = _make_cloner()
        cloner._model.extract_speaker_embedding.return_value = np.array([0.1] * 256)

        with tempfile.TemporaryDirectory() as tmpdir:
            valid_wav = os.path.join(tmpdir, "valid.wav")
            _create_wav_file(valid_wav, duration_sec=3.0)

            result = cloner.analyze_voice_sample(valid_wav)
            assert result.duration_sec >= MIN_SAMPLE_DURATION_SEC


class TestVoiceClonerInitFailure:
    """Test that VoiceCloner raises VoiceSynthesisError when GPT-SoVITS not installed."""

    def test_import_error_raises_voice_synthesis_error(self):
        """When GPT-SoVITS import fails, VoiceSynthesisError is raised."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "GPT_SoVITS" in name:
                raise ImportError("No module named 'GPT_SoVITS'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(VoiceSynthesisError) as exc_info:
                VoiceCloner(model_path="/fake/path", device="cpu")

            assert exc_info.value.recoverable is False
            assert "GPT-SoVITS" in str(exc_info.value)


class TestAnalyzeVoiceSampleNonExistentFile:
    """Test that analyze_voice_sample raises error for non-existent file."""

    def test_nonexistent_file_raises_error(self):
        """A path that doesn't exist should raise VoiceSynthesisError."""
        cloner = _make_cloner()

        with pytest.raises(VoiceSynthesisError) as exc_info:
            cloner.analyze_voice_sample("/nonexistent/path/audio.wav")

        assert exc_info.value.stage == TaskStage.VOICE_ANALYSIS
        assert exc_info.value.recoverable is False
        assert "not found" in str(exc_info.value).lower()


class TestGenerateSilenceClamping:
    """Test that _generate_silence clamps duration to 0.5-1.5 second range."""

    def test_below_minimum_clamped_to_min(self):
        """Duration below 0.5s is clamped to MIN_SILENCE_DURATION_SEC."""
        cloner = _make_cloner()

        silence = cloner._generate_silence(0.1)
        expected_samples = int(TARGET_SAMPLE_RATE * MIN_SILENCE_DURATION_SEC)
        assert len(silence) == expected_samples

    def test_above_maximum_clamped_to_max(self):
        """Duration above 1.5s is clamped to MAX_SILENCE_DURATION_SEC."""
        cloner = _make_cloner()

        silence = cloner._generate_silence(5.0)
        expected_samples = int(TARGET_SAMPLE_RATE * MAX_SILENCE_DURATION_SEC)
        assert len(silence) == expected_samples

    def test_within_range_not_clamped(self):
        """Duration within 0.5-1.5s is used as-is."""
        cloner = _make_cloner()

        silence = cloner._generate_silence(1.0)
        expected_samples = int(TARGET_SAMPLE_RATE * 1.0)
        assert len(silence) == expected_samples

    def test_at_boundaries(self):
        """Test exact boundary values 0.5 and 1.5."""
        cloner = _make_cloner()

        silence_min = cloner._generate_silence(0.5)
        assert len(silence_min) == int(TARGET_SAMPLE_RATE * 0.5)

        silence_max = cloner._generate_silence(1.5)
        assert len(silence_max) == int(TARGET_SAMPLE_RATE * 1.5)

    def test_silence_is_zeros(self):
        """Generated silence samples should all be zero."""
        cloner = _make_cloner()

        silence = cloner._generate_silence(1.0)
        assert np.all(silence == 0)
        assert silence.dtype == np.int16
