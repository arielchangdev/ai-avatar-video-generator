"""Unit tests for VideoComposer - MP3 extraction, compose failure, timeout detection.

Validates Requirements 6.4, 6.7, 6.8:
- 6.4: Extract audio as MP3 alongside composed video
- 6.7: Intermediate files preserved on composition failure
- 6.8: Timeout detection at boundary conditions
"""

import sys

sys.path.insert(0, ".")

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from core.exceptions import VideoComposeError
from models.schemas import TaskStage
from modules.video_composer import should_timeout, VideoComposer


def _make_composer() -> VideoComposer:
    """Create a VideoComposer instance bypassing __init__."""
    composer = VideoComposer.__new__(VideoComposer)
    composer._ffmpeg_path = "/usr/bin/ffmpeg"
    composer._ffprobe_path = "/usr/bin/ffprobe"
    return composer


class TestExtractAudioMp3:
    """Test MP3 extraction produces valid output (Req 6.4)."""

    def test_nonexistent_input_raises_video_compose_error(self):
        """extract_audio_mp3 with non-existent input raises VideoComposeError."""
        composer = _make_composer()

        with pytest.raises(VideoComposeError) as exc_info:
            composer.extract_audio_mp3("/nonexistent/audio.wav", "/tmp/out.mp3")

        assert exc_info.value.stage == TaskStage.VIDEO_COMPOSE
        assert exc_info.value.recoverable is False
        assert "not found" in str(exc_info.value).lower()

    @patch("subprocess.run")
    def test_valid_input_calls_ffmpeg_correctly(self, mock_run):
        """extract_audio_mp3 with valid input calls ffmpeg with correct arguments."""
        composer = _make_composer()

        mock_run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_wav = os.path.join(tmpdir, "input.wav")
            output_mp3 = os.path.join(tmpdir, "output.mp3")

            # Create a fake input file
            with open(input_wav, "wb") as f:
                f.write(b"\x00" * 100)

            result = composer.extract_audio_mp3(input_wav, output_mp3)

            assert result == output_mp3
            mock_run.assert_called_once()

            call_args = mock_run.call_args
            cmd = call_args[0][0]

            assert cmd[0] == "/usr/bin/ffmpeg"
            assert "-y" in cmd
            assert "-i" in cmd
            assert input_wav in cmd
            assert "libmp3lame" in cmd
            assert "128k" in cmd
            assert output_mp3 in cmd


class TestComposeFailurePreservesFiles:
    """Test compose failure preserves intermediate files (Req 6.7)."""

    def test_nonexistent_video_raises_video_compose_error(self):
        """compose with non-existent video raises VideoComposeError."""
        composer = _make_composer()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.wav")
            with open(audio_path, "wb") as f:
                f.write(b"\x00" * 100)

            with pytest.raises(VideoComposeError) as exc_info:
                composer.compose(
                    video_path="/nonexistent/video.mp4",
                    audio_path=audio_path,
                    output_path=os.path.join(tmpdir, "output.mp4"),
                )

            assert exc_info.value.stage == TaskStage.VIDEO_COMPOSE
            assert "not found" in str(exc_info.value).lower()

    def test_nonexistent_audio_raises_video_compose_error(self):
        """compose with non-existent audio raises VideoComposeError."""
        composer = _make_composer()

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")
            with open(video_path, "wb") as f:
                f.write(b"\x00" * 100)

            with pytest.raises(VideoComposeError) as exc_info:
                composer.compose(
                    video_path=video_path,
                    audio_path="/nonexistent/audio.wav",
                    output_path=os.path.join(tmpdir, "output.mp4"),
                )

            assert exc_info.value.stage == TaskStage.VIDEO_COMPOSE
            assert "not found" in str(exc_info.value).lower()

    @patch("subprocess.Popen")
    def test_ffmpeg_failure_preserves_intermediate_input_files(self, mock_popen):
        """When ffmpeg returns non-zero, input video and audio files are preserved."""
        composer = _make_composer()

        # Mock _get_duration to return a value without calling ffprobe
        composer._get_duration = MagicMock(return_value=10.0)

        # Mock Popen to simulate ffmpeg failure
        mock_process = MagicMock()
        mock_process.poll.side_effect = [None, 0]  # First poll: None, second: 0 (done)
        mock_process.returncode = 1  # Non-zero exit code
        mock_process.stderr = MagicMock()
        mock_process.stderr.read.return_value = b"ffmpeg error: encoding failed"
        mock_popen.return_value = mock_process

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input_video.mp4")
            audio_path = os.path.join(tmpdir, "input_audio.wav")
            output_path = os.path.join(tmpdir, "output.mp4")

            # Create input files
            with open(video_path, "wb") as f:
                f.write(b"\x00" * 200)
            with open(audio_path, "wb") as f:
                f.write(b"\x00" * 200)

            with pytest.raises(VideoComposeError):
                composer.compose(
                    video_path=video_path,
                    audio_path=audio_path,
                    output_path=output_path,
                )

            # Verify intermediate input files are preserved
            assert os.path.exists(video_path), "Input video file should be preserved"
            assert os.path.exists(audio_path), "Input audio file should be preserved"


class TestShouldTimeout:
    """Test timeout detection at boundary conditions (Req 6.8)."""

    def test_elapsed_equals_180_duration_60_no_timeout(self):
        """elapsed=180.0, duration=60 -> False (180 not > 180, strict inequality)."""
        result = should_timeout(video_duration_sec=60.0, elapsed_sec=180.0)
        assert result is False

    def test_elapsed_180_1_duration_59_timeout(self):
        """elapsed=180.1, duration=59 -> True (180.1 > 59*3=177 AND 180.1 > 180)."""
        result = should_timeout(video_duration_sec=59.0, elapsed_sec=180.1)
        assert result is True

    def test_elapsed_200_duration_100_no_timeout(self):
        """elapsed=200, duration=100 -> False (200 not > 100*3=300)."""
        result = should_timeout(video_duration_sec=100.0, elapsed_sec=200.0)
        assert result is False

    def test_both_conditions_must_hold(self):
        """Timeout requires BOTH elapsed > duration*3 AND elapsed > 180."""
        # elapsed > duration*3 but NOT > 180
        assert should_timeout(video_duration_sec=10.0, elapsed_sec=50.0) is False

        # elapsed > 180 but NOT > duration*3
        assert should_timeout(video_duration_sec=100.0, elapsed_sec=181.0) is False

        # Both conditions true
        assert should_timeout(video_duration_sec=50.0, elapsed_sec=181.0) is True


class TestVideoComposerInitFailure:
    """Test that __init__ raises VideoComposeError when ffmpeg not found (Req 6.8)."""

    @patch("shutil.which")
    def test_ffmpeg_not_found_raises_video_compose_error(self, mock_which):
        """When ffmpeg is not on PATH, VideoComposeError is raised with recoverable=False."""
        mock_which.return_value = None

        with pytest.raises(VideoComposeError) as exc_info:
            VideoComposer()

        assert exc_info.value.recoverable is False
        assert exc_info.value.stage == TaskStage.VIDEO_COMPOSE
        assert "ffmpeg" in str(exc_info.value).lower()
