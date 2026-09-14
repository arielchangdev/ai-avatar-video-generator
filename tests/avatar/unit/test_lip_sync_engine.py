"""Unit tests for LipSyncEngine - lip sync generation and error handling.

Validates Requirements 5.1, 5.6, 5.7:
- 5.1: Lip sync engine generates video from source image and audio
- 5.6: Error handling for invalid inputs
- 5.7: FPS parameter is respected in output metadata
"""

import sys

sys.path.insert(0, ".")

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from core.exceptions import LipSyncError
from models.schemas import LipSyncResult, TaskStage


class TestInvalidAudioPath:
    """Test that generate raises LipSyncError when audio path does not exist."""

    def test_nonexistent_audio_raises_lip_sync_error(self):
        """A non-existent audio path should raise LipSyncError with 'not found'."""
        from modules.lip_sync_engine import LipSyncEngine

        engine = LipSyncEngine.__new__(LipSyncEngine)
        engine._model = MagicMock()
        engine._micro_expression_config = MagicMock()

        # Create a valid source image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            source_image = f.name

        try:
            with pytest.raises(LipSyncError) as exc_info:
                engine.generate(
                    source_image=source_image,
                    audio_path="/nonexistent/path/audio.wav",
                    output_path="/tmp/output.mp4",
                    fps=25,
                )

            assert "not found" in str(exc_info.value).lower()
            assert exc_info.value.stage == TaskStage.LIP_SYNC
        finally:
            os.unlink(source_image)


class TestInvalidSourceImage:
    """Test that generate raises LipSyncError when source image does not exist."""

    def test_nonexistent_source_image_raises_lip_sync_error(self):
        """A non-existent source image path should raise LipSyncError with 'not found'."""
        from modules.lip_sync_engine import LipSyncEngine

        engine = LipSyncEngine.__new__(LipSyncEngine)
        engine._model = MagicMock()
        engine._micro_expression_config = MagicMock()

        with pytest.raises(LipSyncError) as exc_info:
            engine.generate(
                source_image="/nonexistent/path/image.png",
                audio_path="/some/audio.wav",
                output_path="/tmp/output.mp4",
                fps=25,
            )

        assert "not found" in str(exc_info.value).lower()
        assert exc_info.value.stage == TaskStage.LIP_SYNC


class TestFpsBelowMinimum:
    """Test that generate raises LipSyncError when fps is below MIN_FPS (24)."""

    def test_fps_below_minimum_raises_lip_sync_error(self):
        """FPS below 24 should raise LipSyncError."""
        from modules.lip_sync_engine import LipSyncEngine

        engine = LipSyncEngine.__new__(LipSyncEngine)
        engine._model = MagicMock()
        engine._micro_expression_config = MagicMock()

        with pytest.raises(LipSyncError) as exc_info:
            engine.generate(
                source_image="/any/image.png",
                audio_path="/any/audio.wav",
                output_path="/tmp/output.mp4",
                fps=20,
            )

        assert exc_info.value.stage == TaskStage.LIP_SYNC
        assert exc_info.value.recoverable is True


class TestSadTalkerNotInstalled:
    """Test that __init__ raises LipSyncError when SadTalker is not installed."""

    def test_import_error_raises_lip_sync_error(self):
        """When sadtalker is not importable, LipSyncError should be raised with recoverable=False."""
        from modules.lip_sync_engine import LipSyncEngine

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sadtalker":
                raise ImportError("No module named 'sadtalker'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(LipSyncError) as exc_info:
                LipSyncEngine(checkpoint_dir="/some/dir", device="cpu")

            assert exc_info.value.recoverable is False
            assert exc_info.value.stage == TaskStage.LIP_SYNC
            assert "SadTalker" in str(exc_info.value)


class TestModelNotInitialized:
    """Test that generate raises LipSyncError when model is None."""

    def test_model_none_raises_lip_sync_error(self):
        """When _model is None, generate should raise LipSyncError."""
        from modules.lip_sync_engine import LipSyncEngine

        engine = LipSyncEngine.__new__(LipSyncEngine)
        engine._model = None
        engine._micro_expression_config = MagicMock()

        # Create valid source image and audio files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            source_image = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name

        try:
            with pytest.raises(LipSyncError) as exc_info:
                engine.generate(
                    source_image=source_image,
                    audio_path=audio_path,
                    output_path="/tmp/output.mp4",
                    fps=25,
                )

            assert exc_info.value.stage == TaskStage.LIP_SYNC
            assert exc_info.value.recoverable is False
        finally:
            os.unlink(source_image)
            os.unlink(audio_path)


class TestSuccessfulGeneration:
    """Test that successful generation returns LipSyncResult with correct fps."""

    def test_generate_returns_lip_sync_result_with_correct_fps(self):
        """Successful generation should return LipSyncResult with the requested fps value."""
        from modules.lip_sync_engine import LipSyncEngine, MicroExpressionConfig

        engine = LipSyncEngine.__new__(LipSyncEngine)
        engine._model = MagicMock()
        engine._micro_expression_config = MicroExpressionConfig()

        # Create valid source image and audio files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            source_image = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name

        # Create a temp output directory
        output_dir = tempfile.mkdtemp()
        output_path = os.path.join(output_dir, "result.mp4")

        try:
            # Mock the model's generate to create a temp file (simulating output)
            def fake_generate(**kwargs):
                # Write something to the output path so shutil.move works
                with open(kwargs["output_path"], "w") as f:
                    f.write("fake video data")
                return {}

            engine._model.generate.side_effect = fake_generate

            # Mock duration and resolution helpers
            with patch.object(engine, "_get_video_duration", return_value=5.2):
                with patch.object(engine, "_get_video_resolution", return_value=(1920, 1080)):
                    result = engine.generate(
                        source_image=source_image,
                        audio_path=audio_path,
                        output_path=output_path,
                        fps=30,
                    )

            assert isinstance(result, LipSyncResult)
            assert result.fps == 30.0
            assert result.duration_sec == 5.2
            assert result.resolution == (1920, 1080)
            assert result.video_path == output_path
        finally:
            os.unlink(source_image)
            os.unlink(audio_path)
            # Clean up output
            if os.path.exists(output_path):
                os.unlink(output_path)
            if os.path.exists(output_dir):
                os.rmdir(output_dir)
