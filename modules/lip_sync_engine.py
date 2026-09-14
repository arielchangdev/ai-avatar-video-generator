"""AI Avatar Video Generator - Lip sync engine module.

Based on SadTalker for audio-driven talking face animation.
Handles graceful degradation when SadTalker is not installed.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from core.exceptions import LipSyncError
from models.schemas import LipSyncResult, TaskStage

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Minimum output fps requirement
MIN_FPS = 24

# Micro-expression configuration
BLINK_RATE_MIN = 15  # blinks per minute
BLINK_RATE_MAX = 20  # blinks per minute
HEAD_SWAY_DEGREES = 2.0  # ±2 degrees

# Idle pose configuration
IDLE_SILENCE_THRESHOLD_MS = 500  # silence threshold in milliseconds
IDLE_BREATHING_CYCLE_SEC = 4.0  # one breathing cycle duration


@dataclass
class MicroExpressionConfig:
    """Configuration for micro-expression parameters."""

    blink_rate_min: int = BLINK_RATE_MIN
    blink_rate_max: int = BLINK_RATE_MAX
    head_sway_degrees: float = HEAD_SWAY_DEGREES
    idle_silence_threshold_ms: int = IDLE_SILENCE_THRESHOLD_MS
    idle_breathing_cycle_sec: float = IDLE_BREATHING_CYCLE_SEC


class LipSyncEngine:
    """Lip sync engine based on SadTalker.

    Generates talking face animation from a source image and audio input.
    Includes micro-expression generation (blinks, head sway) and idle pose
    handling for silent segments.
    """

    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        """Load SadTalker model checkpoint weights.

        Args:
            checkpoint_dir: Path to directory containing SadTalker checkpoints.
            device: Computation device ('cuda' or 'cpu').

        Raises:
            LipSyncError: If SadTalker is not installed or checkpoints fail to load.
        """
        self._checkpoint_dir = checkpoint_dir
        self._device = device
        self._model = None
        self._micro_expression_config = MicroExpressionConfig()

        # Try to import SadTalker
        try:
            from sadtalker import SadTalkerInference  # noqa: F401
        except ImportError as e:
            raise LipSyncError(
                stage=TaskStage.LIP_SYNC,
                reason=(
                    "SadTalker is not installed. "
                    "Install from: https://github.com/OpenTalker/SadTalker"
                ),
                recoverable=False,
            ) from e

        # Load model checkpoints
        try:
            from sadtalker import SadTalkerInference

            self._model = SadTalkerInference(
                checkpoint_dir=checkpoint_dir,
                device=device,
            )
            logger.info(
                "SadTalker model loaded from '%s' on device '%s'.",
                checkpoint_dir,
                device,
            )
        except Exception as e:
            raise LipSyncError(
                stage=TaskStage.LIP_SYNC,
                reason=(
                    f"Failed to load SadTalker checkpoints "
                    f"from '{checkpoint_dir}': {e}"
                ),
                recoverable=False,
            ) from e

    @property
    def micro_expression_config(self) -> MicroExpressionConfig:
        """Get the current micro-expression configuration."""
        return self._micro_expression_config

    def generate(
        self,
        source_image: str,
        audio_path: str,
        output_path: str,
        fps: int = 25,
        still_mode: bool = False,
        enhancer: str | None = None,
    ) -> LipSyncResult:
        """Generate lip-synced video from source image and audio.

        The output is written to a temporary file first and moved to
        output_path only on success, ensuring no incomplete file on failure.

        Args:
            source_image: Path to the source face image file.
            audio_path: Path to the driving audio file.
            output_path: Destination path for the generated video.
            fps: Output video frame rate (must be >= 24).
            still_mode: If True, reduce head motion for a more static result.
            enhancer: Optional face enhancer name (e.g., 'gfpgan').

        Returns:
            LipSyncResult with video metadata.

        Raises:
            LipSyncError: If inputs are invalid or generation fails.
        """
        # Validate fps
        if fps < MIN_FPS:
            raise LipSyncError(
                stage=TaskStage.LIP_SYNC,
                reason=(
                    f"FPS must be at least {MIN_FPS}, got {fps}."
                ),
                recoverable=True,
            )

        # Validate source image exists
        if not os.path.isfile(source_image):
            raise LipSyncError(
                stage=TaskStage.LIP_SYNC,
                reason=f"Source image not found: {source_image}",
                recoverable=False,
            )

        # Validate audio path exists
        if not os.path.isfile(audio_path):
            raise LipSyncError(
                stage=TaskStage.LIP_SYNC,
                reason=f"Audio file not found: {audio_path}",
                recoverable=False,
            )

        # Validate model is loaded
        if self._model is None:
            raise LipSyncError(
                stage=TaskStage.LIP_SYNC,
                reason="Lip sync model is not initialized.",
                recoverable=False,
            )

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Write to temp file first, move to output_path on success
        temp_fd = None
        temp_path = None
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=".mp4", dir=output_dir or None
            )
            os.close(temp_fd)
            temp_fd = None

            # Build micro-expression and idle pose parameters
            expression_params = self._build_expression_params(still_mode)

            # Run SadTalker inference
            result_info = self._model.generate(
                source_image=source_image,
                driven_audio=audio_path,
                output_path=temp_path,
                fps=fps,
                still_mode=still_mode,
                enhancer=enhancer,
                **expression_params,
            )

            # Extract result metadata
            duration_sec = self._get_video_duration(temp_path)
            resolution = self._get_video_resolution(temp_path)

            # Move temp file to final output path
            shutil.move(temp_path, output_path)
            temp_path = None  # Prevent cleanup of moved file

            logger.info(
                "Lip sync generation complete: '%s' "
                "(%.1f sec, %d fps, %dx%d)",
                output_path,
                duration_sec,
                fps,
                resolution[0],
                resolution[1],
            )

            return LipSyncResult(
                video_path=output_path,
                fps=float(fps),
                duration_sec=duration_sec,
                resolution=resolution,
            )

        except LipSyncError:
            raise
        except Exception as e:
            raise LipSyncError(
                stage=TaskStage.LIP_SYNC,
                reason=f"Lip sync generation failed: {e}",
                recoverable=True,
            ) from e
        finally:
            # Clean up temp file if it still exists (generation failed)
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except OSError:
                    pass
            if temp_path is not None and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _build_expression_params(self, still_mode: bool) -> dict:
        """Build micro-expression and idle pose parameters for SadTalker.

        Configures:
        - Blink rate: 15-20 blinks per minute
        - Head sway: ±2 degrees
        - Idle pose: breathing animation + random blinks for silence > 500ms

        Args:
            still_mode: If True, reduce head motion parameters.

        Returns:
            Dictionary of expression parameters for SadTalker inference.
        """
        config = self._micro_expression_config

        # Randomize blink rate within configured range
        blink_rate = random.randint(
            config.blink_rate_min, config.blink_rate_max
        )

        # Head sway is reduced in still mode
        head_sway = config.head_sway_degrees if not still_mode else 0.5

        return {
            "blink_rate": blink_rate,
            "head_sway_degrees": head_sway,
            "idle_silence_threshold_ms": config.idle_silence_threshold_ms,
            "idle_breathing_cycle_sec": config.idle_breathing_cycle_sec,
            "idle_random_blinks": True,
        }

    def _get_video_duration(self, video_path: str) -> float:
        """Get the duration of a video file in seconds.

        Uses ffprobe or cv2 as fallback for reading video duration.

        Args:
            video_path: Path to the video file.

        Returns:
            Duration in seconds.
        """
        # Try ffprobe first
        try:
            import subprocess

            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        # Fallback to cv2
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if video_fps > 0:
                    return frame_count / video_fps
        except ImportError:
            pass

        logger.warning(
            "Could not determine video duration for '%s', defaulting to 0.0",
            video_path,
        )
        return 0.0

    def _get_video_resolution(self, video_path: str) -> tuple[int, int]:
        """Get the resolution (width, height) of a video file.

        Uses ffprobe or cv2 as fallback.

        Args:
            video_path: Path to the video file.

        Returns:
            Tuple of (width, height).
        """
        # Try ffprobe first
        try:
            import subprocess

            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=s=x:p=0",
                    video_path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split("x")
                if len(parts) == 2:
                    return (int(parts[0]), int(parts[1]))
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        # Fallback to cv2
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                return (width, height)
        except ImportError:
            pass

        logger.warning(
            "Could not determine video resolution for '%s', defaulting to (0, 0)",
            video_path,
        )
        return (0, 0)
