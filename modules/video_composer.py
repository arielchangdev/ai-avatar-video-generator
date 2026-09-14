"""AI Avatar Video Generator - Video composition module.

Based on FFmpeg for merging video and audio into final MP4 output,
extracting audio as MP3, and measuring audio-video sync offset.
Handles graceful degradation when FFmpeg is not installed.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from core.exceptions import CompositionTimeoutError, VideoComposeError
from models.schemas import ComposeResult, TaskStage

logger = logging.getLogger(__name__)

# Default composition parameters
DEFAULT_RESOLUTION = (1280, 720)
DEFAULT_FPS = 25
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_AUDIO_CODEC = "aac"
DEFAULT_MP3_BITRATE = "128k"

# Timeout parameters
TIMEOUT_MULTIPLIER = 3
TIMEOUT_MIN_SECONDS = 180


def should_timeout(video_duration_sec: float, elapsed_sec: float) -> bool:
    """Determine if composition should be aborted due to timeout.

    Timeout occurs if and only if:
    - elapsed_sec > video_duration_sec * 3, AND
    - elapsed_sec > 180 seconds

    Both conditions must be true simultaneously.

    Args:
        video_duration_sec: Duration of the source video in seconds.
        elapsed_sec: Time elapsed since composition started in seconds.

    Returns:
        True if composition should be aborted, False otherwise.
    """
    return (
        elapsed_sec > video_duration_sec * TIMEOUT_MULTIPLIER
        and elapsed_sec > TIMEOUT_MIN_SECONDS
    )


class VideoComposer:
    """FFmpeg-based video composition module.

    Merges video and audio streams into a final MP4 file, extracts audio
    as MP3, and measures audio-video synchronization offset.
    """

    def __init__(self) -> None:
        """Initialize VideoComposer and verify FFmpeg availability.

        Raises:
            VideoComposeError: If FFmpeg binary is not found on PATH.
        """
        self._ffmpeg_path = self._find_ffmpeg()
        self._ffprobe_path = self._find_ffprobe()

    def _find_ffmpeg(self) -> str:
        """Locate the ffmpeg binary.

        Returns:
            Path to the ffmpeg executable.

        Raises:
            VideoComposeError: If ffmpeg is not found.
        """
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise VideoComposeError(
                stage=TaskStage.VIDEO_COMPOSE,
                reason=(
                    "FFmpeg is not installed or not found on PATH. "
                    "Install FFmpeg from: https://ffmpeg.org/download.html"
                ),
                recoverable=False,
            )
        logger.info("FFmpeg found at: %s", ffmpeg_path)
        return ffmpeg_path

    def _find_ffprobe(self) -> str:
        """Locate the ffprobe binary.

        Returns:
            Path to the ffprobe executable.

        Raises:
            VideoComposeError: If ffprobe is not found.
        """
        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path is None:
            raise VideoComposeError(
                stage=TaskStage.VIDEO_COMPOSE,
                reason=(
                    "FFprobe is not installed or not found on PATH. "
                    "It is typically bundled with FFmpeg."
                ),
                recoverable=False,
            )
        logger.info("FFprobe found at: %s", ffprobe_path)
        return ffprobe_path

    def compose(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        resolution: tuple[int, int] = DEFAULT_RESOLUTION,
        fps: int = DEFAULT_FPS,
        video_codec: str = DEFAULT_VIDEO_CODEC,
        audio_codec: str = DEFAULT_AUDIO_CODEC,
    ) -> ComposeResult:
        """Merge video and audio streams into a final MP4 file.

        Writes to a temporary file first and moves to output_path on success.
        Preserves intermediate artifacts (input video/audio) on failure.

        Args:
            video_path: Path to the input video file.
            audio_path: Path to the input audio file (WAV).
            output_path: Destination path for the final MP4 file.
            resolution: Output resolution as (width, height). Default (1280, 720).
            fps: Output frame rate. Default 25.
            video_codec: Video codec to use. Default "libx264".
            audio_codec: Audio codec to use. Default "aac".

        Returns:
            ComposeResult with metadata about the composed video.

        Raises:
            VideoComposeError: If input files are missing or composition fails.
            CompositionTimeoutError: If composition exceeds timeout threshold.
        """
        # Validate input files exist
        if not os.path.isfile(video_path):
            raise VideoComposeError(
                stage=TaskStage.VIDEO_COMPOSE,
                reason=f"Input video file not found: {video_path}",
                recoverable=False,
            )
        if not os.path.isfile(audio_path):
            raise VideoComposeError(
                stage=TaskStage.VIDEO_COMPOSE,
                reason=f"Input audio file not found: {audio_path}",
                recoverable=False,
            )

        # Get video duration for timeout calculation
        video_duration = self._get_duration(video_path)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Write to temp file first, move on success
        temp_fd = None
        temp_path = None
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=".mp4", dir=output_dir or None
            )
            os.close(temp_fd)
            temp_fd = None

            # Build FFmpeg command
            width, height = resolution
            cmd = [
                self._ffmpeg_path,
                "-y",  # Overwrite output
                "-i", video_path,
                "-i", audio_path,
                "-c:v", video_codec,
                "-c:a", audio_codec,
                "-vf", f"scale={width}:{height}",
                "-r", str(fps),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                temp_path,
            ]

            logger.info("Starting video composition: %s", " ".join(cmd))
            start_time = time.monotonic()

            # Run FFmpeg with timeout monitoring
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Monitor for timeout
            while process.poll() is None:
                elapsed = time.monotonic() - start_time
                if should_timeout(video_duration, elapsed):
                    process.kill()
                    process.wait()
                    raise CompositionTimeoutError(
                        stage=TaskStage.VIDEO_COMPOSE,
                        reason=(
                            f"Video composition timed out after {elapsed:.1f}s "
                            f"(video duration: {video_duration:.1f}s, "
                            f"threshold: max("
                            f"{video_duration * TIMEOUT_MULTIPLIER:.1f}s, "
                            f"{TIMEOUT_MIN_SECONDS}s))"
                        ),
                        recoverable=True,
                    )
                time.sleep(0.5)

            # Check process result
            returncode = process.returncode
            if returncode != 0:
                stderr_output = (
                    process.stderr.read().decode("utf-8", errors="replace")
                )
                raise VideoComposeError(
                    stage=TaskStage.VIDEO_COMPOSE,
                    reason=(
                        f"FFmpeg composition failed "
                        f"(exit code {returncode}): "
                        f"{stderr_output[:500]}"
                    ),
                    recoverable=True,
                )

            # Move temp file to final output
            shutil.move(temp_path, output_path)
            temp_path = None  # Prevent cleanup of moved file

            # Extract MP3 audio alongside the video
            mp3_output_path = str(Path(output_path).with_suffix(".mp3"))
            self.extract_audio_mp3(audio_path, mp3_output_path)

            # Get output video duration
            output_duration = self._get_duration(output_path)

            # Measure AV sync offset
            av_sync_offset = self.get_av_sync_offset(output_path)

            logger.info(
                "Video composition complete: '%s' (%dx%d, %dfps, %.1fs)",
                output_path,
                width,
                height,
                fps,
                output_duration,
            )

            return ComposeResult(
                video_path=output_path,
                audio_mp3_path=mp3_output_path,
                duration_sec=output_duration,
                resolution=resolution,
                fps=fps,
                av_sync_offset_ms=av_sync_offset,
            )

        except (VideoComposeError, CompositionTimeoutError):
            # Re-raise pipeline errors directly; intermediate files preserved
            raise
        except Exception as e:
            raise VideoComposeError(
                stage=TaskStage.VIDEO_COMPOSE,
                reason=f"Unexpected error during video composition: {e}",
                recoverable=True,
            ) from e
        finally:
            # Clean up temp file if it still exists (composition failed)
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

    def extract_audio_mp3(
        self,
        audio_path: str,
        output_path: str,
        bitrate: str = DEFAULT_MP3_BITRATE,
    ) -> str:
        """Convert audio file (WAV) to MP3 format.

        Args:
            audio_path: Path to the input audio file.
            output_path: Destination path for the MP3 file.
            bitrate: Output bitrate. Default "128k".

        Returns:
            Path to the output MP3 file.

        Raises:
            VideoComposeError: If the input file is missing or conversion fails.
        """
        if not os.path.isfile(audio_path):
            raise VideoComposeError(
                stage=TaskStage.VIDEO_COMPOSE,
                reason=f"Input audio file not found: {audio_path}",
                recoverable=False,
            )

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        cmd = [
            self._ffmpeg_path,
            "-y",  # Overwrite output
            "-i", audio_path,
            "-codec:a", "libmp3lame",
            "-b:a", bitrate,
            output_path,
        ]

        logger.info("Extracting audio to MP3: %s", output_path)

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )

            if result.returncode != 0:
                stderr_output = result.stderr.decode("utf-8", errors="replace")
                raise VideoComposeError(
                    stage=TaskStage.VIDEO_COMPOSE,
                    reason=(
                        f"FFmpeg audio extraction failed "
                        f"(exit code {result.returncode}): "
                        f"{stderr_output[:500]}"
                    ),
                    recoverable=True,
                )

        except subprocess.TimeoutExpired as e:
            raise VideoComposeError(
                stage=TaskStage.VIDEO_COMPOSE,
                reason="Audio extraction to MP3 timed out after 120 seconds.",
                recoverable=True,
            ) from e

        logger.info("MP3 extraction complete: %s", output_path)
        return output_path

    def get_av_sync_offset(self, video_path: str) -> float:
        """Measure audio-video synchronization offset in milliseconds.

        Uses ffprobe to detect the start time difference between the first
        video stream and the first audio stream.

        Args:
            video_path: Path to the video file to analyze.

        Returns:
            Offset in milliseconds. Positive means audio leads video,
            negative means video leads audio. Returns 0.0 if measurement
            cannot be performed.

        Raises:
            VideoComposeError: If the video file is missing.
        """
        if not os.path.isfile(video_path):
            raise VideoComposeError(
                stage=TaskStage.VIDEO_COMPOSE,
                reason=f"Video file not found for sync analysis: {video_path}",
                recoverable=False,
            )

        try:
            # Get video stream start time
            video_start = self._get_stream_start_time(video_path, "video")
            # Get audio stream start time
            audio_start = self._get_stream_start_time(video_path, "audio")

            if video_start is None or audio_start is None:
                logger.warning(
                    "Cannot determine AV sync offset for '%s' - "
                    "missing stream start time information.",
                    video_path,
                )
                return 0.0

            # Offset in milliseconds: positive = audio leads video
            offset_ms = (audio_start - video_start) * 1000.0
            logger.info(
                "AV sync offset for '%s': %.2f ms", video_path, offset_ms
            )
            return offset_ms

        except VideoComposeError:
            raise
        except Exception as e:
            logger.warning(
                "Failed to measure AV sync offset for '%s': %s",
                video_path,
                e,
            )
            return 0.0

    def _get_stream_start_time(
        self, video_path: str, stream_type: str
    ) -> float | None:
        """Get the start time of a specific stream type using ffprobe.

        Args:
            video_path: Path to the media file.
            stream_type: "video" or "audio".

        Returns:
            Start time in seconds, or None if not available.
        """
        cmd = [
            self._ffprobe_path,
            "-v", "error",
            "-select_streams",
            f"{'v' if stream_type == 'video' else 'a'}:0",
            "-show_entries", "stream=start_time",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )

            if result.returncode != 0:
                return None

            output = result.stdout.decode("utf-8").strip()
            if output and output != "N/A":
                return float(output)
            return None

        except (subprocess.TimeoutExpired, ValueError):
            return None

    def _get_duration(self, file_path: str) -> float:
        """Get the duration of a media file using ffprobe.

        Args:
            file_path: Path to the media file.

        Returns:
            Duration in seconds.

        Raises:
            VideoComposeError: If duration cannot be determined.
        """
        cmd = [
            self._ffprobe_path,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )

            if result.returncode != 0:
                stderr_output = result.stderr.decode("utf-8", errors="replace")
                raise VideoComposeError(
                    stage=TaskStage.VIDEO_COMPOSE,
                    reason=(
                        f"Cannot determine duration of '{file_path}': "
                        f"{stderr_output[:200]}"
                    ),
                    recoverable=True,
                )

            output = result.stdout.decode("utf-8").strip()
            if not output or output == "N/A":
                raise VideoComposeError(
                    stage=TaskStage.VIDEO_COMPOSE,
                    reason=(
                        f"No duration information available for '{file_path}'."
                    ),
                    recoverable=True,
                )

            return float(output)

        except subprocess.TimeoutExpired as e:
            raise VideoComposeError(
                stage=TaskStage.VIDEO_COMPOSE,
                reason=f"Timeout while probing duration of '{file_path}'.",
                recoverable=True,
            ) from e
        except ValueError as e:
            raise VideoComposeError(
                stage=TaskStage.VIDEO_COMPOSE,
                reason=(
                    f"Invalid duration value for '{file_path}': {e}"
                ),
                recoverable=True,
            ) from e
