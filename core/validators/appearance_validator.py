"""AI Avatar Video Generator - Appearance asset validator.

Validates user-uploaded appearance assets (images and videos) against
format, resolution, file size, and duration constraints before they enter
the processing pipeline.
"""

import json
import subprocess
from pathlib import Path

from models.schemas import ValidationResult


class AppearanceAssetValidator:
    """外觀素材驗證器

    Validates image and video files for format, resolution, file size,
    and (for videos) duration constraints. Automatically detects whether
    the file is an image or video based on its extension.
    """

    ALLOWED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "webp"]
    ALLOWED_VIDEO_FORMATS = ["mp4", "mov"]
    MIN_RESOLUTION = (512, 512)
    MAX_IMAGE_SIZE_MB = 20
    MAX_VIDEO_SIZE_MB = 200
    MAX_VIDEO_DURATION_SEC = 60

    def validate(self, file_path: str) -> ValidationResult:
        """驗證外觀素材檔案

        Auto-detects file type (image or video) from extension and runs
        the appropriate validation checks.

        Args:
            file_path: Path to the uploaded appearance asset file.

        Returns:
            ValidationResult with is_valid flag and list of error messages.
        """
        path = Path(file_path)

        if not path.exists():
            return ValidationResult(
                is_valid=False,
                errors=["檔案不存在"],
            )

        ext = path.suffix.lstrip(".").lower()

        if ext in self.ALLOWED_IMAGE_FORMATS:
            return self._validate_image(file_path, ext)
        elif ext in self.ALLOWED_VIDEO_FORMATS:
            return self._validate_video(file_path, ext)
        else:
            all_formats = self.ALLOWED_IMAGE_FORMATS + self.ALLOWED_VIDEO_FORMATS
            return ValidationResult(
                is_valid=False,
                errors=[
                    f"不支援的檔案格式 '{ext}'，允許的格式為: "
                    f"{', '.join(all_formats)}"
                ],
            )

    def _validate_image(self, file_path: str, ext: str) -> ValidationResult:
        """驗證圖片素材

        Checks image format, resolution (>= 512x512), and file size (<= 20MB).

        Args:
            file_path: Path to the image file.
            ext: Normalized file extension (without dot).

        Returns:
            ValidationResult with specific violation messages.
        """
        errors: list[str] = []
        path = Path(file_path)

        # File size check
        file_size_bytes = path.stat().st_size
        max_size_bytes = self.MAX_IMAGE_SIZE_MB * 1024 * 1024
        if file_size_bytes > max_size_bytes:
            size_mb = file_size_bytes / (1024 * 1024)
            errors.append(
                f"圖片檔案大小 {size_mb:.1f}MB 超過上限 "
                f"{self.MAX_IMAGE_SIZE_MB}MB"
            )

        # Resolution check using Pillow
        try:
            from PIL import Image

            with Image.open(file_path) as img:
                width, height = img.size
                min_w, min_h = self.MIN_RESOLUTION
                if width < min_w or height < min_h:
                    errors.append(
                        f"圖片解析度 {width}x{height} 低於最低要求 "
                        f"{min_w}x{min_h} 像素"
                    )
        except ImportError:
            # Fallback: try using ffprobe for resolution
            resolution = self._get_resolution_ffprobe(file_path)
            if resolution is None:
                errors.append("無法讀取圖片解析度，檔案可能已損壞")
            else:
                width, height = resolution
                min_w, min_h = self.MIN_RESOLUTION
                if width < min_w or height < min_h:
                    errors.append(
                        f"圖片解析度 {width}x{height} 低於最低要求 "
                        f"{min_w}x{min_h} 像素"
                    )
        except Exception:
            errors.append("無法讀取圖片解析度，檔案可能已損壞")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _validate_video(self, file_path: str, ext: str) -> ValidationResult:
        """驗證影片素材

        Checks video format, resolution (>= 512x512), duration (<= 60 sec),
        and file size (<= 200MB).

        Args:
            file_path: Path to the video file.
            ext: Normalized file extension (without dot).

        Returns:
            ValidationResult with specific violation messages.
        """
        errors: list[str] = []
        path = Path(file_path)

        # File size check
        file_size_bytes = path.stat().st_size
        max_size_bytes = self.MAX_VIDEO_SIZE_MB * 1024 * 1024
        if file_size_bytes > max_size_bytes:
            size_mb = file_size_bytes / (1024 * 1024)
            errors.append(
                f"影片檔案大小 {size_mb:.1f}MB 超過上限 "
                f"{self.MAX_VIDEO_SIZE_MB}MB"
            )

        # Use ffprobe to get resolution and duration
        metadata = self._get_video_metadata(file_path)

        if metadata is None:
            errors.append("無法讀取影片metadata，檔案可能已損壞")
            return ValidationResult(is_valid=False, errors=errors)

        # Resolution check
        width, height, duration = metadata
        min_w, min_h = self.MIN_RESOLUTION
        if width < min_w or height < min_h:
            errors.append(
                f"影片解析度 {width}x{height} 低於最低要求 "
                f"{min_w}x{min_h} 像素"
            )

        # Duration check
        if duration > self.MAX_VIDEO_DURATION_SEC:
            errors.append(
                f"影片時長 {duration:.1f} 秒超過上限 "
                f"{self.MAX_VIDEO_DURATION_SEC} 秒"
            )

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _get_video_metadata(
        self, file_path: str
    ) -> tuple[int, int, float] | None:
        """使用 ffprobe 取得影片的解析度和時長

        Args:
            file_path: Path to the video file.

        Returns:
            Tuple of (width, height, duration_sec) or None if unreadable.
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-show_format",
                file_path,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)

            # Find video stream
            video_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break

            if video_stream is None:
                return None

            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))

            # Get duration from format or stream
            duration = 0.0
            if "duration" in data.get("format", {}):
                duration = float(data["format"]["duration"])
            elif "duration" in video_stream:
                duration = float(video_stream["duration"])

            return (width, height, duration)
        except (
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            ValueError,
            OSError,
        ):
            return None

    def _get_resolution_ffprobe(
        self, file_path: str
    ) -> tuple[int, int] | None:
        """使用 ffprobe 取得圖片/影片的解析度 (fallback)

        Args:
            file_path: Path to the file.

        Returns:
            Tuple of (width, height) or None if unreadable.
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                file_path,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)

            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    width = int(stream.get("width", 0))
                    height = int(stream.get("height", 0))
                    return (width, height)

            return None
        except (
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            ValueError,
            OSError,
        ):
            return None
