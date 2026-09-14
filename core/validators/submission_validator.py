"""AI Avatar Video Generator - Submission validator.

Validates a complete task submission by checking that all three required inputs
(voice sample, appearance asset, script) are present with non-zero size,
then running individual validators and producing a summary on success.
"""

import os
import wave
from pathlib import Path

from pydantic import BaseModel

from core.validators.appearance_validator import AppearanceAssetValidator
from core.validators.script_validator import ScriptValidator
from core.validators.voice_validator import VoiceSampleValidator
from models.schemas import ValidationResult


class SubmissionSummary(BaseModel):
    """素材摘要 - 所有輸入通過驗證後產生的摘要資訊"""

    voice_duration_sec: float
    appearance_type: str  # "image" or "video"
    script_char_count: int


class SubmissionResult(BaseModel):
    """提交驗證結果"""

    is_valid: bool
    errors: list[str]
    summary: SubmissionSummary | None = None  # Only populated on success


class SubmissionValidator:
    """提交驗證器 - 驗證完整任務提交的所有必要輸入素材。

    Validates that all three required inputs are present and non-zero size,
    runs individual validators, and produces a summary on success.
    """

    IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    VIDEO_EXTENSIONS = {"mp4", "mov"}

    def __init__(self) -> None:
        self._voice_validator = VoiceSampleValidator()
        self._appearance_validator = AppearanceAssetValidator()
        self._script_validator = ScriptValidator()

    def validate(
        self,
        voice_sample_path: str | None,
        appearance_asset_path: str | None,
        script_text: str | None,
    ) -> SubmissionResult:
        """驗證完整任務提交。

        Logic:
        1. Check presence: each input must be provided, and files must exist
           with size > 0.
        2. If any missing/zero-size, return early with errors listing each
           missing item.
        3. If all present, run individual validators and collect all errors.
        4. If all pass, compute and return summary.

        Args:
            voice_sample_path: Path to the voice sample file, or None.
            appearance_asset_path: Path to the appearance asset file, or None.
            script_text: Script text content, or None.

        Returns:
            SubmissionResult with validation outcome and optional summary.
        """
        # Step 1: Check presence and non-zero size
        missing_errors = self._check_presence(
            voice_sample_path, appearance_asset_path, script_text
        )

        if missing_errors:
            return SubmissionResult(is_valid=False, errors=missing_errors)

        # Step 2: Run individual validators
        # At this point we know all inputs are present and non-zero
        assert voice_sample_path is not None
        assert appearance_asset_path is not None
        assert script_text is not None

        all_errors: list[str] = []

        voice_result = self._voice_validator.validate(voice_sample_path)
        if not voice_result.is_valid:
            all_errors.extend(voice_result.errors)

        appearance_result = self._appearance_validator.validate(
            appearance_asset_path
        )
        if not appearance_result.is_valid:
            all_errors.extend(appearance_result.errors)

        script_result = self._script_validator.validate(script_text)
        if not script_result.is_valid:
            all_errors.extend(script_result.errors)

        if all_errors:
            return SubmissionResult(is_valid=False, errors=all_errors)

        # Step 3: All validations passed - compute summary
        summary = self._compute_summary(
            voice_sample_path, appearance_asset_path, script_text
        )

        return SubmissionResult(
            is_valid=True, errors=[], summary=summary
        )

    def _check_presence(
        self,
        voice_sample_path: str | None,
        appearance_asset_path: str | None,
        script_text: str | None,
    ) -> list[str]:
        """檢查所有必要輸入是否存在且大小大於 0。

        Returns:
            List of error messages for missing/invalid items.
        """
        errors: list[str] = []

        # Voice sample: must be provided, file must exist, size > 0
        if not voice_sample_path or not self._file_exists_nonzero(
            voice_sample_path
        ):
            errors.append("Missing: voice_sample")

        # Appearance asset: must be provided, file must exist, size > 0
        if not appearance_asset_path or not self._file_exists_nonzero(
            appearance_asset_path
        ):
            errors.append("Missing: appearance_asset")

        # Script: must be provided and not empty
        if script_text is None or len(script_text) == 0:
            errors.append("Missing: script")

        return errors

    def _file_exists_nonzero(self, file_path: str) -> bool:
        """檢查檔案是否存在且大小大於 0 bytes."""
        try:
            path = Path(file_path)
            return path.exists() and path.stat().st_size > 0
        except (OSError, ValueError):
            return False

    def _compute_summary(
        self,
        voice_sample_path: str,
        appearance_asset_path: str,
        script_text: str,
    ) -> SubmissionSummary:
        """計算素材摘要。

        Args:
            voice_sample_path: Validated voice sample file path.
            appearance_asset_path: Validated appearance asset file path.
            script_text: Validated script text.

        Returns:
            SubmissionSummary with duration, asset type, and char count.
        """
        voice_duration = self._get_voice_duration(voice_sample_path)
        appearance_type = self._get_appearance_type(appearance_asset_path)
        script_char_count = len(script_text)

        return SubmissionSummary(
            voice_duration_sec=voice_duration,
            appearance_type=appearance_type,
            script_char_count=script_char_count,
        )

    def _get_voice_duration(self, file_path: str) -> float:
        """取得聲音檔案的時長（秒）。

        Supports WAV (via wave module), MP3/FLAC (via mutagen).
        """
        path = Path(file_path)
        ext = path.suffix.lower().lstrip(".")

        if ext == "wav":
            with wave.open(file_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / rate
        elif ext == "mp3":
            from mutagen.mp3 import MP3

            audio = MP3(file_path)
            return audio.info.length
        elif ext == "flac":
            from mutagen.flac import FLAC

            audio = FLAC(file_path)
            return audio.info.length
        else:
            # Fallback: try mutagen generic
            from mutagen import File as MutagenFile

            audio = MutagenFile(file_path)
            if audio and audio.info:
                return audio.info.length
            return 0.0

    def _get_appearance_type(self, file_path: str) -> str:
        """判斷外觀素材類型（image 或 video）。

        Based on file extension.
        """
        ext = Path(file_path).suffix.lower().lstrip(".")

        if ext in self.IMAGE_EXTENSIONS:
            return "image"
        elif ext in self.VIDEO_EXTENSIONS:
            return "video"
        else:
            # Shouldn't reach here after validation, but safe fallback
            return "unknown"
