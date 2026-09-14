"""AI Avatar Video Generator - 聲音樣本驗證器.

Validates voice sample files for format, sample rate, duration, and file size
before they enter the processing pipeline.
"""

import os
import wave
from pathlib import Path

from mutagen import File as MutagenFile
from mutagen.mp3 import MP3
from mutagen.flac import FLAC

from models.schemas import ValidationResult


class VoiceSampleValidator:
    """聲音樣本驗證器 - 驗證上傳的音訊檔案是否符合系統需求.

    Validates:
        - Format: WAV, MP3, FLAC
        - Sample rate: >= 16000 Hz
        - Duration: 3 ~ 300 seconds (inclusive)
        - File size: <= 50 MB
    """

    ALLOWED_FORMATS = ["wav", "mp3", "flac"]
    MIN_DURATION_SEC = 3
    MAX_DURATION_SEC = 300
    MAX_FILE_SIZE_MB = 50
    MIN_SAMPLE_RATE_HZ = 16000

    # Magic bytes for supported formats
    _MAGIC_BYTES = {
        "wav": b"RIFF",
        "mp3": [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"ID3"],
        "flac": b"fLaC",
    }

    def validate(self, file_path: str) -> ValidationResult:
        """驗證聲音樣本檔案.

        Args:
            file_path: 音訊檔案的完整路徑

        Returns:
            ValidationResult with is_valid flag and any error messages
        """
        errors: list[str] = []

        # Check file exists
        path = Path(file_path)
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                errors=[f"檔案不存在: {file_path}"],
            )

        # Check file size
        file_size_bytes = os.path.getsize(file_path)
        max_size_bytes = self.MAX_FILE_SIZE_MB * 1024 * 1024
        if file_size_bytes > max_size_bytes:
            errors.append(
                f"檔案大小 ({file_size_bytes / (1024 * 1024):.1f}MB) 超過上限 {self.MAX_FILE_SIZE_MB}MB"
            )

        # Check format by extension
        extension = path.suffix.lower().lstrip(".")
        if extension not in self.ALLOWED_FORMATS:
            errors.append(
                f"不支援的檔案格式: .{extension}，支援格式為: {', '.join(self.ALLOWED_FORMATS)}"
            )
            # Can't proceed with audio analysis if format is unsupported
            return ValidationResult(is_valid=False, errors=errors)

        # Verify magic bytes match extension
        if not self._verify_magic_bytes(file_path, extension):
            errors.append(
                f"檔案內容與副檔名 (.{extension}) 不符，檔案可能已損壞或格式錯誤"
            )
            return ValidationResult(is_valid=False, errors=errors)

        # Read audio metadata (sample rate, duration)
        try:
            sample_rate, duration = self._read_audio_metadata(file_path, extension)
        except Exception:
            errors.append("無法解碼音訊檔案，檔案可能已損壞")
            return ValidationResult(is_valid=False, errors=errors)

        # Check sample rate
        if sample_rate < self.MIN_SAMPLE_RATE_HZ:
            errors.append(
                f"取樣率 ({sample_rate}Hz) 低於最低要求 {self.MIN_SAMPLE_RATE_HZ}Hz"
            )

        # Check duration
        if duration < self.MIN_DURATION_SEC:
            errors.append(
                f"音訊時長 ({duration:.1f}秒) 少於最低要求 {self.MIN_DURATION_SEC}秒"
            )
        elif duration > self.MAX_DURATION_SEC:
            errors.append(
                f"音訊時長 ({duration:.1f}秒) 超過上限 {self.MAX_DURATION_SEC}秒"
            )

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _verify_magic_bytes(self, file_path: str, extension: str) -> bool:
        """驗證檔案的 magic bytes 是否與宣稱的格式一致."""
        try:
            with open(file_path, "rb") as f:
                header = f.read(12)
        except (OSError, IOError):
            return False

        if len(header) < 4:
            return False

        magic = self._MAGIC_BYTES.get(extension)
        if magic is None:
            return False

        if isinstance(magic, list):
            # MP3 can have multiple magic byte patterns
            return any(header.startswith(m) for m in magic)
        else:
            if extension == "wav":
                # WAV: starts with "RIFF" and has "WAVE" at offset 8
                return header[:4] == b"RIFF" and header[8:12] == b"WAVE"
            return header.startswith(magic)

    def _read_audio_metadata(
        self, file_path: str, extension: str
    ) -> tuple[int, float]:
        """讀取音訊的取樣率和時長.

        Args:
            file_path: 音訊檔案路徑
            extension: 檔案副檔名 (wav, mp3, flac)

        Returns:
            Tuple of (sample_rate_hz, duration_seconds)

        Raises:
            Exception: 當檔案無法解碼時
        """
        if extension == "wav":
            return self._read_wav_metadata(file_path)
        elif extension == "mp3":
            return self._read_mp3_metadata(file_path)
        elif extension == "flac":
            return self._read_flac_metadata(file_path)
        else:
            raise ValueError(f"Unsupported format: {extension}")

    def _read_wav_metadata(self, file_path: str) -> tuple[int, float]:
        """使用標準 wave 模組讀取 WAV 檔案的 metadata."""
        with wave.open(file_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / sample_rate
        return sample_rate, duration

    def _read_mp3_metadata(self, file_path: str) -> tuple[int, float]:
        """使用 mutagen 讀取 MP3 檔案的 metadata."""
        audio = MP3(file_path)
        if audio.info is None:
            raise ValueError("Cannot read MP3 audio info")
        sample_rate = audio.info.sample_rate
        duration = audio.info.length
        return sample_rate, duration

    def _read_flac_metadata(self, file_path: str) -> tuple[int, float]:
        """使用 mutagen 讀取 FLAC 檔案的 metadata."""
        audio = FLAC(file_path)
        if audio.info is None:
            raise ValueError("Cannot read FLAC audio info")
        sample_rate = audio.info.sample_rate
        duration = audio.info.length
        return sample_rate, duration
