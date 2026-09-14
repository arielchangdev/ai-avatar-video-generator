"""AI Avatar Video Generator - Voice cloning and synthesis module.

Based on GPT-SoVITS for few-shot voice cloning with Chinese/English support.
Handles graceful degradation when GPT-SoVITS is not installed.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.exceptions import VoiceSynthesisError
from models.schemas import (
    LanguageSegment,
    LanguageType,
    ScriptResult,
    SynthesisResult,
    TaskStage,
    VoiceProfile,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Target output audio parameters
TARGET_SAMPLE_RATE = 44100
TARGET_BIT_DEPTH = 16
TARGET_CHANNELS = 1

# Silence insertion parameters
DEFAULT_SILENCE_DURATION_SEC = 1.0
MIN_SILENCE_DURATION_SEC = 0.5
MAX_SILENCE_DURATION_SEC = 1.5

# Minimum voice sample requirements
MIN_SAMPLE_DURATION_SEC = 3.0


class VoiceCloner:
    """Voice cloning module based on GPT-SoVITS.

    Handles voice sample analysis and speech synthesis using the GPT-SoVITS
    model. Gracefully degrades when GPT-SoVITS is not installed.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """Load GPT-SoVITS model.

        Args:
            model_path: Path to the GPT-SoVITS model weights directory.
            device: Computation device ('cuda' or 'cpu').

        Raises:
            VoiceSynthesisError: If GPT-SoVITS is not installed or model fails to load.
        """
        self._model_path = model_path
        self._device = device
        self._model = None

        try:
            from GPT_SoVITS.inference import GPTSoVITSInference  # noqa: F401
        except ImportError as e:
            raise VoiceSynthesisError(
                stage=TaskStage.VOICE_SYNTHESIS,
                reason=(
                    "GPT-SoVITS is not installed. "
                    "Install from: https://github.com/RVC-Boss/GPT-SoVITS"
                ),
                recoverable=False,
            ) from e

        try:
            from GPT_SoVITS.inference import GPTSoVITSInference

            self._model = GPTSoVITSInference(
                model_path=model_path,
                device=device,
            )
            logger.info(
                "GPT-SoVITS model loaded from '%s' on device '%s'.",
                model_path,
                device,
            )
        except Exception as e:
            raise VoiceSynthesisError(
                stage=TaskStage.VOICE_SYNTHESIS,
                reason=f"Failed to load GPT-SoVITS model from '{model_path}': {e}",
                recoverable=False,
            ) from e

    def analyze_voice_sample(self, audio_path: str) -> VoiceProfile:
        """Analyze voice sample and extract speaker characteristics.

        Args:
            audio_path: Path to the voice sample audio file (WAV/MP3/FLAC).

        Returns:
            VoiceProfile containing sample metadata and speaker embedding.

        Raises:
            VoiceSynthesisError: If the audio file cannot be read, is too short,
                or voice features cannot be extracted.
        """
        if not os.path.isfile(audio_path):
            raise VoiceSynthesisError(
                stage=TaskStage.VOICE_ANALYSIS,
                reason=f"Voice sample file not found: {audio_path}",
                recoverable=False,
            )

        # Read audio file to get duration and sample rate
        try:
            duration_sec, sample_rate = self._get_audio_info(audio_path)
        except Exception as e:
            raise VoiceSynthesisError(
                stage=TaskStage.VOICE_ANALYSIS,
                reason=f"Failed to read voice sample: {e}",
                recoverable=False,
            ) from e

        # Validate minimum duration
        if duration_sec < MIN_SAMPLE_DURATION_SEC:
            raise VoiceSynthesisError(
                stage=TaskStage.VOICE_ANALYSIS,
                reason=(
                    "Voice sample does not meet minimum requirements: "
                    f"duration {duration_sec:.1f}s is less than "
                    f"{MIN_SAMPLE_DURATION_SEC:.0f}s minimum"
                ),
                recoverable=True,
            )

        # Extract speaker embedding using GPT-SoVITS
        try:
            embedding = self._extract_embedding(audio_path)
        except Exception as e:
            raise VoiceSynthesisError(
                stage=TaskStage.VOICE_ANALYSIS,
                reason=(
                    "Voice sample does not meet minimum requirements: "
                    f"cannot extract valid voice features - {e}"
                ),
                recoverable=True,
            ) from e

        return VoiceProfile(
            sample_path=audio_path,
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            embedding=embedding,
        )

    def synthesize(
        self,
        script_result: ScriptResult,
        voice_profile: VoiceProfile,
        output_path: str,
    ) -> SynthesisResult:
        """Synthesize speech from script using cloned voice.

        The output is written to a temporary file first and moved to
        output_path only on success, ensuring no incomplete file on failure.

        Args:
            script_result: Processed script with language segments and
                paragraph break positions.
            voice_profile: Voice profile extracted from the reference sample.
            output_path: Destination path for the synthesized WAV file.

        Returns:
            SynthesisResult with audio metadata.

        Raises:
            VoiceSynthesisError: If synthesis fails at any point.
        """
        if self._model is None:
            raise VoiceSynthesisError(
                stage=TaskStage.VOICE_SYNTHESIS,
                reason="Voice synthesis model is not initialized.",
                recoverable=False,
            )

        if not script_result.segments:
            raise VoiceSynthesisError(
                stage=TaskStage.VOICE_SYNTHESIS,
                reason="No text segments to synthesize.",
                recoverable=False,
            )

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Use a temporary file to prevent incomplete output on failure
        temp_fd = None
        temp_path = None
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=".wav", dir=output_dir or None
            )
            os.close(temp_fd)
            temp_fd = None

            # Synthesize all segments and collect audio data
            all_audio_samples = self._synthesize_segments(
                script_result, voice_profile
            )

            # Write final WAV file to temp path
            self._write_wav(temp_path, all_audio_samples)

            # Calculate duration
            total_samples = len(all_audio_samples)
            duration_sec = total_samples / TARGET_SAMPLE_RATE

            # Move temp file to final output path
            shutil.move(temp_path, output_path)
            temp_path = None  # Prevent cleanup of moved file

            logger.info(
                "Voice synthesis complete: '%s' (%.1f sec, %dHz, %d-bit)",
                output_path,
                duration_sec,
                TARGET_SAMPLE_RATE,
                TARGET_BIT_DEPTH,
            )

            return SynthesisResult(
                audio_path=output_path,
                duration_sec=duration_sec,
                sample_rate=TARGET_SAMPLE_RATE,
                bit_depth=TARGET_BIT_DEPTH,
            )

        except VoiceSynthesisError:
            raise
        except Exception as e:
            raise VoiceSynthesisError(
                stage=TaskStage.VOICE_SYNTHESIS,
                reason=f"Voice synthesis failed: {e}",
                recoverable=True,
            ) from e
        finally:
            # Clean up temp file if it still exists (synthesis failed)
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

    def _synthesize_segments(
        self,
        script_result: ScriptResult,
        voice_profile: VoiceProfile,
    ) -> np.ndarray:
        """Synthesize all segments and insert paragraph break silences."""
        audio_chunks: list[np.ndarray] = []
        paragraph_breaks_set = set(script_result.paragraph_breaks)

        for segment in script_result.segments:
            # Insert silence at paragraph breaks
            if segment.start_index in paragraph_breaks_set:
                silence = self._generate_silence(DEFAULT_SILENCE_DURATION_SEC)
                audio_chunks.append(silence)

            # Synthesize this segment
            segment_audio = self._synthesize_single_segment(
                segment, voice_profile
            )
            audio_chunks.append(segment_audio)

        if not audio_chunks:
            raise VoiceSynthesisError(
                stage=TaskStage.VOICE_SYNTHESIS,
                reason="No audio generated from segments.",
                recoverable=True,
            )

        return np.concatenate(audio_chunks)

    def _synthesize_single_segment(
        self,
        segment: LanguageSegment,
        voice_profile: VoiceProfile,
    ) -> np.ndarray:
        """Synthesize a single language segment using GPT-SoVITS."""
        language_map = {
            LanguageType.CHINESE: "zh",
            LanguageType.ENGLISH: "en",
            LanguageType.UNKNOWN: "zh",
        }
        lang = language_map.get(segment.language, "zh")

        try:
            audio_data = self._model.synthesize(
                text=segment.text,
                ref_audio_path=voice_profile.sample_path,
                language=lang,
            )

            if isinstance(audio_data, np.ndarray):
                return self._ensure_target_format(audio_data)
            else:
                raise VoiceSynthesisError(
                    stage=TaskStage.VOICE_SYNTHESIS,
                    reason=f"Unexpected audio output type: {type(audio_data)}",
                    recoverable=True,
                )

        except VoiceSynthesisError:
            raise
        except Exception as e:
            text_preview = segment.text[:30]
            raise VoiceSynthesisError(
                stage=TaskStage.VOICE_SYNTHESIS,
                reason=f"Failed to synthesize segment '{text_preview}...': {e}",
                recoverable=True,
            ) from e

    def _ensure_target_format(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio is int16. Convert from float if needed."""
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)
        elif audio.dtype != np.int16:
            audio = audio.astype(np.int16)
        return audio

    def _generate_silence(self, duration_sec: float) -> np.ndarray:
        """Generate silence samples for paragraph breaks (0.5-1.5s)."""
        duration_sec = max(
            MIN_SILENCE_DURATION_SEC,
            min(MAX_SILENCE_DURATION_SEC, duration_sec),
        )
        num_samples = int(TARGET_SAMPLE_RATE * duration_sec)
        return np.zeros(num_samples, dtype=np.int16)

    def _write_wav(self, path: str, samples: np.ndarray) -> None:
        """Write int16 samples to a WAV file."""
        with wave.open(path, "wb") as wf:
            wf.setnchannels(TARGET_CHANNELS)
            wf.setsampwidth(TARGET_BIT_DEPTH // 8)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(samples.tobytes())

    def _get_audio_info(self, audio_path: str) -> tuple[float, int]:
        """Get audio duration and sample rate from file."""
        # Try soundfile first (supports WAV, FLAC, and many formats)
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return info.duration, info.samplerate
        except ImportError:
            pass
        except Exception:
            pass

        # Fall back to wave module for WAV files
        ext = Path(audio_path).suffix.lower()
        if ext == ".wav":
            try:
                with wave.open(audio_path, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / rate
                    return duration, rate
            except Exception as e:
                raise RuntimeError(f"Cannot read WAV file: {e}") from e

        # Try librosa as last resort (supports MP3)
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            return duration, sr
        except ImportError:
            pass
        except Exception as e:
            raise RuntimeError(f"Cannot read audio file: {e}") from e

        raise RuntimeError(
            "No suitable audio library available. Install soundfile or librosa."
        )

    def _extract_embedding(self, audio_path: str) -> list[float]:
        """Extract speaker embedding from audio using GPT-SoVITS."""
        try:
            embedding = self._model.extract_speaker_embedding(audio_path)
            if isinstance(embedding, np.ndarray):
                return embedding.tolist()
            return list(embedding)
        except AttributeError:
            # Fallback: basic feature extraction if model lacks the method
            try:
                import soundfile as sf
                data, sr = sf.read(audio_path)
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                frame_size = int(sr * 0.025)
                hop_size = int(sr * 0.010)
                frames = [
                    data[i : i + frame_size]
                    for i in range(0, len(data) - frame_size, hop_size)
                ]
                energies = [np.sum(f**2) for f in frames[:256]]
                embedding_arr = np.array(energies, dtype=np.float32)
                if embedding_arr.max() > 0:
                    embedding_arr = embedding_arr / embedding_arr.max()
                return embedding_arr.tolist()
            except Exception as e:
                raise RuntimeError(f"Failed to extract voice embedding: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Speaker embedding extraction failed: {e}") from e
