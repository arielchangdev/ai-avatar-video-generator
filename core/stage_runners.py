"""AI Avatar Video Generator - Default pipeline stage runners.

Wires the concrete AI modules into the pipeline orchestration defined in
``core.pipeline``. Each runner takes the accumulated context dict and returns
updates to merge back in.

These runners load heavy AI models (GPT-SoVITS, SadTalker, InsightFace) and
run FFmpeg, so they are exercised by the GPU integration tests rather than the
unit suite. The orchestration logic itself is unit-tested with fake runners.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import cv2

from config.settings import settings
from core.task_manager import TaskManager
from modules.face_detector import FaceDetector
from modules.lip_sync_engine import LipSyncEngine
from modules.script_processor import ScriptProcessor
from modules.video_composer import VideoComposer
from modules.voice_cloner import VoiceCloner

logger = logging.getLogger(__name__)


def _task_dirs(task_id: str) -> tuple[Path, Path, Path]:
    """Return (input_dir, intermediate_dir, output_dir) for a task."""
    base = settings.tasks_dir / task_id
    return base / "input", base / "intermediate", base / "output"


def build_default_runners(
    task_manager: TaskManager,
) -> dict[str, Callable[[dict], dict]]:
    """Build the default stage runner mapping wired to real AI modules.

    Models are lazily constructed on first use and cached in a closure so the
    worker process loads each model once.
    """
    cache: dict[str, object] = {}

    def _script_runner(ctx: dict) -> dict:
        meta = task_manager.get_task_meta(ctx["task_id"])
        script_text = meta["script_text"]
        processor = ScriptProcessor()
        return {"script_result": processor.process(script_text)}

    def _voice_runner(ctx: dict) -> dict:
        meta = task_manager.get_task_meta(ctx["task_id"])
        _, intermediate, _ = _task_dirs(ctx["task_id"])
        intermediate.mkdir(parents=True, exist_ok=True)

        cloner = cache.get("voice")
        if cloner is None:
            cloner = VoiceCloner(model_path=str(settings.model_dir / "gpt_sovits"))
            cache["voice"] = cloner

        profile = cloner.analyze_voice_sample(meta["voice_sample_path"])
        audio_out = str(intermediate / "voice.wav")
        result = cloner.synthesize(ctx["script_result"], profile, audio_out)
        return {"synthesis_result": result, "audio_path": result.audio_path}

    def _face_runner(ctx: dict) -> dict:
        meta = task_manager.get_task_meta(ctx["task_id"])
        detector = cache.get("face")
        if detector is None:
            detector = FaceDetector()
            cache["face"] = detector

        image = cv2.imread(meta["appearance_asset_path"])
        faces = detector.detect_faces(image)
        primary = detector.select_primary_face(faces)
        return {"primary_face": primary, "source_image": meta["appearance_asset_path"]}

    def _lipsync_runner(ctx: dict) -> dict:
        _, intermediate, _ = _task_dirs(ctx["task_id"])
        engine = cache.get("lipsync")
        if engine is None:
            engine = LipSyncEngine(
                checkpoint_dir=str(settings.model_dir / "sadtalker")
            )
            cache["lipsync"] = engine

        video_out = str(intermediate / "lipsync.mp4")
        result = engine.generate(
            source_image=ctx["source_image"],
            audio_path=ctx["audio_path"],
            output_path=video_out,
        )
        return {"lipsync_result": result, "raw_video_path": result.video_path}

    def _compose_runner(ctx: dict) -> dict:
        _, _, output = _task_dirs(ctx["task_id"])
        output.mkdir(parents=True, exist_ok=True)
        composer = VideoComposer()

        video_out = str(output / "avatar.mp4")
        result = composer.compose(
            video_path=ctx["raw_video_path"],
            audio_path=ctx["audio_path"],
            output_path=video_out,
        )
        # Persist output paths onto task metadata for the download endpoints.
        meta_key = f"task:{ctx['task_id']}:meta"
        try:
            meta = task_manager.get_task_meta(ctx["task_id"])
            meta["output_video_path"] = result.video_path
            meta["output_audio_path"] = result.audio_mp3_path
            import json

            task_manager._redis.set(meta_key, json.dumps(meta))
        except Exception:  # pragma: no cover - best effort
            logger.warning("Failed to persist output paths for task.")
        return {"compose_result": result}

    return {
        "script": _script_runner,
        "voice": _voice_runner,
        "face": _face_runner,
        "lipsync": _lipsync_runner,
        "compose": _compose_runner,
    }
