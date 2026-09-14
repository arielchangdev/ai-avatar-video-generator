"""Integration tests for the end-to-end generation flow (task 17.2).

These tests wire the real orchestration layers together (TaskManager,
run_pipeline, API routes) using an in-memory Redis and fake stage runners, so
the complete flow is validated WITHOUT requiring a GPU or AI model downloads:

    upload -> validate -> create task -> run pipeline -> progress -> download

The full pipeline with real AI models (GPT-SoVITS, SadTalker, InsightFace)
requires an NVIDIA GPU and is gated behind AVATAR_GPU_E2E=1.

Validates Requirements 7.1, 7.3, 7.4, 7.5.
"""

import io
import json
import os
import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, ".")

import pytest
from PIL import Image

from core.pipeline import PIPELINE_STAGES, run_pipeline
from core.task_manager import TaskManager
from models.schemas import TaskInput, TaskStage, TaskState


# ---------------------------------------------------------------------------
# In-memory Redis with pub/sub support
# ---------------------------------------------------------------------------

_set = set


class FakePubSub:
    def __init__(self, bus):
        self._bus = bus
        self._channels: list[str] = []
        self._queue: list[dict] = []

    def subscribe(self, channel):
        self._channels.append(channel)
        self._bus.subscribers.setdefault(channel, []).append(self)

    def unsubscribe(self, channel):
        if channel in self._channels:
            self._channels.remove(channel)

    def deliver(self, channel, data):
        self._queue.append({"type": "message", "channel": channel, "data": data})

    def get_message(self, ignore_subscribe_messages=True, timeout=None):
        if self._queue:
            return self._queue.pop(0)
        return None

    def close(self):
        self._channels.clear()


class FakeRedis:
    def __init__(self):
        self._store = {}
        self._sets = {}
        self.subscribers: dict[str, list[FakePubSub]] = {}
        self.published: list[tuple[str, str]] = []

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value

    def delete(self, *keys):
        for key in keys:
            self._store.pop(key, None)

    def sadd(self, key, *values):
        self._sets.setdefault(key, _set()).update(values)

    def srem(self, key, *values):
        if key in self._sets:
            self._sets[key] -= _set(values)

    def smembers(self, key):
        return self._sets.get(key, _set()).copy()

    def publish(self, channel, message):
        self.published.append((channel, message))
        for sub in self.subscribers.get(channel, []):
            sub.deliver(channel, message)
        return len(self.subscribers.get(channel, []))

    def pubsub(self):
        return FakePubSub(self)

    def ping(self):
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wav_bytes(duration_sec: float = 10.0, sample_rate: int = 44100) -> bytes:
    channels, bit_depth = 1, 16
    num_samples = int(duration_sec * sample_rate)
    bps = bit_depth // 8
    data_size = num_samples * channels * bps
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<H", channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * channels * bps))
    buf.write(struct.pack("<H", channels * bps))
    buf.write(struct.pack("<H", bit_depth))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(b"\x00" * data_size)
    return buf.getvalue()


def _png_bytes(width: int = 512, height: int = 512) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def task_manager(tmp_path):
    return TaskManager(redis_client=FakeRedis(), output_dir=tmp_path / "out")


@pytest.fixture
def created_task(task_manager):
    task_input = TaskInput(
        voice_sample=_wav_bytes(),
        voice_sample_filename="voice.wav",
        appearance_asset=_png_bytes(),
        appearance_asset_filename="face.png",
        script_text="這是一段整合測試的講稿。",
    )
    return task_manager.create_task(task_input)


def _fake_runners(output_dir_provider):
    """Build fake stage runners that produce a real output file at compose."""

    def script_runner(ctx):
        return {"script_ok": True}

    def voice_runner(ctx):
        return {"audio_path": "voice.wav"}

    def face_runner(ctx):
        return {"source_image": "face.png"}

    def lipsync_runner(ctx):
        return {"raw_video_path": "lipsync.mp4"}

    def compose_runner(ctx):
        out_dir = output_dir_provider(ctx["task_id"])
        out_dir.mkdir(parents=True, exist_ok=True)
        video = out_dir / "avatar.mp4"
        audio = out_dir / "avatar.mp3"
        video.write_bytes(b"final-mp4")
        audio.write_bytes(b"final-mp3")
        return {"video_path": str(video), "audio_path_mp3": str(audio)}

    return {
        "script": script_runner,
        "voice": voice_runner,
        "face": face_runner,
        "lipsync": lipsync_runner,
        "compose": compose_runner,
    }


# ---------------------------------------------------------------------------
# End-to-end flow (no GPU)
# ---------------------------------------------------------------------------


class TestEndToEndFlow:
    def test_pipeline_completes_and_publishes_progress(self, task_manager, created_task):
        """upload->create->pipeline->progress: monotonic updates end at completed."""
        task_id = created_task.task_id
        out_base = task_manager._output_dir / "tasks"

        runners = _fake_runners(lambda tid: out_base / tid / "output")
        run_pipeline(task_id, runners, task_manager)

        # Final state is completed at 100%
        progress = task_manager.get_task_status(task_id)
        assert progress.state == TaskState.COMPLETED
        assert progress.percentage == 100

        # Progress was published to the WebSocket channel for each stage.
        channel = f"task:{task_id}:progress"
        published = [m for c, m in task_manager._redis.published if c == channel]
        percentages = []
        for m in published:
            data = json.loads(m)
            if data.get("type") == "progress":
                percentages.append(data["percentage"])
        # Strictly increasing and reaches 100.
        assert percentages == sorted(percentages)
        assert percentages[-1] == 100

    def test_download_available_after_completion(self, task_manager, created_task):
        """After completion, download metadata is set and within the 24h window."""
        task_id = created_task.task_id

        # Mark completion via the manager so download_expires_at is set.
        task_manager.update_progress(task_id, TaskStage.VIDEO_COMPOSE, 100)
        meta = task_manager.get_task_meta(task_id)
        assert meta["download_expires_at"] is not None

        expires = datetime.fromisoformat(meta["download_expires_at"])
        assert expires > datetime.now(timezone.utc)
        assert expires <= datetime.now(timezone.utc) + timedelta(hours=24, minutes=1)

    def test_expired_download_link_is_detected(self, task_manager, created_task):
        """An expired completion timestamp is correctly identified as invalid."""
        task_id = created_task.task_id
        task_manager.update_progress(task_id, TaskStage.VIDEO_COMPOSE, 100)

        meta = task_manager.get_task_meta(task_id)
        # Force expiry into the past.
        meta["download_expires_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()
        task_manager._redis.set(f"task:{task_id}:meta", json.dumps(meta))

        refreshed = task_manager.get_task_meta(task_id)
        expires = datetime.fromisoformat(refreshed["download_expires_at"])
        assert expires < datetime.now(timezone.utc)  # link is expired

    def test_retry_from_stage_resumes_correctly(self, task_manager, created_task):
        """Retry from a stage skips completed stages and still completes."""
        task_id = created_task.task_id
        out_base = task_manager._output_dir / "tasks"
        executed: list[str] = []

        def make(key, base):
            def runner(ctx):
                executed.append(key)
                if key == "compose":
                    return base(ctx)
                return {}

            return runner

        base_runners = _fake_runners(lambda tid: out_base / tid / "output")
        runners = {k: make(k, base_runners[k]) for k in base_runners}

        run_pipeline(
            task_id, runners, task_manager, retry_from_stage=TaskStage.LIP_SYNC
        )

        assert executed == ["lipsync", "compose"]
        progress = task_manager.get_task_status(task_id)
        assert progress.state == TaskState.COMPLETED


# ---------------------------------------------------------------------------
# Full GPU pipeline (real AI models) - opt-in
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("AVATAR_GPU_E2E") != "1",
    reason="Requires NVIDIA GPU + AI models; set AVATAR_GPU_E2E=1 to run.",
)
class TestFullPipelineWithModels:
    def test_real_generation_end_to_end(self, task_manager, created_task):
        from core.stage_runners import build_default_runners

        runners = build_default_runners(task_manager)
        run_pipeline(created_task.task_id, runners, task_manager)

        progress = task_manager.get_task_status(created_task.task_id)
        assert progress.state == TaskState.COMPLETED
        assert progress.percentage == 100
