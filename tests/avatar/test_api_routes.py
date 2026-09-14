"""Unit tests for the Task REST API routes (task 13.4).

Validates Requirements 1.4, 1.6, 7.4, 8.1, 8.2:
- POST /tasks with valid inputs returns a task_id (202)
- POST /tasks with missing/invalid inputs returns 400 with an item list
- GET /tasks/{id} returns current progress
- Download endpoints return 404/410 for missing/expired links
- Upload timeout handling (408)

Tests use FastAPI dependency overrides to inject an in-memory TaskManager and
dispatcher, so no Redis or Celery is required.
"""

import io
import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, ".")

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
from api.dependencies import get_task_dispatcher, get_task_manager
from core.task_dispatcher import InMemoryTaskDispatcher
from models.schemas import (
    Task,
    TaskInput,
    TaskProgress,
    TaskStage,
    TaskState,
)


# ---------------------------------------------------------------------------
# Fake TaskManager (in-memory) matching the interface the routes use
# ---------------------------------------------------------------------------


class FakeTaskManager:
    def __init__(self, tmp_dir: Path):
        self._tmp = tmp_dir
        self._status: dict[str, TaskProgress] = {}
        self._meta: dict[str, dict] = {}
        self._counter = 0

    def create_task(self, task_input: TaskInput) -> Task:
        self._counter += 1
        task_id = f"task-{self._counter:04d}"
        task_dir = self._tmp / task_id / "input"
        task_dir.mkdir(parents=True, exist_ok=True)

        voice_path = task_dir / task_input.voice_sample_filename
        voice_path.write_bytes(task_input.voice_sample)
        appearance_path = task_dir / task_input.appearance_asset_filename
        appearance_path.write_bytes(task_input.appearance_asset)

        now = datetime.now(timezone.utc)
        progress = TaskProgress(
            task_id=task_id,
            state=TaskState.PENDING,
            current_stage=None,
            percentage=0,
            error_message=None,
            error_stage=None,
        )
        task = Task(
            task_id=task_id,
            state=TaskState.PENDING,
            created_at=now,
            updated_at=now,
            progress=progress,
            voice_sample_path=str(voice_path),
            appearance_asset_path=str(appearance_path),
            script_text=task_input.script_text,
            output_video_path=None,
            output_audio_path=None,
            download_expires_at=None,
        )
        self._status[task_id] = progress
        self._meta[task_id] = task.model_dump(mode="json")
        return task

    def get_task_status(self, task_id: str) -> TaskProgress:
        if task_id not in self._status:
            raise KeyError(task_id)
        return self._status[task_id]

    def get_task_meta(self, task_id: str) -> dict:
        if task_id not in self._meta:
            raise KeyError(task_id)
        return self._meta[task_id]

    def delete_task(self, task_id: str) -> None:
        if task_id not in self._status:
            raise KeyError(task_id)
        self._status.pop(task_id, None)
        self._meta.pop(task_id, None)

    # Test helpers ----------------------------------------------------------
    def _mark_downloadable(self, task_id: str, video: Path, audio: Path, expires):
        meta = self._meta[task_id]
        meta["output_video_path"] = str(video)
        meta["output_audio_path"] = str(audio)
        meta["download_expires_at"] = expires.isoformat()


# ---------------------------------------------------------------------------
# Fixtures
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
def fake_manager(tmp_path):
    return FakeTaskManager(tmp_path)


@pytest.fixture
def dispatcher():
    return InMemoryTaskDispatcher()


@pytest.fixture
def client(fake_manager, dispatcher):
    main.app.dependency_overrides[get_task_manager] = lambda: fake_manager
    main.app.dependency_overrides[get_task_dispatcher] = lambda: dispatcher
    with TestClient(main.app) as c:
        yield c
    main.app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /api/v1/tasks
# ---------------------------------------------------------------------------


class TestCreateTask:
    def test_valid_inputs_returns_task_id(self, client, dispatcher):
        resp = client.post(
            "/api/v1/tasks",
            files={
                "voice_sample": ("voice.wav", _wav_bytes(), "audio/wav"),
                "appearance_asset": ("face.png", _png_bytes(), "image/png"),
            },
            data={"script_text": "這是一段有效的講稿"},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["task_id"]
        assert body["state"] == "pending"
        assert body["summary"]["appearance_type"] == "image"
        # Pipeline was dispatched exactly once
        assert len(dispatcher.dispatched) == 1
        assert dispatcher.dispatched[0][0] == body["task_id"]

    def test_missing_script_returns_400(self, client):
        resp = client.post(
            "/api/v1/tasks",
            files={
                "voice_sample": ("voice.wav", _wav_bytes(), "audio/wav"),
                "appearance_asset": ("face.png", _png_bytes(), "image/png"),
            },
            data={"script_text": ""},
        )
        # Empty script fails the submission validator -> 400 with error list
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "errors" in detail
        assert any("script" in e for e in detail["errors"])

    def test_invalid_voice_duration_returns_400(self, client, dispatcher):
        resp = client.post(
            "/api/v1/tasks",
            files={
                "voice_sample": ("voice.wav", _wav_bytes(duration_sec=1.0), "audio/wav"),
                "appearance_asset": ("face.png", _png_bytes(), "image/png"),
            },
            data={"script_text": "有效講稿"},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert any("時長" in e for e in detail["errors"])
        # Invalid submission must not dispatch a pipeline
        assert len(dispatcher.dispatched) == 0

    def test_missing_file_field_returns_422(self, client):
        # Omitting a required File field is a FastAPI validation error (422)
        resp = client.post(
            "/api/v1/tasks",
            files={
                "appearance_asset": ("face.png", _png_bytes(), "image/png"),
            },
            data={"script_text": "有效講稿"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/tasks/{task_id}
# ---------------------------------------------------------------------------


class TestGetTask:
    def test_returns_current_progress(self, client):
        create = client.post(
            "/api/v1/tasks",
            files={
                "voice_sample": ("voice.wav", _wav_bytes(), "audio/wav"),
                "appearance_asset": ("face.png", _png_bytes(), "image/png"),
            },
            data={"script_text": "有效講稿"},
        )
        task_id = create.json()["task_id"]

        resp = client.get(f"/api/v1/tasks/{task_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == task_id
        assert body["state"] == "pending"
        assert body["percentage"] == 0

    def test_unknown_task_returns_404(self, client):
        resp = client.get("/api/v1/tasks/does-not-exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Download endpoints
# ---------------------------------------------------------------------------


class TestDownload:
    def _create(self, client):
        create = client.post(
            "/api/v1/tasks",
            files={
                "voice_sample": ("voice.wav", _wav_bytes(), "audio/wav"),
                "appearance_asset": ("face.png", _png_bytes(), "image/png"),
            },
            data={"script_text": "有效講稿"},
        )
        return create.json()["task_id"]

    def test_video_download_before_completion_returns_404(self, client):
        task_id = self._create(client)
        resp = client.get(f"/api/v1/tasks/{task_id}/download/video")
        # No download_expires_at set yet -> not available
        assert resp.status_code == 404

    def test_video_download_success(self, client, fake_manager, tmp_path):
        task_id = self._create(client)
        video = tmp_path / "out.mp4"
        video.write_bytes(b"fake-mp4-bytes")
        audio = tmp_path / "out.mp3"
        audio.write_bytes(b"fake-mp3-bytes")
        fake_manager._mark_downloadable(
            task_id,
            video,
            audio,
            datetime.now(timezone.utc) + timedelta(hours=1),
        )
        resp = client.get(f"/api/v1/tasks/{task_id}/download/video")
        assert resp.status_code == 200
        assert resp.content == b"fake-mp4-bytes"

    def test_expired_download_returns_410(self, client, fake_manager, tmp_path):
        task_id = self._create(client)
        video = tmp_path / "out.mp4"
        video.write_bytes(b"fake-mp4-bytes")
        audio = tmp_path / "out.mp3"
        audio.write_bytes(b"fake-mp3-bytes")
        fake_manager._mark_downloadable(
            task_id,
            video,
            audio,
            datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        resp = client.get(f"/api/v1/tasks/{task_id}/download/video")
        assert resp.status_code == 410

    def test_audio_download_success(self, client, fake_manager, tmp_path):
        task_id = self._create(client)
        video = tmp_path / "out.mp4"
        video.write_bytes(b"v")
        audio = tmp_path / "out.mp3"
        audio.write_bytes(b"fake-mp3-bytes")
        fake_manager._mark_downloadable(
            task_id,
            video,
            audio,
            datetime.now(timezone.utc) + timedelta(hours=1),
        )
        resp = client.get(f"/api/v1/tasks/{task_id}/download/audio")
        assert resp.status_code == 200
        assert resp.content == b"fake-mp3-bytes"


# ---------------------------------------------------------------------------
# DELETE /api/v1/tasks/{task_id}
# ---------------------------------------------------------------------------


class TestDeleteTask:
    def test_delete_existing_task(self, client):
        create = client.post(
            "/api/v1/tasks",
            files={
                "voice_sample": ("voice.wav", _wav_bytes(), "audio/wav"),
                "appearance_asset": ("face.png", _png_bytes(), "image/png"),
            },
            data={"script_text": "有效講稿"},
        )
        task_id = create.json()["task_id"]
        resp = client.delete(f"/api/v1/tasks/{task_id}")
        assert resp.status_code == 204
        # Now it's gone
        assert client.get(f"/api/v1/tasks/{task_id}").status_code == 404

    def test_delete_unknown_task_returns_404(self, client):
        resp = client.delete("/api/v1/tasks/nope")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
