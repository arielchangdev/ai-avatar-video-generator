"""Unit tests for TaskManager with Redis state storage.

Tests task creation, progress tracking with strict monotonicity,
state transitions, and cleanup of expired tasks.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.task_manager import (
    DOWNLOAD_EXPIRY_HOURS,
    STAGE_ORDER,
    TaskManager,
    _stage_index,
)
from models.schemas import (
    Task,
    TaskInput,
    TaskProgress,
    TaskStage,
    TaskState,
)

# Save builtin set before class definition shadows it
_set = set


class FakeRedis:
    """A minimal in-memory Redis mock for testing."""

    def __init__(self):
        self._store = {}
        self._sets = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value

    def delete(self, *keys):
        for key in keys:
            self._store.pop(key, None)

    def sadd(self, key, *values):
        if key not in self._sets:
            self._sets[key] = _set()
        self._sets[key].update(values)

    def srem(self, key, *values):
        if key in self._sets:
            self._sets[key] -= _set(values)

    def smembers(self, key):
        return self._sets.get(key, _set()).copy()


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def tmp_output_dir(tmp_path):
    return tmp_path / "avatar-output"


@pytest.fixture
def task_manager(fake_redis, tmp_output_dir):
    return TaskManager(redis_client=fake_redis, output_dir=tmp_output_dir)


@pytest.fixture
def sample_task_input():
    return TaskInput(
        voice_sample=b"fake_wav_data_for_testing",
        voice_sample_filename="test_voice.wav",
        appearance_asset=b"fake_image_data_for_testing",
        appearance_asset_filename="test_face.jpg",
        script_text="This is test script text.",
    )


class TestCreateTask:
    def test_creates_task_with_valid_input(self, task_manager, sample_task_input, tmp_output_dir):
        task = task_manager.create_task(sample_task_input)
        assert task.task_id is not None
        assert len(task.task_id) == 36
        assert task.state == TaskState.PENDING
        assert task.progress.state == TaskState.PENDING
        assert task.progress.percentage == 0
        assert task.progress.current_stage is None
        assert task.download_expires_at is None

    def test_creates_directory_structure(self, task_manager, sample_task_input, tmp_output_dir):
        task = task_manager.create_task(sample_task_input)
        task_dir = tmp_output_dir / "tasks" / task.task_id
        assert (task_dir / "input").is_dir()
        assert (task_dir / "intermediate").is_dir()
        assert (task_dir / "output").is_dir()

    def test_saves_input_files(self, task_manager, sample_task_input, tmp_output_dir):
        task = task_manager.create_task(sample_task_input)
        task_dir = tmp_output_dir / "tasks" / task.task_id
        voice_path = task_dir / "input" / "test_voice.wav"
        assert voice_path.exists()
        assert voice_path.read_bytes() == b"fake_wav_data_for_testing"
        appearance_path = task_dir / "input" / "test_face.jpg"
        assert appearance_path.exists()
        assert appearance_path.read_bytes() == b"fake_image_data_for_testing"
        script_path = task_dir / "input" / "script.txt"
        assert script_path.exists()
        assert "This is test script text." in script_path.read_text("utf-8")

    def test_stores_status_in_redis(self, task_manager, sample_task_input, fake_redis):
        task = task_manager.create_task(sample_task_input)
        status_key = f"task:{task.task_id}:status"
        assert fake_redis.get(status_key) is not None
        progress = TaskProgress.model_validate_json(fake_redis.get(status_key))
        assert progress.task_id == task.task_id
        assert progress.state == TaskState.PENDING

    def test_stores_metadata_in_redis(self, task_manager, sample_task_input, fake_redis):
        task = task_manager.create_task(sample_task_input)
        meta_key = f"task:{task.task_id}:meta"
        assert fake_redis.get(meta_key) is not None
        meta = json.loads(fake_redis.get(meta_key))
        assert meta["task_id"] == task.task_id
        assert meta["state"] == TaskState.PENDING.value

    def test_adds_to_active_set(self, task_manager, sample_task_input, fake_redis):
        task = task_manager.create_task(sample_task_input)
        active = fake_redis.smembers("tasks:active")
        assert task.task_id in active

class TestGetTaskStatus:
    def test_retrieves_existing_task(self, task_manager, sample_task_input):
        task = task_manager.create_task(sample_task_input)
        progress = task_manager.get_task_status(task.task_id)
        assert progress.task_id == task.task_id
        assert progress.state == TaskState.PENDING
        assert progress.percentage == 0

    def test_raises_key_error_for_unknown_task(self, task_manager):
        with pytest.raises(KeyError, match="not found"):
            task_manager.get_task_status("non-existent-id")


class TestUpdateProgress:
    def test_updates_stage_and_percentage(self, task_manager, sample_task_input):
        task = task_manager.create_task(sample_task_input)
        task_manager.update_progress(task.task_id, TaskStage.VOICE_ANALYSIS, 10)
        progress = task_manager.get_task_status(task.task_id)
        assert progress.current_stage == TaskStage.VOICE_ANALYSIS
        assert progress.percentage == 10
        assert progress.state == TaskState.PROCESSING

    def test_rejects_non_monotonic_percentage(self, task_manager, sample_task_input):
        task = task_manager.create_task(sample_task_input)
        task_manager.update_progress(task.task_id, TaskStage.VOICE_ANALYSIS, 10)
        with pytest.raises(ValueError, match="strictly monotonic"):
            task_manager.update_progress(task.task_id, TaskStage.VOICE_ANALYSIS, 10)
        with pytest.raises(ValueError, match="strictly monotonic"):
            task_manager.update_progress(task.task_id, TaskStage.VOICE_SYNTHESIS, 5)

    def test_rejects_stage_order_violation(self, task_manager, sample_task_input):
        task = task_manager.create_task(sample_task_input)
        task_manager.update_progress(task.task_id, TaskStage.FACE_MODELING, 50)
        with pytest.raises(ValueError, match="Stage order violation"):
            task_manager.update_progress(task.task_id, TaskStage.VOICE_ANALYSIS, 60)

    def test_allows_same_stage_with_higher_percentage(self, task_manager, sample_task_input):
        task = task_manager.create_task(sample_task_input)
        task_manager.update_progress(task.task_id, TaskStage.VOICE_SYNTHESIS, 20)
        task_manager.update_progress(task.task_id, TaskStage.VOICE_SYNTHESIS, 35)
        progress = task_manager.get_task_status(task.task_id)
        assert progress.percentage == 35
        assert progress.current_stage == TaskStage.VOICE_SYNTHESIS

    def test_allows_forward_stage_with_higher_percentage(self, task_manager, sample_task_input):
        task = task_manager.create_task(sample_task_input)
        task_manager.update_progress(task.task_id, TaskStage.VOICE_ANALYSIS, 10)
        task_manager.update_progress(task.task_id, TaskStage.VOICE_SYNTHESIS, 30)
        task_manager.update_progress(task.task_id, TaskStage.FACE_MODELING, 55)
        progress = task_manager.get_task_status(task.task_id)
        assert progress.current_stage == TaskStage.FACE_MODELING
        assert progress.percentage == 55

    def test_sets_completed_state_at_100_percent(self, task_manager, sample_task_input):
        task = task_manager.create_task(sample_task_input)
        task_manager.update_progress(task.task_id, TaskStage.VIDEO_COMPOSE, 100)
        progress = task_manager.get_task_status(task.task_id)
        assert progress.state == TaskState.COMPLETED
        assert progress.percentage == 100

    def test_download_expires_at_set_on_completion(self, task_manager, sample_task_input, fake_redis):
        task = task_manager.create_task(sample_task_input)
        task_manager.update_progress(task.task_id, TaskStage.VIDEO_COMPOSE, 100)
        meta_key = f"task:{task.task_id}:meta"
        meta = json.loads(fake_redis.get(meta_key))
        assert meta["download_expires_at"] is not None
        expires_at = datetime.fromisoformat(meta["download_expires_at"])
        expected = datetime.now(timezone.utc) + timedelta(hours=24)
        assert abs((expires_at - expected).total_seconds()) < 5

    def test_rejects_out_of_range_percentage(self, task_manager, sample_task_input):
        task = task_manager.create_task(sample_task_input)
        with pytest.raises(ValueError, match="between 0 and 100"):
            task_manager.update_progress(task.task_id, TaskStage.VOICE_ANALYSIS, -1)
        with pytest.raises(ValueError, match="between 0 and 100"):
            task_manager.update_progress(task.task_id, TaskStage.VOICE_ANALYSIS, 101)

class TestMarkFailed:
    def test_marks_task_as_failed(self, task_manager, sample_task_input):
        task = task_manager.create_task(sample_task_input)
        task_manager.update_progress(task.task_id, TaskStage.VOICE_SYNTHESIS, 30)
        task_manager.mark_failed(task.task_id, TaskStage.VOICE_SYNTHESIS, "GPT-SoVITS model crashed")
        progress = task_manager.get_task_status(task.task_id)
        assert progress.state == TaskState.FAILED
        assert progress.error_stage == TaskStage.VOICE_SYNTHESIS
        assert progress.error_message == "GPT-SoVITS model crashed"
        assert progress.percentage == 30


class TestCleanupExpiredTasks:
    def test_removes_expired_tasks(self, task_manager, sample_task_input, fake_redis, tmp_output_dir):
        task = task_manager.create_task(sample_task_input)
        meta_key = f"task:{task.task_id}:meta"
        meta = json.loads(fake_redis.get(meta_key))
        old_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        meta["created_at"] = old_time
        fake_redis.set(meta_key, json.dumps(meta))
        removed = task_manager.cleanup_expired_tasks(max_age_hours=24)
        assert removed == 1
        assert fake_redis.get(f"task:{task.task_id}:status") is None
        assert fake_redis.get(f"task:{task.task_id}:meta") is None
        assert task.task_id not in fake_redis.smembers("tasks:active")
        task_dir = tmp_output_dir / "tasks" / task.task_id
        assert not task_dir.exists()

    def test_preserves_active_tasks(self, task_manager, sample_task_input, fake_redis, tmp_output_dir):
        task = task_manager.create_task(sample_task_input)
        removed = task_manager.cleanup_expired_tasks(max_age_hours=24)
        assert removed == 0
        assert fake_redis.get(f"task:{task.task_id}:status") is not None
        assert task.task_id in fake_redis.smembers("tasks:active")
        task_dir = tmp_output_dir / "tasks" / task.task_id
        assert task_dir.exists()

    def test_cleans_up_orphaned_entries(self, task_manager, fake_redis):
        fake_redis.sadd("tasks:active", "orphaned-task-id")
        removed = task_manager.cleanup_expired_tasks(max_age_hours=24)
        assert removed == 1
        assert "orphaned-task-id" not in fake_redis.smembers("tasks:active")


class TestStateTransitions:
    def test_full_lifecycle_pending_to_completed(self, task_manager, sample_task_input):
        task = task_manager.create_task(sample_task_input)
        progress = task_manager.get_task_status(task.task_id)
        assert progress.state == TaskState.PENDING
        task_manager.update_progress(task.task_id, TaskStage.VOICE_ANALYSIS, 10)
        progress = task_manager.get_task_status(task.task_id)
        assert progress.state == TaskState.PROCESSING
        task_manager.update_progress(task.task_id, TaskStage.VOICE_SYNTHESIS, 40)
        task_manager.update_progress(task.task_id, TaskStage.FACE_MODELING, 55)
        task_manager.update_progress(task.task_id, TaskStage.LIP_SYNC, 80)
        task_manager.update_progress(task.task_id, TaskStage.VIDEO_COMPOSE, 100)
        progress = task_manager.get_task_status(task.task_id)
        assert progress.state == TaskState.COMPLETED
        assert progress.percentage == 100

    def test_full_lifecycle_pending_to_failed(self, task_manager, sample_task_input):
        task = task_manager.create_task(sample_task_input)
        task_manager.update_progress(task.task_id, TaskStage.VOICE_ANALYSIS, 10)
        task_manager.mark_failed(task.task_id, TaskStage.VOICE_ANALYSIS, "Voice sample corrupted")
        progress = task_manager.get_task_status(task.task_id)
        assert progress.state == TaskState.FAILED
        assert progress.error_stage == TaskStage.VOICE_ANALYSIS