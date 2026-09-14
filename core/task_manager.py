"""AI Avatar Video Generator - Task lifecycle management.

Manages task creation, progress tracking, and cleanup using Redis for state
storage and local filesystem for task artifacts.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config.settings import settings
from models.schemas import (
    Task,
    TaskInput,
    TaskProgress,
    TaskStage,
    TaskState,
)

logger = logging.getLogger(__name__)

# Ordered stages for monotonicity enforcement
STAGE_ORDER: list[TaskStage] = [
    TaskStage.VOICE_ANALYSIS,
    TaskStage.VOICE_SYNTHESIS,
    TaskStage.FACE_MODELING,
    TaskStage.LIP_SYNC,
    TaskStage.VIDEO_COMPOSE,
]

# Download link validity duration
DOWNLOAD_EXPIRY_HOURS = 24


def _stage_index(stage: TaskStage) -> int:
    """Return the ordinal index of a stage (0-based)."""
    return STAGE_ORDER.index(stage)


class TaskManager:
    """Task lifecycle management with Redis state storage.

    Handles task creation, progress tracking with strict monotonicity,
    and cleanup of expired tasks. Gracefully degrades when Redis is
    unavailable by logging warnings and raising appropriate errors.
    """

    def __init__(self, redis_client=None, output_dir: Path | None = None):
        """Initialize TaskManager.

        Args:
            redis_client: A Redis client instance. If None, creates one
                from settings.redis_url.
            output_dir: Base directory for task artifacts. Defaults to
                settings.avatar_output_dir.
        """
        self._output_dir = output_dir or settings.avatar_output_dir
        self._tasks_dir = self._output_dir / "tasks"

        if redis_client is not None:
            self._redis = redis_client
        else:
            self._redis = self._create_redis_client()

    def _create_redis_client(self):
        """Create a Redis client from settings."""
        try:
            import redis

            return redis.Redis.from_url(
                settings.redis_url,
                decode_responses=True,
            )
        except Exception as e:
            logger.warning("Failed to connect to Redis: %s", e)
            raise ConnectionError(
                f"Cannot connect to Redis at {settings.redis_url}: {e}"
            ) from e

    def create_task(self, task_input: TaskInput) -> Task:
        """Create a new generation task.

        Creates the task directory structure, stores input files, and
        registers the task in Redis.

        Args:
            task_input: The validated task input containing voice sample,
                appearance asset, and script text.

        Returns:
            The created Task object with all metadata.

        Raises:
            ConnectionError: If Redis is unavailable.
            OSError: If directory creation fails.
        """
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Create directory structure
        task_dir = self._tasks_dir / task_id
        input_dir = task_dir / "input"
        intermediate_dir = task_dir / "intermediate"
        output_dir = task_dir / "output"

        input_dir.mkdir(parents=True, exist_ok=True)
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save input files
        voice_sample_path = input_dir / task_input.voice_sample_filename
        voice_sample_path.write_bytes(task_input.voice_sample)

        appearance_asset_path = input_dir / task_input.appearance_asset_filename
        appearance_asset_path.write_bytes(task_input.appearance_asset)

        script_path = input_dir / "script.txt"
        script_path.write_text(task_input.script_text, encoding="utf-8")

        # Create TaskProgress
        progress = TaskProgress(
            task_id=task_id,
            state=TaskState.PENDING,
            current_stage=None,
            percentage=0,
            error_message=None,
            error_stage=None,
        )

        # Create Task
        task = Task(
            task_id=task_id,
            state=TaskState.PENDING,
            created_at=now,
            updated_at=now,
            progress=progress,
            voice_sample_path=str(voice_sample_path),
            appearance_asset_path=str(appearance_asset_path),
            script_text=task_input.script_text,
            output_video_path=None,
            output_audio_path=None,
            download_expires_at=None,
        )

        # Store in Redis
        self._store_task_in_redis(task)

        logger.info("Created task '%s' at '%s'.", task_id, task_dir)
        return task

    def get_task_status(self, task_id: str) -> TaskProgress:
        """Retrieve current task progress from Redis.

        Args:
            task_id: The unique task identifier.

        Returns:
            TaskProgress for the given task.

        Raises:
            KeyError: If task_id is not found in Redis.
            ConnectionError: If Redis is unavailable.
        """
        status_key = f"task:{task_id}:status"

        try:
            data = self._redis.get(status_key)
        except Exception as e:
            logger.error("Redis error while getting task status: %s", e)
            raise ConnectionError(
                f"Cannot retrieve task status from Redis: {e}"
            ) from e

        if data is None:
            raise KeyError(f"Task '{task_id}' not found.")

        return TaskProgress.model_validate_json(data)

    def update_progress(
        self, task_id: str, stage: TaskStage, percentage: int
    ) -> None:
        """Update task progress with strict monotonicity enforcement.

        Progress percentage must be strictly greater than the current
        percentage. Stage transitions must follow the defined order.

        Args:
            task_id: The unique task identifier.
            stage: The current processing stage.
            percentage: The new progress percentage (0-100).

        Raises:
            ValueError: If percentage is not strictly greater than current,
                or if stage order is violated.
            KeyError: If task_id is not found.
            ConnectionError: If Redis is unavailable.
        """
        if percentage < 0 or percentage > 100:
            raise ValueError(
                f"Percentage must be between 0 and 100, got {percentage}."
            )

        # Get current progress
        current = self.get_task_status(task_id)

        # Enforce strict monotonicity on percentage
        if percentage <= current.percentage:
            raise ValueError(
                f"Progress must be strictly monotonic: "
                f"new percentage {percentage} must be greater than "
                f"current {current.percentage}."
            )

        # Enforce stage order
        if current.current_stage is not None:
            current_stage_idx = _stage_index(current.current_stage)
            new_stage_idx = _stage_index(stage)
            if new_stage_idx < current_stage_idx:
                raise ValueError(
                    f"Stage order violation: cannot go from "
                    f"'{current.current_stage.value}' to '{stage.value}'."
                )

        # Determine new state
        new_state = TaskState.PROCESSING

        # Update progress
        updated_progress = TaskProgress(
            task_id=task_id,
            state=new_state,
            current_stage=stage,
            percentage=percentage,
            error_message=None,
            error_stage=None,
        )

        # Check if task is complete (100%)
        now = datetime.now(timezone.utc)
        download_expires_at = None

        if percentage == 100:
            new_state = TaskState.COMPLETED
            updated_progress = TaskProgress(
                task_id=task_id,
                state=TaskState.COMPLETED,
                current_stage=stage,
                percentage=100,
                error_message=None,
                error_stage=None,
            )
            download_expires_at = now + timedelta(hours=DOWNLOAD_EXPIRY_HOURS)

        # Store updated progress in Redis
        status_key = f"task:{task_id}:status"
        try:
            self._redis.set(
                status_key, updated_progress.model_dump_json()
            )
        except Exception as e:
            logger.error("Redis error while updating progress: %s", e)
            raise ConnectionError(
                f"Cannot update task progress in Redis: {e}"
            ) from e

        # Update task metadata
        meta_key = f"task:{task_id}:meta"
        try:
            meta_data = self._redis.get(meta_key)
            if meta_data:
                meta = json.loads(meta_data)
                meta["state"] = new_state if percentage < 100 else TaskState.COMPLETED.value
                meta["updated_at"] = now.isoformat()
                meta["progress"] = json.loads(updated_progress.model_dump_json())
                if download_expires_at:
                    meta["download_expires_at"] = download_expires_at.isoformat()
                self._redis.set(meta_key, json.dumps(meta))
        except Exception as e:
            logger.warning("Failed to update task metadata: %s", e)

        logger.info(
            "Task '%s' progress: stage=%s, percentage=%d%%.",
            task_id,
            stage.value,
            percentage,
        )

    def mark_failed(
        self, task_id: str, error_stage: TaskStage, error_message: str
    ) -> None:
        """Mark a task as failed.

        Args:
            task_id: The unique task identifier.
            error_stage: The stage where failure occurred.
            error_message: Human-readable error description.

        Raises:
            KeyError: If task_id is not found.
            ConnectionError: If Redis is unavailable.
        """
        current = self.get_task_status(task_id)

        failed_progress = TaskProgress(
            task_id=task_id,
            state=TaskState.FAILED,
            current_stage=current.current_stage,
            percentage=current.percentage,
            error_message=error_message,
            error_stage=error_stage,
        )

        status_key = f"task:{task_id}:status"
        now = datetime.now(timezone.utc)

        try:
            self._redis.set(status_key, failed_progress.model_dump_json())
        except Exception as e:
            logger.error("Redis error while marking task failed: %s", e)
            raise ConnectionError(
                f"Cannot mark task as failed in Redis: {e}"
            ) from e

        # Update metadata
        meta_key = f"task:{task_id}:meta"
        try:
            meta_data = self._redis.get(meta_key)
            if meta_data:
                meta = json.loads(meta_data)
                meta["state"] = TaskState.FAILED.value
                meta["updated_at"] = now.isoformat()
                meta["progress"] = json.loads(failed_progress.model_dump_json())
                self._redis.set(meta_key, json.dumps(meta))
        except Exception as e:
            logger.warning("Failed to update task metadata on failure: %s", e)

        logger.warning(
            "Task '%s' failed at stage '%s': %s",
            task_id,
            error_stage.value,
            error_message,
        )

    def get_task_meta(self, task_id: str) -> dict:
        """Retrieve the full task metadata dict from Redis.

        Args:
            task_id: The unique task identifier.

        Returns:
            The parsed metadata dictionary for the task.

        Raises:
            KeyError: If task_id is not found in Redis.
            ConnectionError: If Redis is unavailable.
        """
        meta_key = f"task:{task_id}:meta"
        try:
            data = self._redis.get(meta_key)
        except Exception as e:
            logger.error("Redis error while getting task meta: %s", e)
            raise ConnectionError(
                f"Cannot retrieve task metadata from Redis: {e}"
            ) from e

        if data is None:
            raise KeyError(f"Task '{task_id}' not found.")

        return json.loads(data)

    def delete_task(self, task_id: str) -> None:
        """Delete a task's Redis keys and filesystem artifacts.

        Args:
            task_id: The unique task identifier.

        Raises:
            KeyError: If task_id is not found in Redis.
            ConnectionError: If Redis is unavailable.
        """
        status_key = f"task:{task_id}:status"
        try:
            exists = self._redis.get(status_key) is not None
        except Exception as e:
            logger.error("Redis error while checking task existence: %s", e)
            raise ConnectionError(
                f"Cannot access task store: {e}"
            ) from e

        if not exists:
            raise KeyError(f"Task '{task_id}' not found.")

        self._remove_task(task_id)

    def cleanup_expired_tasks(self, max_age_hours: int = 24) -> int:
        """Remove tasks older than the specified age.

        Deletes both the filesystem artifacts and Redis keys for expired
        tasks.

        Args:
            max_age_hours: Maximum task age in hours before cleanup.
                Defaults to 24 hours.

        Returns:
            Number of tasks cleaned up.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        cleaned = 0

        try:
            active_ids = self._redis.smembers("tasks:active")
        except Exception as e:
            logger.error("Redis error during cleanup: %s", e)
            return 0

        for task_id in active_ids:
            try:
                meta_key = f"task:{task_id}:meta"
                meta_data = self._redis.get(meta_key)

                if meta_data is None:
                    # Orphaned entry — remove from active set
                    self._redis.srem("tasks:active", task_id)
                    cleaned += 1
                    continue

                meta = json.loads(meta_data)
                created_at = datetime.fromisoformat(meta["created_at"])

                if created_at < cutoff:
                    self._remove_task(task_id)
                    cleaned += 1

            except Exception as e:
                logger.warning(
                    "Error cleaning up task '%s': %s", task_id, e
                )

        logger.info("Cleanup complete: removed %d expired tasks.", cleaned)
        return cleaned

    def _store_task_in_redis(self, task: Task) -> None:
        """Store task progress and metadata in Redis."""
        status_key = f"task:{task.task_id}:status"
        meta_key = f"task:{task.task_id}:meta"

        try:
            # Store progress
            self._redis.set(
                status_key, task.progress.model_dump_json()
            )

            # Store full task metadata
            meta = task.model_dump(mode="json")
            self._redis.set(meta_key, json.dumps(meta))

            # Add to active set
            self._redis.sadd("tasks:active", task.task_id)

        except Exception as e:
            logger.error("Redis error while storing task: %s", e)
            raise ConnectionError(
                f"Cannot store task in Redis: {e}"
            ) from e

    def _remove_task(self, task_id: str) -> None:
        """Remove a task's Redis keys and filesystem artifacts."""
        status_key = f"task:{task_id}:status"
        meta_key = f"task:{task_id}:meta"

        # Remove Redis keys
        try:
            self._redis.delete(status_key, meta_key)
            self._redis.srem("tasks:active", task_id)
        except Exception as e:
            logger.warning(
                "Redis error removing task '%s' keys: %s", task_id, e
            )

        # Remove filesystem artifacts
        task_dir = self._tasks_dir / task_id
        if task_dir.exists():
            try:
                shutil.rmtree(task_dir)
                logger.info("Removed task directory: %s", task_dir)
            except OSError as e:
                logger.warning(
                    "Failed to remove task directory '%s': %s", task_dir, e
                )
