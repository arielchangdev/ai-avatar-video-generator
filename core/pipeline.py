"""AI Avatar Video Generator - Processing pipeline orchestration.

Runs the end-to-end generation pipeline in order:
    1. Script processing   -> VOICE_ANALYSIS stage,  10%
    2. Voice synthesis     -> VOICE_SYNTHESIS stage, 40%
    3. Face modeling       -> FACE_MODELING stage,   55%
    4. Lip sync            -> LIP_SYNC stage,         80%
    5. Video composition   -> VIDEO_COMPOSE stage,   100%

The orchestration logic (``run_pipeline``) is deliberately decoupled from
Celery so it can be unit-tested directly. The Celery task (``generate_video_task``)
is a thin wrapper that is only defined when Celery is available.

Progress and failures are reported through the TaskManager, which also
publishes updates to the Redis pub/sub channel consumed by the WebSocket
handler.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.exceptions import (
    LipSyncError,
    PipelineError,
    VideoComposeError,
    VoiceSynthesisError,
)
from core.task_manager import TaskManager
from models.schemas import ErrorNotification, TaskStage

logger = logging.getLogger(__name__)


# Ordered pipeline stages with their target completion percentage.
# ``key`` is a stable identifier used for retry-from-stage / skip logic.
@dataclass(frozen=True)
class StagePlan:
    key: str
    stage: TaskStage
    percentage: int


PIPELINE_STAGES: list[StagePlan] = [
    StagePlan("script", TaskStage.VOICE_ANALYSIS, 10),
    StagePlan("voice", TaskStage.VOICE_SYNTHESIS, 40),
    StagePlan("face", TaskStage.FACE_MODELING, 55),
    StagePlan("lipsync", TaskStage.LIP_SYNC, 80),
    StagePlan("compose", TaskStage.VIDEO_COMPOSE, 100),
]

STAGE_KEYS = [s.key for s in PIPELINE_STAGES]


def _stage_start_index(retry_from_stage: TaskStage | None) -> int:
    """Return the index of the first stage to execute.

    On retry, stages before the failed one are skipped.
    """
    if retry_from_stage is None:
        return 0
    for i, plan in enumerate(PIPELINE_STAGES):
        if plan.stage == retry_from_stage:
            return i
    return 0


def _publish_error(task_manager: TaskManager, task_id: str, err: PipelineError) -> None:
    """Publish an ErrorNotification to the task's progress channel."""
    notification = ErrorNotification(
        task_id=task_id,
        stage=err.stage,
        error_category=type(err).__name__,
        message=err.reason,
        recoverable=err.recoverable,
        retry_from_stage=err.stage if err.recoverable else None,
    )
    try:
        task_manager._redis.publish(
            f"task:{task_id}:progress", notification.model_dump_json()
        )
    except Exception as e:  # pragma: no cover - best effort
        logger.warning("Failed to publish error notification: %s", e)


def run_pipeline(
    task_id: str,
    stage_runners: dict[str, Callable[[dict], dict]],
    task_manager: TaskManager,
    retry_from_stage: TaskStage | None = None,
) -> dict:
    """Execute the generation pipeline for a task.

    Args:
        task_id: The created task's identifier.
        stage_runners: Mapping of stage key -> callable. Each callable receives
            the accumulated context dict and returns updates to merge into it.
            This indirection keeps the orchestration testable and lets the
            Celery task wire in the real AI modules.
        task_manager: TaskManager used to record progress and failures.
        retry_from_stage: If provided, resume from this stage (skip earlier ones).

    Returns:
        The accumulated context dict on success.

    Raises:
        PipelineError: Re-raised after the task is marked failed, so the caller
            (Celery) can apply its own retry policy.
    """
    start = _stage_start_index(retry_from_stage)
    context: dict = {"task_id": task_id}

    for plan in PIPELINE_STAGES[start:]:
        runner = stage_runners.get(plan.key)
        if runner is None:
            raise KeyError(f"No runner registered for stage '{plan.key}'.")

        try:
            updates = runner(context)
            if updates:
                context.update(updates)
        except PipelineError as err:
            logger.warning(
                "Pipeline for task '%s' failed at stage '%s': %s",
                task_id,
                plan.stage.value,
                err.reason,
            )
            task_manager.mark_failed(task_id, err.stage, err.reason)
            _publish_error(task_manager, task_id, err)
            raise

        # Report progress only after the stage completes successfully.
        task_manager.update_progress(task_id, plan.stage, plan.percentage)
        try:
            progress = task_manager.get_task_status(task_id)
            task_manager._redis.publish(
                f"task:{task_id}:progress",
                json.dumps(
                    {"type": "progress", **json.loads(progress.model_dump_json())}
                ),
            )
        except Exception:  # pragma: no cover - best effort
            pass

    return context


# ---------------------------------------------------------------------------
# Celery task wrapper (only defined when Celery is installed)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised only in a full deployment
    from core.celery_app import celery_app

    @celery_app.task(name="avatar.generate_video", bind=True)
    def generate_video_task(self, task_id: str, retry_from_stage: str | None = None):
        """Celery entry point that wires the real AI modules into the pipeline."""
        from core.stage_runners import build_default_runners  # lazy import

        task_manager = TaskManager()
        stage = TaskStage(retry_from_stage) if retry_from_stage else None
        runners = build_default_runners(task_manager)
        return run_pipeline(task_id, runners, task_manager, stage)

except Exception:  # Celery not installed / not configured
    generate_video_task = None  # type: ignore[assignment]
