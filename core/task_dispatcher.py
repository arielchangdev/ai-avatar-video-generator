"""AI Avatar Video Generator - Task dispatcher abstraction.

Decouples the API layer from the concrete background execution backend
(Celery). The API submits work through a TaskDispatcher; the Celery worker
(task 14) provides the real implementation. Tests use an in-memory dispatcher.
"""

from __future__ import annotations

from typing import Protocol

from models.schemas import TaskStage


class TaskDispatcher(Protocol):
    """Protocol for dispatching generation work to a background worker."""

    def dispatch(
        self, task_id: str, retry_from_stage: TaskStage | None = None
    ) -> None:
        """Enqueue the generation pipeline for a created task.

        Args:
            task_id: The identifier of an already-created task.
            retry_from_stage: If provided, resume the pipeline from this stage
                (skipping stages that completed on a previous attempt).
        """
        ...


class InMemoryTaskDispatcher:
    """A no-op dispatcher that records dispatches.

    Useful for unit tests and running the API without a live Celery worker.
    """

    def __init__(self) -> None:
        self.dispatched: list[tuple[str, TaskStage | None]] = []

    def dispatch(
        self, task_id: str, retry_from_stage: TaskStage | None = None
    ) -> None:
        self.dispatched.append((task_id, retry_from_stage))


class CeleryTaskDispatcher:
    """Dispatches work to the Celery pipeline task (task 14).

    Imports Celery lazily so the API layer does not require Celery to be
    installed for import-time use (e.g. unit tests with a fake dispatcher).
    """

    def dispatch(
        self, task_id: str, retry_from_stage: TaskStage | None = None
    ) -> None:
        from core.pipeline import generate_video_task  # lazy import

        generate_video_task.delay(
            task_id,
            retry_from_stage.value if retry_from_stage else None,
        )
