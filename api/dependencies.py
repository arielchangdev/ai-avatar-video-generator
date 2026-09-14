"""AI Avatar Video Generator - Shared FastAPI dependencies.

Provides injectable singletons (TaskManager, TaskDispatcher) that can be
overridden in tests via ``app.dependency_overrides``.
"""

from __future__ import annotations

from functools import lru_cache

from core.task_dispatcher import CeleryTaskDispatcher, TaskDispatcher
from core.task_manager import TaskManager


@lru_cache(maxsize=1)
def get_task_manager() -> TaskManager:
    """Return a process-wide TaskManager instance."""
    return TaskManager()


@lru_cache(maxsize=1)
def get_task_dispatcher() -> TaskDispatcher:
    """Return a process-wide TaskDispatcher instance."""
    return CeleryTaskDispatcher()
