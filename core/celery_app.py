"""AI Avatar Video Generator - Celery application configuration.

Configures Celery with Redis as both broker and result backend. This module
imports Celery at top level, so importing it will fail if Celery is not
installed; callers (e.g. core.pipeline) guard the import accordingly.
"""

from __future__ import annotations

from celery import Celery

from config.settings import settings

celery_app = Celery(
    "avatar",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    # GPU pipeline is long-running; give generous soft/hard time limits.
    task_soft_time_limit=60 * 30,
    task_time_limit=60 * 35,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# Ensure the task module is imported so the worker registers the task.
# The generation task lives in core.pipeline; list it explicitly so the worker
# imports and registers it on startup.
celery_app.conf.update(imports=("core.pipeline",))
