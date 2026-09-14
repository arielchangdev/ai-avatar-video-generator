"""AI Avatar Video Generator - WebSocket progress handler.

Implements:
    WS /api/v1/ws/tasks/{task_id}/progress   即時進度推送

Behaviour:
    - On connect, immediately send the current task state (supports reconnection).
    - Subscribe to a Redis pub/sub channel for the task and forward updates.
    - Push a heartbeat with the latest progress at least every 3 seconds so the
      client always sees fresh state even without new pipeline events.
    - On failure, the pipeline publishes an ErrorNotification which is forwarded.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from api.dependencies import get_task_manager
from core.task_manager import TaskManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ws", tags=["websocket"])

# Maximum interval between progress pushes (requirement: <= 3 seconds).
HEARTBEAT_INTERVAL_SEC = 3.0


def progress_channel(task_id: str) -> str:
    """Redis pub/sub channel name for a task's progress updates."""
    return f"task:{task_id}:progress"


async def _send_current_state(
    websocket: WebSocket, task_manager: TaskManager, task_id: str
) -> bool:
    """Send the current task progress. Returns False if the task is unknown."""
    try:
        progress = task_manager.get_task_status(task_id)
    except KeyError:
        await websocket.send_json(
            {"type": "error", "message": f"Task '{task_id}' not found."}
        )
        return False
    except ConnectionError as e:
        await websocket.send_json(
            {"type": "error", "message": f"Task store unavailable: {e}"}
        )
        return False

    await websocket.send_json(
        {"type": "progress", **json.loads(progress.model_dump_json())}
    )
    return True


@router.websocket("/tasks/{task_id}/progress")
async def task_progress(
    websocket: WebSocket,
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager),
) -> None:
    """Stream real-time progress for a task over a WebSocket."""
    await websocket.accept()

    # Send current state immediately (handles client reconnection).
    ok = await _send_current_state(websocket, task_manager, task_id)
    if not ok:
        await websocket.close()
        return

    pubsub = None
    try:
        redis_client = task_manager._redis  # shared client
        pubsub = redis_client.pubsub()
        pubsub.subscribe(progress_channel(task_id))

        while True:
            # Non-blocking poll for a published message.
            message = await asyncio.to_thread(
                pubsub.get_message,
                ignore_subscribe_messages=True,
                timeout=HEARTBEAT_INTERVAL_SEC,
            )

            if message is not None and message.get("type") == "message":
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                await websocket.send_text(data)
            else:
                # Heartbeat: no event within the interval, push latest state so
                # the client receives an update at least every 3 seconds.
                try:
                    progress = task_manager.get_task_status(task_id)
                    await websocket.send_json(
                        {"type": "progress", **json.loads(progress.model_dump_json())}
                    )
                except KeyError:
                    break

    except WebSocketDisconnect:
        logger.info("Client disconnected from task '%s' progress stream.", task_id)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("WebSocket error for task '%s': %s", task_id, e)
    finally:
        if pubsub is not None:
            try:
                pubsub.unsubscribe(progress_channel(task_id))
                pubsub.close()
            except Exception:
                pass
