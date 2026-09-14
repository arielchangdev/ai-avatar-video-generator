"""AI Avatar Video Generator - Task REST API routes.

Implements the task lifecycle endpoints:
    POST   /api/v1/tasks                       建立生成任務 (multipart form)
    GET    /api/v1/tasks/{task_id}             查詢任務狀態
    GET    /api/v1/tasks/{task_id}/download/video  下載影片 (含過期檢查)
    GET    /api/v1/tasks/{task_id}/download/audio  下載音訊 (含過期檢查)
    DELETE /api/v1/tasks/{task_id}             刪除任務及相關檔案
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse

from api.dependencies import get_task_dispatcher, get_task_manager
from core.task_dispatcher import TaskDispatcher
from core.task_manager import TaskManager
from core.validators.submission_validator import SubmissionValidator
from models.schemas import TaskInput, TaskStage, TaskState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

# No data received for this many seconds during upload -> 408
UPLOAD_TIMEOUT_SEC = 120


def _map_retry_stage(value: str | None) -> TaskStage | None:
    """Convert an optional retry-stage string into a TaskStage."""
    if value is None:
        return None
    try:
        return TaskStage(value)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid retry_from_stage: {value!r}",
        )


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    voice_sample: UploadFile = File(...),
    appearance_asset: UploadFile = File(...),
    # Default to empty so an empty script is not rejected by FastAPI field
    # validation (422) but flows to the submission validator, which is the
    # single source of truth for reporting missing/invalid inputs (400).
    script_text: str = Form(""),
    retry_from_stage: str | None = Form(None),
    task_manager: TaskManager = Depends(get_task_manager),
    dispatcher: TaskDispatcher = Depends(get_task_dispatcher),
) -> JSONResponse:
    """建立生成任務。

    Accepts a multipart form with the voice sample, appearance asset, and
    script text. Runs the submission validator, persists the task, and
    dispatches the generation pipeline.
    """
    # Read uploads with a timeout so a stalled client cannot block forever.
    try:
        voice_bytes = await asyncio.wait_for(
            voice_sample.read(), timeout=UPLOAD_TIMEOUT_SEC
        )
        appearance_bytes = await asyncio.wait_for(
            appearance_asset.read(), timeout=UPLOAD_TIMEOUT_SEC
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Upload timed out (no data within 120 seconds).",
        )

    retry_stage = _map_retry_stage(retry_from_stage)

    # Create the task first so the validator can inspect real files on disk.
    task_input = TaskInput(
        voice_sample=voice_bytes,
        voice_sample_filename=voice_sample.filename or "voice_sample",
        appearance_asset=appearance_bytes,
        appearance_asset_filename=appearance_asset.filename or "appearance_asset",
        script_text=script_text,
    )

    try:
        task = task_manager.create_task(task_input)
    except ConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Task store unavailable: {e}",
        )

    validator = SubmissionValidator()
    result = validator.validate(
        task.voice_sample_path,
        task.appearance_asset_path,
        script_text,
    )

    if not result.is_valid:
        # Roll back the just-created task so invalid submissions leave no trace.
        try:
            task_manager.delete_task(task.task_id)
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.warning("Failed to roll back invalid task %s", task.task_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"errors": result.errors},
        )

    dispatcher.dispatch(task.task_id, retry_stage)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "task_id": task.task_id,
            "state": task.state.value,
            "summary": result.summary.model_dump() if result.summary else None,
        },
    )


@router.get("/{task_id}")
def get_task(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager),
) -> dict:
    """查詢任務狀態，回傳目前的 TaskProgress。"""
    try:
        progress = task_manager.get_task_status(task_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )
    except ConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Task store unavailable: {e}",
        )
    return json.loads(progress.model_dump_json())


def _resolve_download(
    task_manager: TaskManager, task_id: str, kind: str
) -> tuple[Path, str, str]:
    """Resolve a downloadable artifact, enforcing existence and expiry.

    Returns (path, media_type, filename). Raises HTTPException on any failure.
    """
    try:
        meta = task_manager.get_task_meta(task_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )

    # Expiry check
    expires_raw = meta.get("download_expires_at")
    if not expires_raw:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Download is not available for this task.",
        )
    expires_at = datetime.fromisoformat(expires_raw)
    if datetime.now(timezone.utc) >= expires_at:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Download link has expired (24-hour validity elapsed).",
        )

    if kind == "video":
        path_str = meta.get("output_video_path")
        media_type, filename = "video/mp4", "avatar.mp4"
    else:
        path_str = meta.get("output_audio_path")
        media_type, filename = "audio/mpeg", "avatar.mp3"

    if not path_str or not Path(path_str).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{kind.capitalize()} artifact not found.",
        )

    return Path(path_str), media_type, filename


@router.get("/{task_id}/download/video")
def download_video(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager),
) -> FileResponse:
    """下載最終 MP4 影片（含 24 小時過期檢查）。"""
    path, media_type, filename = _resolve_download(task_manager, task_id, "video")
    return FileResponse(path, media_type=media_type, filename=filename)


@router.get("/{task_id}/download/audio")
def download_audio(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager),
) -> FileResponse:
    """下載最終 MP3 音訊（含 24 小時過期檢查）。"""
    path, media_type, filename = _resolve_download(task_manager, task_id, "audio")
    return FileResponse(path, media_type=media_type, filename=filename)


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager),
):
    """刪除任務目錄與 Redis 鍵值。"""
    try:
        task_manager.delete_task(task_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)
