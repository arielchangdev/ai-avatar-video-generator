"""Unit tests for the pipeline orchestration error handling (task 14.2).

Validates Requirements 4.7, 5.6, 6.7, 7.3:
- Pipeline stops on VoiceSynthesisError
- Pipeline stops on LipSyncError
- Pipeline preserves earlier results on VideoComposeError
- Retry starts from the correct failed stage
- Progress updates are monotonic through the pipeline

Uses fake stage runners and an in-memory TaskManager so no AI models, Redis,
or Celery are required.
"""

import sys

sys.path.insert(0, ".")

import pytest

from core.exceptions import (
    LipSyncError,
    VideoComposeError,
    VoiceSynthesisError,
)
from core.pipeline import PIPELINE_STAGES, run_pipeline
from models.schemas import TaskStage


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTaskManager:
    """Records progress updates and failures; provides a no-op Redis."""

    class _NoOpRedis:
        def publish(self, *_args, **_kwargs):
            return 0

        def get(self, *_args, **_kwargs):
            return None

        def set(self, *_args, **_kwargs):
            return True

    def __init__(self):
        self.updates: list[tuple[TaskStage, int]] = []
        self.failures: list[tuple[TaskStage, str]] = []
        self._redis = self._NoOpRedis()
        self._status = None

    def update_progress(self, task_id, stage, percentage):
        # Enforce strict monotonicity like the real manager.
        if self.updates and percentage <= self.updates[-1][1]:
            raise ValueError("Progress must be strictly monotonic")
        self.updates.append((stage, percentage))

    def mark_failed(self, task_id, error_stage, error_message):
        self.failures.append((error_stage, error_message))

    def get_task_status(self, task_id):
        from models.schemas import TaskProgress, TaskState

        stage, pct = self.updates[-1] if self.updates else (None, 0)
        return TaskProgress(
            task_id=task_id,
            state=TaskState.PROCESSING,
            current_stage=stage,
            percentage=pct,
            error_message=None,
            error_stage=None,
        )


def _ok_runner(name):
    def runner(ctx):
        return {f"{name}_done": True}

    return runner


def _all_ok_runners(record=None):
    runners = {}
    for plan in PIPELINE_STAGES:
        key = plan.key

        def make(k):
            def runner(ctx):
                if record is not None:
                    record.append(k)
                return {f"{k}_done": True}

            return runner

        runners[key] = make(key)
    return runners


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSuccessPath:
    def test_full_pipeline_runs_all_stages_monotonically(self):
        tm = FakeTaskManager()
        executed = []
        result = run_pipeline("t1", _all_ok_runners(executed), tm)

        # All five stages ran in order
        assert executed == ["script", "voice", "face", "lipsync", "compose"]

        # Progress is strictly monotonic and ends at 100
        percentages = [p for _, p in tm.updates]
        assert percentages == [10, 40, 55, 80, 100]
        assert percentages == sorted(percentages)
        assert percentages[-1] == 100
        assert result["compose_done"] is True


class TestErrorHandling:
    def test_stops_on_voice_synthesis_error(self):
        tm = FakeTaskManager()
        runners = _all_ok_runners()

        def failing_voice(ctx):
            raise VoiceSynthesisError(
                stage=TaskStage.VOICE_SYNTHESIS,
                reason="GPT-SoVITS crashed",
                recoverable=True,
            )

        runners["voice"] = failing_voice

        with pytest.raises(VoiceSynthesisError):
            run_pipeline("t1", runners, tm)

        # Only the script stage completed (10%); voice failed before progress.
        assert tm.updates == [(TaskStage.VOICE_ANALYSIS, 10)]
        assert tm.failures == [(TaskStage.VOICE_SYNTHESIS, "GPT-SoVITS crashed")]

    def test_stops_on_lip_sync_error(self):
        tm = FakeTaskManager()
        runners = _all_ok_runners()

        def failing_lipsync(ctx):
            raise LipSyncError(
                stage=TaskStage.LIP_SYNC,
                reason="SadTalker failed",
                recoverable=True,
            )

        runners["lipsync"] = failing_lipsync

        with pytest.raises(LipSyncError):
            run_pipeline("t1", runners, tm)

        # script(10), voice(40), face(55) completed; lipsync failed.
        assert [p for _, p in tm.updates] == [10, 40, 55]
        assert tm.failures == [(TaskStage.LIP_SYNC, "SadTalker failed")]

    def test_preserves_progress_on_compose_error(self):
        tm = FakeTaskManager()
        runners = _all_ok_runners()

        def failing_compose(ctx):
            raise VideoComposeError(
                stage=TaskStage.VIDEO_COMPOSE,
                reason="FFmpeg mux failed",
                recoverable=True,
            )

        runners["compose"] = failing_compose

        with pytest.raises(VideoComposeError):
            run_pipeline("t1", runners, tm)

        # Everything up to compose completed (80%); compose never hit 100%.
        assert [p for _, p in tm.updates] == [10, 40, 55, 80]
        assert tm.failures == [(TaskStage.VIDEO_COMPOSE, "FFmpeg mux failed")]


class TestRetryFromStage:
    def test_retry_skips_completed_stages(self):
        tm = FakeTaskManager()
        executed = []
        runners = _all_ok_runners(executed)

        run_pipeline("t1", runners, tm, retry_from_stage=TaskStage.LIP_SYNC)

        # Only lipsync and compose ran; earlier stages skipped.
        assert executed == ["lipsync", "compose"]
        # Progress starts at the lipsync percentage.
        assert [p for _, p in tm.updates] == [80, 100]

    def test_retry_from_face_stage(self):
        tm = FakeTaskManager()
        executed = []
        runners = _all_ok_runners(executed)

        run_pipeline("t1", runners, tm, retry_from_stage=TaskStage.FACE_MODELING)

        assert executed == ["face", "lipsync", "compose"]
        assert [p for _, p in tm.updates] == [55, 80, 100]

    def test_retry_from_none_runs_all(self):
        tm = FakeTaskManager()
        executed = []
        runners = _all_ok_runners(executed)

        run_pipeline("t1", runners, tm, retry_from_stage=None)
        assert executed == ["script", "voice", "face", "lipsync", "compose"]


class TestMissingRunner:
    def test_missing_runner_raises_key_error(self):
        tm = FakeTaskManager()
        runners = _all_ok_runners()
        del runners["voice"]

        with pytest.raises(KeyError):
            run_pipeline("t1", runners, tm)
