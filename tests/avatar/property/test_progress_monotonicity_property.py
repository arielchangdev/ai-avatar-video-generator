"""Property-based test for TaskManager progress stage percentage monotonicity.

Property 7: Progress stage percentage monotonicity
For any sequence of stage transitions that follows the defined stage order,
the progress percentage recorded at each update SHALL be strictly greater than
the percentage of the previous update. Any update whose percentage is not
strictly greater than the current percentage SHALL be rejected.

**Validates: Requirements 7.2**
"""

import sys

sys.path.insert(0, ".")

from pathlib import Path

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis.strategies import lists, integers, composite

from core.task_manager import STAGE_ORDER, TaskManager
from models.schemas import TaskInput, TaskStage, TaskState


# Save builtin set before it can be shadowed
_set = set


class FakeRedis:
    """Minimal in-memory Redis mock."""

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
        self._sets.setdefault(key, _set()).update(values)

    def srem(self, key, *values):
        if key in self._sets:
            self._sets[key] -= _set(values)

    def smembers(self, key):
        return self._sets.get(key, _set()).copy()


def _make_task_manager(tmp_path: Path) -> TaskManager:
    return TaskManager(redis_client=FakeRedis(), output_dir=tmp_path / "out")


def _make_input() -> TaskInput:
    return TaskInput(
        voice_sample=b"voice-bytes",
        voice_sample_filename="v.wav",
        appearance_asset=b"image-bytes",
        appearance_asset_filename="f.jpg",
        script_text="hello",
    )


@composite
def strictly_increasing_percentages(draw):
    """Generate a strictly increasing list of percentages in [1, 100]."""
    deltas = draw(lists(integers(min_value=1, max_value=20), min_size=1, max_size=8))
    seq = []
    total = 0
    for d in deltas:
        total += d
        if total > 100:
            break
        seq.append(total)
    # Ensure at least one value
    if not seq:
        seq = [draw(integers(min_value=1, max_value=100))]
    return seq


def _stage_for_percentage(pct: int) -> TaskStage:
    """Map a percentage bucket onto a stage in defined order (non-decreasing)."""
    idx = min(pct * len(STAGE_ORDER) // 101, len(STAGE_ORDER) - 1)
    return STAGE_ORDER[idx]


@pytest.mark.property
class TestProgressStageMonotonicity:
    """Property 7: Progress stage percentage monotonicity.

    **Validates: Requirements 7.2**
    """

    @given(percentages=strictly_increasing_percentages())
    @settings(
        max_examples=150,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def test_strictly_increasing_sequence_is_accepted_and_monotonic(
        self, percentages, tmp_path
    ):
        """A strictly increasing percentage sequence (with non-decreasing stages)
        is accepted, and the stored percentage is monotonic increasing."""
        tm = _make_task_manager(tmp_path)
        task = tm.create_task(_make_input())

        prev = 0
        for pct in percentages:
            stage = _stage_for_percentage(pct)
            tm.update_progress(task.task_id, stage, pct)
            progress = tm.get_task_status(task.task_id)
            assert progress.percentage == pct
            assert progress.percentage > prev, (
                f"Percentage {progress.percentage} not strictly greater than "
                f"previous {prev}"
            )
            prev = pct

    @given(percentages=strictly_increasing_percentages())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def test_repeating_percentage_is_rejected(self, percentages, tmp_path):
        """Re-applying the same percentage (non-strict) is always rejected."""
        tm = _make_task_manager(tmp_path)
        task = tm.create_task(_make_input())

        first = percentages[0]
        stage = _stage_for_percentage(first)
        tm.update_progress(task.task_id, stage, first)

        # Applying the same percentage again must be rejected
        with pytest.raises(ValueError, match="strictly monotonic"):
            tm.update_progress(task.task_id, stage, first)

    @given(
        first=integers(min_value=2, max_value=100),
        drop=integers(min_value=1, max_value=99),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def test_decreasing_percentage_is_rejected(self, first, drop, tmp_path):
        """Any percentage lower than the current one is rejected."""
        lower = first - drop
        if lower < 0:
            lower = 0
        tm = _make_task_manager(tmp_path)
        task = tm.create_task(_make_input())

        stage = _stage_for_percentage(first)
        tm.update_progress(task.task_id, stage, first)

        with pytest.raises(ValueError):
            tm.update_progress(task.task_id, stage, lower)
