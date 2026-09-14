"""Property-based test for should_timeout composition timeout decision.

Property 6: Composition timeout decision correctness
For any (video_duration_sec, elapsed_sec) pair, should_timeout SHALL return True
if and only if elapsed_sec > video_duration_sec * 3 AND elapsed_sec > 180.

**Validates: Requirements 6.8**
"""

import sys

sys.path.insert(0, ".")

import pytest
from hypothesis import given, settings
from hypothesis.strategies import floats

from modules.video_composer import should_timeout


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Duration of source video: 0.1s to 10000s
video_duration_strategy = floats(
    min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False
)

# Elapsed composition time: 0.0s to 100000s
elapsed_time_strategy = floats(
    min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False
)


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------


@pytest.mark.property
class TestCompositionTimeoutDecisionCorrectness:
    """Property 6: Composition timeout decision correctness.

    **Validates: Requirements 6.8**
    """

    @given(
        video_duration=video_duration_strategy,
        elapsed=elapsed_time_strategy,
    )
    @settings(max_examples=200)
    def test_timeout_iff_both_conditions_met(self, video_duration, elapsed):
        """For random (video_duration, elapsed) pairs, should_timeout returns
        True iff (elapsed > duration * 3 AND elapsed > 180)."""
        result = should_timeout(video_duration, elapsed)
        expected = (elapsed > video_duration * 3) and (elapsed > 180)

        assert result == expected, (
            f"should_timeout({video_duration}, {elapsed}) = {result}, "
            f"expected {expected}. "
            f"elapsed > duration*3: {elapsed > video_duration * 3}, "
            f"elapsed > 180: {elapsed > 180}"
        )

    @given(
        video_duration=video_duration_strategy,
        elapsed=floats(
            min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100)
    def test_elapsed_lte_180_always_false(self, video_duration, elapsed):
        """When elapsed <= 180, should_timeout is always False regardless of duration."""
        result = should_timeout(video_duration, elapsed)

        if elapsed <= 180:
            assert result is False, (
                f"should_timeout({video_duration}, {elapsed}) should be False "
                f"when elapsed <= 180, but got True"
            )

    @given(
        video_duration=video_duration_strategy,
        elapsed=elapsed_time_strategy,
    )
    @settings(max_examples=100)
    def test_elapsed_lte_duration_times_3_always_false(self, video_duration, elapsed):
        """When elapsed <= duration * 3, should_timeout is always False."""
        threshold = video_duration * 3
        if elapsed <= threshold:
            result = should_timeout(video_duration, elapsed)
            assert result is False, (
                f"should_timeout({video_duration}, {elapsed}) should be False "
                f"when elapsed ({elapsed}) <= duration*3 ({threshold}), but got True"
            )

    @given(
        video_duration=floats(
            min_value=0.1, max_value=59.0, allow_nan=False, allow_infinity=False
        ),
        elapsed=floats(
            min_value=180.1, max_value=100000.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100)
    def test_both_conditions_met_always_true(self, video_duration, elapsed):
        """When BOTH conditions met (elapsed > duration * 3 AND elapsed > 180),
        should_timeout always returns True."""
        # Ensure both conditions are met
        if elapsed > video_duration * 3 and elapsed > 180:
            result = should_timeout(video_duration, elapsed)
            assert result is True, (
                f"should_timeout({video_duration}, {elapsed}) should be True "
                f"when elapsed > duration*3 ({video_duration * 3}) AND elapsed > 180, "
                f"but got False"
            )
