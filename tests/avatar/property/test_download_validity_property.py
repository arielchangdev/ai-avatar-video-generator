"""Property-based test for download link temporal validity.

Property 8: Download link temporal validity
A download link created when a task completes SHALL be valid if and only if
the time elapsed between completion and access is strictly less than 86400
seconds (24 hours).

**Validates: Requirements 7.4**
"""

import sys

sys.path.insert(0, ".")

from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers, floats

from core.task_manager import DOWNLOAD_EXPIRY_HOURS


DOWNLOAD_VALIDITY_SECONDS = 86400  # 24 hours


def is_download_valid(completion_time: datetime, access_time: datetime) -> bool:
    """Return True iff the download link is still valid at access_time.

    Mirrors TaskManager behaviour: expiry = completion + DOWNLOAD_EXPIRY_HOURS.
    A link is valid iff access_time < expiry, i.e. the elapsed time since
    completion is strictly less than 86400 seconds.
    """
    expires_at = completion_time + timedelta(hours=DOWNLOAD_EXPIRY_HOURS)
    return access_time < expires_at


@pytest.mark.property
class TestDownloadLinkTemporalValidity:
    """Property 8: Download link temporal validity.

    **Validates: Requirements 7.4**
    """

    def test_expiry_constant_is_24_hours(self):
        """The configured expiry window equals 86400 seconds."""
        assert DOWNLOAD_EXPIRY_HOURS * 3600 == DOWNLOAD_VALIDITY_SECONDS

    @given(elapsed_seconds=integers(min_value=-1_000_000, max_value=1_000_000))
    @settings(max_examples=300)
    def test_valid_iff_elapsed_under_86400(self, elapsed_seconds):
        """For any elapsed offset, the link is valid iff elapsed < 86400 seconds."""
        completion = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        access = completion + timedelta(seconds=elapsed_seconds)

        result = is_download_valid(completion, access)
        expected = elapsed_seconds < DOWNLOAD_VALIDITY_SECONDS

        assert result == expected, (
            f"elapsed={elapsed_seconds}s: is_download_valid returned {result}, "
            f"expected {expected}"
        )

    @given(elapsed_seconds=floats(
        min_value=0.0, max_value=86399.0, allow_nan=False, allow_infinity=False
    ))
    @settings(max_examples=150)
    def test_within_window_always_valid(self, elapsed_seconds):
        """Access strictly within the 24h window is always valid."""
        completion = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        access = completion + timedelta(seconds=elapsed_seconds)
        assert is_download_valid(completion, access) is True

    @given(elapsed_seconds=floats(
        min_value=86400.0, max_value=500000.0, allow_nan=False, allow_infinity=False
    ))
    @settings(max_examples=150)
    def test_at_or_after_window_always_invalid(self, elapsed_seconds):
        """Access at or after exactly 24h is always invalid."""
        completion = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        access = completion + timedelta(seconds=elapsed_seconds)
        assert is_download_valid(completion, access) is False

    def test_exact_boundary_is_invalid(self):
        """At exactly 86400 seconds the link is expired (strict inequality)."""
        completion = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        access = completion + timedelta(seconds=86400)
        assert is_download_valid(completion, access) is False
