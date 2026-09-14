"""Pytest configuration for project-level test settings.

Inserting the repository root onto sys.path at import time (this file lives at
the repo root and pytest imports it before collecting any tests) guarantees
that ``from core ...`` / ``from models ...`` resolve on any runner, regardless
of the working directory or pytest import mode.
"""

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def pytest_configure(config):
    """Register custom markers (supplements pyproject.toml markers)."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
