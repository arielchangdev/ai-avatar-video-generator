"""Pytest configuration for project-level test settings."""

import pytest


def pytest_configure(config):
    """Register custom markers (supplements pyproject.toml markers)."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
