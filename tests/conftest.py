"""Pytest configuration and shared fixtures for vbt-futures.

The session-scoped autouse fixture below clears all byte-code / numba caches
under the project root before every ``pytest`` invocation.  This guarantees
that tests always pick up the latest version of the simulator even after
edits (otherwise a stale ``__pycache__/*.nbi`` can mask real bugs).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

# Project root = parent of the tests/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def _clear_caches() -> None:
    """Wipe all ``__pycache__`` directories and numba cache files under the
    project root before the test session starts.

    Safe to run repeatedly: missing directories are silently skipped.
    """
    # 1. Remove every __pycache__ directory anywhere under the project root.
    for cache_dir in PROJECT_ROOT.rglob("__pycache__"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)

    # 2. Remove top-level numba cache artefacts.
    for pattern in ("*.nbi", "*.nbc"):
        for f in PROJECT_ROOT.rglob(pattern):
            try:
                f.unlink()
            except OSError:  # pragma: no cover
                pass

    # 3. Remove pytest's htmlcov / .pytest_cache if they exist (cosmetic only).
    for sub in ("htmlcov", ".pytest_cache"):
        p = PROJECT_ROOT / sub
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
