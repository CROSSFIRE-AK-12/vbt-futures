"""Tests for src/vbt_futures/simulator.py.

This file accumulates per-feature tests as Tasks 8-20 land.
"""
from __future__ import annotations

import numpy as np
import pytest

from vbt_futures.simulator import simulate_futures_nb


def _all_false(T: int, N: int) -> np.ndarray:
    return np.zeros((T, N), dtype=bool)


def _const_close(T: int, N: int, value: float) -> np.ndarray:
    return np.full((T, N), value, dtype=np.float64)


def test_no_signals_returns_init_state() -> None:
    """With all signals False, cash, position, margin_locked, orders are
    unchanged from initial state at every bar."""
    T, N = 5, 2
    close = _const_close(T, N, 100.0)
    orders, cash, pos, mrg = simulate_futures_nb(
        close=close,
        long_entries=_all_false(T, N), long_exits=_all_false(T, N),
        short_entries=_all_false(T, N), short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0, 10.0]),
        margin_rate=np.array([0.10, 0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([2, 2], dtype=np.int8),
        init_cash=50_000.0,
    )
    assert cash.shape == (T,)
    assert pos.shape == (T, N)
    assert mrg.shape == (T, N)
    assert len(orders) == 0
    assert np.all(cash == 50_000.0)
    assert np.all(pos == 0.0)
    assert np.all(mrg == 0.0)
