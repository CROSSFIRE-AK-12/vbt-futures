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


def test_open_long_consumes_margin() -> None:
    """A single long_entry at bar 0 consumes 1 * price * mult * margin_rate.

    close = [100, 100], long_entries = [True, False].
    At t=0: cash = 10000 - 100*10*0.10 = 9900, position = [1, 0], margin = [100, 0].
    At t=1: nothing changes (no signal, no price move).
    """
    T, N = 2, 1
    close = _const_close(T, N, 100.0)
    long_entries = np.zeros((T, N), dtype=bool)
    long_entries[0, 0] = True
    orders, cash, pos, mrg = simulate_futures_nb(
        close=close, long_entries=long_entries,
        long_exits=_all_false(T, N),
        short_entries=_all_false(T, N), short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0]),
        margin_rate=np.array([0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([2], dtype=np.int8),
        init_cash=10_000.0,
    )
    # At t=0: open long, no price change, no fees, no slippage.
    assert cash[0] == 9_900.0
    assert pos[0, 0] == 1.0
    assert mrg[0, 0] == 100.0
    # At t=1: no signal, no price change, no margin diff.
    assert cash[1] == 9_900.0
    assert pos[1, 0] == 1.0
    assert mrg[1, 0] == 100.0
    # One order recorded: open long at 100.
    assert len(orders) == 1
    assert orders[0]["side"] == 0          # OPEN_LONG
    assert orders[0]["size"] == 1.0
    assert orders[0]["price"] == 100.0
    assert orders[0]["margin"] == 100.0
    assert orders[0]["pnl"] == 0.0
    assert orders[0]["col"] == 0
    assert orders[0]["idx"] == 0
    assert orders[0]["id"] == 0
    assert orders[0]["fees"] == 0.0


def test_mark_to_market_increases_cash_when_price_rises() -> None:
    """Long 1 lot, price moves 100 -> 102.

    t=0: open long at 100 -> cash=9900, mrg=100, pos=1
    t=1: mtm = 1 * (102-100) * 10 = +20 -> cash=9920
         new_margin = 1*102*10*0.10 = 102 -> cash -= 2 -> cash=9918
    """
    T, N = 2, 1
    close = np.array([[100.0], [102.0]])
    long_entries = np.zeros((T, N), dtype=bool)
    long_entries[0, 0] = True
    orders, cash, pos, mrg = simulate_futures_nb(
        close=close, long_entries=long_entries,
        long_exits=_all_false(T, N),
        short_entries=_all_false(T, N), short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0]),
        margin_rate=np.array([0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([2], dtype=np.int8),
        init_cash=10_000.0,
    )
    assert cash[0] == 9_900.0
    assert cash[1] == 9_918.0
    assert pos[1, 0] == 1.0
    assert mrg[1, 0] == 102.0
    # equity = cash + margin_locked should be 10_020 at both ends.
    assert cash[0] + mrg[0, 0] == 10_000.0
    assert cash[1] + mrg[1, 0] == 10_020.0
    assert len(orders) == 1


def test_mark_to_market_decreases_cash_when_price_falls() -> None:
    """Long 1 lot, price moves 100 -> 95.

    t=0: open long, cash=9900, mrg=100
    t=1: mtm = 1 * (95-100) * 10 = -50 -> cash=9850
         new_margin = 95*10*0.10 = 95 -> cash += 5 -> cash=9855
    """
    T, N = 2, 1
    close = np.array([[100.0], [95.0]])
    long_entries = np.zeros((T, N), dtype=bool)
    long_entries[0, 0] = True
    _, cash, _, mrg = simulate_futures_nb(
        close=close, long_entries=long_entries,
        long_exits=_all_false(T, N),
        short_entries=_all_false(T, N), short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0]),
        margin_rate=np.array([0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([2], dtype=np.int8),
        init_cash=10_000.0,
    )
    assert cash[1] == 9_855.0
    assert mrg[1, 0] == 95.0
    # Equity = 9855 + 95 = 9950, which is the original 10000 minus 50 pnl.
    assert cash[1] + mrg[1, 0] == 9_950.0
