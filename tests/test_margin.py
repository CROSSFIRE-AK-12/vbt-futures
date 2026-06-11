"""Focused tests for margin dynamics (multi-column, invariants)."""
from __future__ import annotations

import numpy as np
import pytest

from vbt_futures.simulator import simulate_futures_nb


def _all_false(T: int, N: int) -> np.ndarray:
    return np.zeros((T, N), dtype=bool)


def _const_close(T: int, N: int, value: float) -> np.ndarray:
    return np.full((T, N), value, dtype=np.float64)


def test_margin_increases_with_price_move() -> None:
    """Long 1 lot, price rises 100 -> 110, margin rate 10%.
    At t=0 margin = 1*100*10*0.1 = 100.  At t=1 margin = 1*110*10*0.1 = 110.
    Cash drops by 10 (the 10 difference) at t=1's STEP 3.
    """
    T, N = 2, 1
    close = np.array([[100.0], [110.0]])
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
    assert mrg[0, 0] == 100.0
    assert mrg[1, 0] == 110.0
    # t=0 cash=9900; t=1 mtm 1*(110-100)*10=100 -> 10000; mrg diff 10 -> 9990
    assert cash[1] == pytest.approx(9_990.0)


def test_margin_decreases_with_price_move() -> None:
    """Long 1 lot, price falls 100 -> 90.  At t=1 mrg = 1*90*10*0.1 = 90.
    Cash gains 10 from margin release (partially offset by -100 mtm).
    """
    T, N = 2, 1
    close = np.array([[100.0], [90.0]])
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
    assert mrg[1, 0] == 90.0
    # t=0 cash=9900; t=1 mtm -100 -> 9800; mrg diff -10 -> 9810
    assert cash[1] == pytest.approx(9_810.0)


def test_zero_position_zero_margin_invariant() -> None:
    """When position==0, margin_locked must also be 0 (no margin without a position)."""
    T, N = 2, 1
    close = np.array([[100.0], [102.0]])
    long_entries = np.zeros((T, N), dtype=bool)
    long_entries[0, 0] = True
    long_exits = np.zeros((T, N), dtype=bool)
    long_exits[1, 0] = True
    _, cash, pos, mrg = simulate_futures_nb(
        close=close, long_entries=long_entries, long_exits=long_exits,
        short_entries=_all_false(T, N), short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0]),
        margin_rate=np.array([0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([2], dtype=np.int8),
        init_cash=10_000.0,
    )
    assert pos[1, 0] == 0.0
    assert mrg[1, 0] == 0.0
    # t=0: cash = 10000 - 100 (margin) = 9900
    # t=1:
    #   STEP 1 mtm: 1*(102-100)*10 = +20 -> cash = 9920
    #   STEP 2 long_exit -> do_close: release 100 + pnl 20 = +120 -> cash = 10040
    # Net profit = 20 (1 lot * 2 diff * 10 mult), so 10000 -> 10040.
    assert cash[1] == pytest.approx(10_040.0)


def test_multi_column_independent_margin() -> None:
    """3 contracts on 3 columns.  Each maintains its own margin independently."""
    T, N = 2, 3
    close = np.array([[100.0, 200.0, 50.0], [100.0, 200.0, 50.0]])
    long_entries = np.zeros((T, N), dtype=bool)
    long_entries[0, :] = True
    _, cash, pos, mrg = simulate_futures_nb(
        close=close, long_entries=long_entries,
        long_exits=_all_false(T, N),
        short_entries=_all_false(T, N), short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0, 5.0, 100.0]),
        margin_rate=np.array([0.10, 0.20, 0.05]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([2, 2, 2], dtype=np.int8),
        init_cash=100_000.0,
    )
    # col 0: 100 * 10 * 0.10 = 100
    # col 1: 200 * 5  * 0.20 = 200
    # col 2: 50  * 100 * 0.05 = 250
    assert mrg[0, 0] == 100.0
    assert mrg[0, 1] == 200.0
    assert mrg[0, 2] == 250.0
    assert pos[0, 0] == 1.0
    assert pos[0, 1] == 1.0
    assert pos[0, 2] == 1.0
    # total locked: 550, total cash: 100000 - 550 = 99450
    assert cash[0] == 100_000.0 - 550.0
