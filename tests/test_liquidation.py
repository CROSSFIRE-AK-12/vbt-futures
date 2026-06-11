"""Focused tests for forced liquidation (equity <= 0)."""
from __future__ import annotations

import numpy as np
import pytest

from vbt_futures.simulator import simulate_futures_nb
from vbt_futures.enums import LIQUIDATED


def _all_false(T: int, N: int) -> np.ndarray:
    return np.zeros((T, N), dtype=bool)


def _const_close(T: int, N: int, value: float) -> np.ndarray:
    return np.full((T, N), value, dtype=np.float64)


def test_liquidate_single_column_when_equity_below_zero() -> None:
    """Open a long; price crashes so equity -> 0; expect forced close with LIQUIDATED side."""
    T, N = 2, 1
    # Use a price drop that takes equity below 0: 100 -> 50, mtm = -500.
    # Cash starts at 9900, mtm -500 = 9400, mrg release 100 = 9500.  Wait,
    # equity is cash + margin_locked = 9400 + 0 = 9400 (after STEP 1 mtm
    # but before STEP 3).  So no liquidation here.  Need a bigger drop.
    # mtm = 1 * (50-100) * 10 = -500; after STEP 1 cash = 9900 - 500 = 9400.
    # Equity = 9400 + 100 (mrg) = 9500, still positive.  No liq.
    # Try close=30: mtm = 1 * (30-100) * 10 = -700; cash = 9900-700 = 9200;
    # equity = 9200 + 100 = 9300.  Still positive.
    # Need cash to go below 0 considering mrg:  cash < -mrg => 9900-100*x < -100
    # => 100*x > 10000 => x > 100.  So price must fall by > 100 to liquidate.
    # price = 0: mtm = 1 * (0-100) * 10 = -1000; cash = 9900 - 1000 = 8900.
    # equity = 8900 + 100 = 9000.  Still positive.
    # Actually we can't drive equity below 0 with mult=10 and 1 lot unless
    # we set initial cash very low.  Let's do that.
    close = np.array([[100.0], [0.0]])
    long_entries = np.zeros((T, N), dtype=bool)
    long_entries[0, 0] = True
    # init_cash = 50 -> t=0: pay 100 margin?  No, reject.  Need cash >= 100 + 0 fees.
    # Use init_cash = 100: t=0 cash=0, mrg=100; t=1 mtm -1000, cash=-1000; equity=-1000+100=-900<0 -> liq.
    orders, cash, pos, mrg = simulate_futures_nb(
        close=close, long_entries=long_entries,
        long_exits=_all_false(T, N),
        short_entries=_all_false(T, N), short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0]),
        margin_rate=np.array([0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([2], dtype=np.int8),
        init_cash=100.0,
    )
    # Order 0: OPEN_LONG at t=0.  Order 1: forced CLOSE at t=1 with side=LIQUIDATED.
    assert len(orders) == 2
    assert orders[1]["side"] == LIQUIDATED
    assert pos[1, 0] == 0.0
    assert mrg[1, 0] == 0.0


def test_no_new_orders_after_liquidation() -> None:
    """Once a column is liquidated, signals on that column are ignored."""
    T, N = 2, 1
    close = np.array([[100.0], [0.0]])
    long_entries = np.zeros((T, N), dtype=bool)
    long_entries[0, 0] = True
    long_entries[1, 0] = True  # would be another entry, but column is liquidated
    orders, _, pos, _ = simulate_futures_nb(
        close=close, long_entries=long_entries,
        long_exits=_all_false(T, N),
        short_entries=_all_false(T, N), short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0]),
        margin_rate=np.array([0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([2], dtype=np.int8),
        init_cash=100.0,
    )
    # Only OPEN_LONG at t=0 and LIQUIDATED at t=1 - the t=1 long_entry is suppressed.
    assert len(orders) == 2
    assert pos[1, 0] == 0.0


def test_liquidation_preserves_equity_series_length() -> None:
    """Output time series always have length T even when all columns liquidated."""
    T, N = 3, 1
    close = np.array([[100.0], [0.0], [50.0]])
    long_entries = np.zeros((T, N), dtype=bool)
    long_entries[0, 0] = True
    _, cash, pos, mrg = simulate_futures_nb(
        close=close, long_entries=long_entries,
        long_exits=_all_false(T, N),
        short_entries=_all_false(T, N), short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0]),
        margin_rate=np.array([0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([2], dtype=np.int8),
        init_cash=100.0,
    )
    assert cash.shape == (T,)
    assert pos.shape == (T, N)
    assert mrg.shape == (T, N)


def test_no_liquidation_when_equity_above_zero() -> None:
    """If price drops but equity stays > 0, normal close (not liquidated) applies."""
    T, N = 2, 1
    close = np.array([[100.0], [99.0]])
    long_entries = np.zeros((T, N), dtype=bool)
    long_entries[0, 0] = True
    long_exits = np.zeros((T, N), dtype=bool)
    long_exits[1, 0] = True
    orders, cash, pos, mrg = simulate_futures_nb(
        close=close, long_entries=long_entries, long_exits=long_exits,
        short_entries=_all_false(T, N), short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0]),
        margin_rate=np.array([0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([2], dtype=np.int8),
        init_cash=10_000.0,
    )
    # No LIQUIDATED side recorded.
    sides = list(orders["side"])
    assert LIQUIDATED not in sides
    # equity = 10000 - 10 + 0 = 9990 (small loss), still positive.
    assert cash[1] + mrg[1, 0] > 0.0
