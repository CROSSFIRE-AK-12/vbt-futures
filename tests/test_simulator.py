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


def test_close_long_releases_margin_and_books_pnl() -> None:
    """Open long at 100, close at 105 -> 1*5*10 = 50 pnl."""
    T, N = 2, 1
    close = np.array([[100.0], [105.0]])
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
    # t=0: open long, cash=9900, mrg=100
    assert cash[0] == 9_900.0
    assert pos[0, 0] == 1.0
    assert mrg[0, 0] == 100.0
    # t=1:
    #   STEP 1 mtm: 1*(105-100)*10 = +50  -> cash=9950
    #   STEP 2 long_exit -> do_close: release margin 100 + realized pnl 50 = +150 -> cash=10100
    #   STEP 3: position=0, mrg=0, no change
    # Net: started 10000, ended 10100, profit = 100 = mult*1*(105-100) = 10*5*1
    assert cash[1] == 10_100.0
    assert pos[1, 0] == 0.0
    assert mrg[1, 0] == 0.0
    assert len(orders) == 2
    assert orders[0]["side"] == 0     # OPEN_LONG
    assert orders[1]["side"] == 1     # CLOSE_LONG
    assert orders[1]["size"] == -1.0
    assert orders[1]["price"] == 105.0
    assert orders[1]["pnl"] == 50.0
    assert orders[1]["margin"] == -100.0


def test_open_short_consumes_margin() -> None:
    """Opening a short position also consumes margin (notional * margin_rate)."""
    T, N = 2, 1
    close = _const_close(T, N, 100.0)
    short_entries = np.zeros((T, N), dtype=bool)
    short_entries[0, 0] = True
    orders, cash, pos, mrg = simulate_futures_nb(
        close=close, long_entries=_all_false(T, N),
        long_exits=_all_false(T, N),
        short_entries=short_entries, short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0]),
        margin_rate=np.array([0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([2], dtype=np.int8),
        init_cash=10_000.0,
    )
    assert cash[0] == 9_900.0
    assert pos[0, 0] == -1.0
    assert mrg[0, 0] == 100.0
    assert len(orders) == 1
    assert orders[0]["side"] == 2     # OPEN_SHORT
    assert orders[0]["size"] == -1.0


def test_close_short_releases_margin_and_books_pnl() -> None:
    """Open short at 100, close at 95 -> 1*5*10 = 50 pnl."""
    T, N = 2, 1
    close = np.array([[100.0], [95.0]])
    short_entries = np.zeros((T, N), dtype=bool)
    short_entries[0, 0] = True
    short_exits = np.zeros((T, N), dtype=bool)
    short_exits[1, 0] = True
    orders, cash, pos, mrg = simulate_futures_nb(
        close=close, long_entries=_all_false(T, N),
        long_exits=_all_false(T, N),
        short_entries=short_entries, short_exits=short_exits,
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0]),
        margin_rate=np.array([0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([2], dtype=np.int8),
        init_cash=10_000.0,
    )
    # t=0: open short, cash=9900, mrg=100, pos=-1
    assert cash[0] == 9_900.0
    assert pos[0, 0] == -1.0
    # t=1:
    #   STEP 1 mtm: -1*(95-100)*10 = +50 -> cash=9950
    #   STEP 2 short_exit -> do_close: release 100 + pnl 50 = +150 -> cash=10100
    assert cash[1] == 10_100.0
    assert pos[1, 0] == 0.0
    assert mrg[1, 0] == 0.0
    assert len(orders) == 2
    assert orders[0]["side"] == 2     # OPEN_SHORT
    assert orders[1]["side"] == 3     # CLOSE_SHORT
    assert orders[1]["size"] == 1.0
    assert orders[1]["price"] == 95.0
    assert orders[1]["pnl"] == 50.0


def test_mark_to_market_decreases_cash_when_price_rises_short() -> None:
    """Short 1 lot, price rises 100 -> 102 -> loss of 20."""
    T, N = 2, 1
    close = np.array([[100.0], [102.0]])
    short_entries = np.zeros((T, N), dtype=bool)
    short_entries[0, 0] = True
    _, cash, _, mrg = simulate_futures_nb(
        close=close, long_entries=_all_false(T, N),
        long_exits=_all_false(T, N),
        short_entries=short_entries, short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0]),
        margin_rate=np.array([0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([2], dtype=np.int8),
        init_cash=10_000.0,
    )
    # t=1: mtm = -1*(102-100)*10 = -20 -> cash=9880
    #      new_margin = 102*10*0.10 = 102 -> cash -= 2 -> 9878
    assert cash[1] == 9_878.0
    assert mrg[1, 0] == 102.0
    # equity = 9878 + 102 = 9980 (= 10000 - 20 loss)
    assert cash[1] + mrg[1, 0] == 9_980.0


def test_reversal_long_to_short_emits_two_records() -> None:
    """Holding long 1 lot, short_entry fires -> reverse to short.

    Sequence: OPEN_LONG (t=0), then CLOSE_LONG + OPEN_SHORT at t=1.
    Total: 3 records (1 open + 2 reversal).  The reversal is a pair: close
    first (with pnl), then open the new side at the same bar.
    """
    T, N = 2, 1
    close = np.array([[100.0], [102.0]])
    long_entries = np.zeros((T, N), dtype=bool)
    long_entries[0, 0] = True
    short_entries = np.zeros((T, N), dtype=bool)
    short_entries[1, 0] = True
    orders, cash, pos, mrg = simulate_futures_nb(
        close=close, long_entries=long_entries, long_exits=_all_false(T, N),
        short_entries=short_entries, short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0]),
        margin_rate=np.array([0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([2], dtype=np.int8),
        init_cash=10_000.0,
    )
    # 1 open + 2 reversal = 3 records.
    assert len(orders) == 3
    assert orders[0]["side"] == 0     # OPEN_LONG
    assert orders[1]["side"] == 1     # CLOSE_LONG
    assert orders[2]["side"] == 2     # OPEN_SHORT
    assert orders[1]["size"] == -1.0
    assert orders[2]["size"] == -1.0
    # After reversal, position is -1 and margin is locked for the new short.
    assert pos[1, 0] == -1.0
    assert mrg[1, 0] == 102.0
    # Order of long-open at t=0, reversal records at t=1.
    assert orders[0]["idx"] == 0
    assert orders[1]["idx"] == 1
    assert orders[2]["idx"] == 1
    # Close_long at 102 with avg entry 100 -> pnl = 1*(102-100)*10 = 20
    assert orders[1]["pnl"] == 20.0
    assert orders[1]["price"] == 102.0


def test_reversal_short_to_long_emits_two_records() -> None:
    """Holding short 1 lot, long_entry fires -> reverse to long.

    Sequence: OPEN_SHORT (t=0), then CLOSE_SHORT + OPEN_LONG at t=1.
    Total: 3 records.
    """
    T, N = 2, 1
    close = np.array([[100.0], [98.0]])
    short_entries = np.zeros((T, N), dtype=bool)
    short_entries[0, 0] = True
    long_entries = np.zeros((T, N), dtype=bool)
    long_entries[1, 0] = True
    orders, cash, pos, mrg = simulate_futures_nb(
        close=close, long_entries=long_entries, long_exits=_all_false(T, N),
        short_entries=short_entries, short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0]),
        margin_rate=np.array([0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([2], dtype=np.int8),
        init_cash=10_000.0,
    )
    assert len(orders) == 3
    assert orders[0]["side"] == 2     # OPEN_SHORT
    assert orders[1]["side"] == 3     # CLOSE_SHORT
    assert orders[2]["side"] == 0     # OPEN_LONG
    assert orders[1]["size"] == 1.0
    assert orders[2]["size"] == 1.0
    assert pos[1, 0] == 1.0
    assert mrg[1, 0] == 98.0
    # Close_short at 98 with avg entry 100 -> pnl = -1*(98-100)*10 = 20
    assert orders[1]["pnl"] == 20.0
    assert orders[1]["price"] == 98.0


def test_long_exit_then_long_entry_same_bar_emits_two_records() -> None:
    """Holding long, long_exit AND long_entry both fire on same bar.

    Per spec §5.1: PASS 1 closes the position; then PASS 2 (position==0
    now) re-opens a long.  Net result: 2 records, both at the same bar.
    """
    T, N = 2, 1
    close = _const_close(T, N, 100.0)
    long_entries = np.zeros((T, N), dtype=bool)
    long_entries[0, 0] = True
    long_entries[1, 0] = True
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
    # 3 records: OPEN_LONG (t=0), CLOSE_LONG (t=1), OPEN_LONG (t=1).
    assert len(orders) == 3
    assert orders[0]["side"] == 0     # OPEN_LONG at t=0
    assert orders[1]["side"] == 1     # CLOSE_LONG at t=1
    assert orders[2]["side"] == 0     # OPEN_LONG at t=1
    assert orders[1]["idx"] == 1
    assert orders[2]["idx"] == 1
    # At t=1: PASS 1 closes at 100 (no pnl since entry 100), PASS 2 reopens at 100.
    # After t=1: position=1, mrg=100.
    assert pos[1, 0] == 1.0
    assert mrg[1, 0] == 100.0
    assert orders[1]["pnl"] == 0.0
    assert orders[2]["margin"] == 100.0


def test_flat_conflict_skip_when_both_entries_true() -> None:
    """Both long_entry and short_entry fire on a flat bar.  Default 'skip' -> no order."""
    T, N = 1, 1
    close = _const_close(T, N, 100.0)
    long_entries = np.ones((T, N), dtype=bool)
    short_entries = np.ones((T, N), dtype=bool)
    orders, cash, pos, mrg = simulate_futures_nb(
        close=close, long_entries=long_entries, long_exits=_all_false(T, N),
        short_entries=short_entries, short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0]),
        margin_rate=np.array([0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([2], dtype=np.int8),  # skip
        init_cash=10_000.0,
    )
    assert len(orders) == 0
    assert pos[0, 0] == 0.0
    assert mrg[0, 0] == 0.0
    assert cash[0] == 10_000.0


def test_flat_conflict_long_when_configured() -> None:
    """flat_conflict='long' code 0 -> open long when both fire."""
    T, N = 1, 1
    close = _const_close(T, N, 100.0)
    long_entries = np.ones((T, N), dtype=bool)
    short_entries = np.ones((T, N), dtype=bool)
    orders, cash, pos, mrg = simulate_futures_nb(
        close=close, long_entries=long_entries, long_exits=_all_false(T, N),
        short_entries=short_entries, short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0]),
        margin_rate=np.array([0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([0], dtype=np.int8),  # long
        init_cash=10_000.0,
    )
    assert len(orders) == 1
    assert orders[0]["side"] == 0     # OPEN_LONG
    assert pos[0, 0] == 1.0


def test_flat_conflict_short_when_configured() -> None:
    """flat_conflict='short' code 1 -> open short when both fire."""
    T, N = 1, 1
    close = _const_close(T, N, 100.0)
    long_entries = np.ones((T, N), dtype=bool)
    short_entries = np.ones((T, N), dtype=bool)
    orders, cash, pos, mrg = simulate_futures_nb(
        close=close, long_entries=long_entries, long_exits=_all_false(T, N),
        short_entries=short_entries, short_exits=_all_false(T, N),
        size=_const_close(T, N, 1.0),
        mult=np.array([10.0]),
        margin_rate=np.array([0.10]),
        fees=np.zeros(N), fixed_fees=np.zeros(N), slippage=np.zeros(N),
        flat_conflict_code=np.array([1], dtype=np.int8),  # short
        init_cash=10_000.0,
    )
    assert len(orders) == 1
    assert orders[0]["side"] == 2     # OPEN_SHORT
    assert pos[0, 0] == -1.0
