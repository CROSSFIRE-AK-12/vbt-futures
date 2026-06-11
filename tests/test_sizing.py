"""Tests for src/vbt_futures/sizing.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vbt_futures.sizing import (
    SIZE_ANTI_MARTINGALE,
    SIZE_EQUITY_PROPORTIONAL,
    SIZE_FIXED,
    _broadcast_size,
    _entry_signal_mask,
    _exit_signal_mask,
    resolve_size,
)


# ---------------------------------------------------------------------------
# _broadcast_size
# ---------------------------------------------------------------------------
def test_broadcast_size_scalar() -> None:
    arr = _broadcast_size(2.5, T=3, N=4)
    assert arr.shape == (3, 4)
    assert arr[0, 0] == 2.5
    assert (arr == 2.5).all()


def test_broadcast_size_1d_per_column() -> None:
    arr = _broadcast_size(np.array([1.0, 2.0, 3.0]), T=5, N=3)
    assert arr.shape == (5, 3)
    assert arr[:, 0].mean() == 1.0
    assert arr[:, 1].mean() == 2.0


def test_broadcast_size_2d_passthrough() -> None:
    src = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    arr = _broadcast_size(src, T=3, N=2)
    np.testing.assert_array_equal(arr, src)


def test_broadcast_size_1d_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="size 1D array must have length"):
        _broadcast_size(np.array([1.0, 2.0]), T=3, N=3)


def test_broadcast_size_2d_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match="size 2D array must be shape"):
        _broadcast_size(np.zeros((2, 3)), T=3, N=2)


def test_broadcast_size_3d_raises() -> None:
    with pytest.raises(ValueError, match="size must be scalar, 1D, or 2D"):
        _broadcast_size(np.zeros((2, 3, 4)), T=3, N=2)


# ---------------------------------------------------------------------------
# _entry_signal_mask / _exit_signal_mask
# ---------------------------------------------------------------------------
def test_entry_signal_mask_long_only() -> None:
    le = np.array([[True], [False], [True]])
    mask = _entry_signal_mask(le, None)
    np.testing.assert_array_equal(mask, le)


def test_entry_signal_mask_short_only() -> None:
    se = np.array([[False], [True], [False]])
    mask = _entry_signal_mask(None, se)
    np.testing.assert_array_equal(mask, se)


def test_entry_signal_mask_combined() -> None:
    le = np.array([[True, False]])
    se = np.array([[False, True]])
    mask = _entry_signal_mask(le, se)
    np.testing.assert_array_equal(mask, np.array([[True, True]]))


def test_exit_signal_mask_both_none() -> None:
    assert _exit_signal_mask(None, None) is None


def test_exit_signal_mask_long_only() -> None:
    lx = np.array([[True, False]])
    mask = _exit_signal_mask(lx, None)
    np.testing.assert_array_equal(mask, lx)


def test_exit_signal_mask_short_only() -> None:
    sx = np.array([[False, True]])
    mask = _exit_signal_mask(None, sx)
    np.testing.assert_array_equal(mask, sx)


# ---------------------------------------------------------------------------
# fixed mode
# ---------------------------------------------------------------------------
def test_fixed_mode_returns_base_size() -> None:
    base = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    le = np.array([[True, False], [False, True], [False, False]])
    out = resolve_size(
        SIZE_FIXED, base,
        long_entries=le, long_exits=None,
        short_entries=np.zeros_like(le), short_exits=None,
        close=np.zeros((3, 2)), init_cash=10_000.0,
    )
    np.testing.assert_array_equal(out, base)


# ---------------------------------------------------------------------------
# equity_proportional
# ---------------------------------------------------------------------------
def test_equity_proportional_starts_at_base() -> None:
    """At t=0, equity = init_cash, so scale = 1.0, size = base."""
    base = np.array([[2.0, 3.0], [0.0, 0.0]])
    le = np.array([[True, True], [False, False]])
    lx = np.array([[False, False], [True, True]])
    close = np.array([[100.0, 200.0], [100.0, 200.0]])
    out = resolve_size(
        SIZE_EQUITY_PROPORTIONAL, base,
        long_entries=le, long_exits=lx,
        short_entries=np.zeros_like(le), short_exits=None,
        close=close, init_cash=10_000.0,
    )
    # At t=0, scale = 1.0, so size = base.
    assert out[0, 0] == 2.0
    assert out[0, 1] == 3.0
    # After exit (t=1), no size.
    assert out[1, 0] == 0.0


def test_equity_proportional_scales_after_profit() -> None:
    """After a profitable round, equity grows, so next entry size is scaled up."""
    base = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    le = np.array([[True, False], [False, False], [True, False]])
    lx = np.array([[False, False], [True, False], [False, False]])
    close = np.array([[100.0, 100.0], [200.0, 100.0], [200.0, 100.0]])
    out = resolve_size(
        SIZE_EQUITY_PROPORTIONAL, base,
        long_entries=le, long_exits=lx,
        short_entries=np.zeros_like(le), short_exits=None,
        close=close, init_cash=10_000.0,
    )
    # At t=0 entry: scale=1, size=1.
    # t=1 exit: cash += 1 * (200-100) = +100, equity so far = 10100.
    # t=2 entry: scale = 10100/10000 = 1.01, size ≈ 1.01.
    assert out[2, 0] > out[0, 0]
    np.testing.assert_allclose(out[2, 0], 1.01, rtol=0.01)


def test_equity_proportional_with_zero_equity() -> None:
    """When cum_equity <= 0, scale = 0, no entry allowed."""
    base = np.array([[1.0, 1.0]])
    le = np.array([[True, True]])
    lx = np.array([[False, False]])
    close = np.array([[100.0, 100.0]])
    out = resolve_size(
        SIZE_EQUITY_PROPORTIONAL, base,
        long_entries=le, long_exits=lx,
        short_entries=np.zeros_like(le), short_exits=None,
        close=close, init_cash=0.0,
    )
    assert out[0, 0] == 0.0
    assert out[0, 1] == 0.0


def test_equity_proportional_short_position() -> None:
    """A short position should also be tracked correctly (mtm can be negative)."""
    base = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    # Short entry at t=0, exit at t=1 (no PnL since price unchanged), re-enter at t=2.
    se = np.array([[True, True], [False, False], [True, True]])
    sx = np.array([[False, False], [True, True], [False, False]])
    le = np.zeros_like(se)
    lx = np.zeros_like(se)
    close = np.array([[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]])
    out = resolve_size(
        SIZE_EQUITY_PROPORTIONAL, base,
        long_entries=le, long_exits=lx,
        short_entries=se, short_exits=sx,
        close=close, init_cash=10_000.0,
    )
    # t=0 short at 100, t=1 exit at 100 (no PnL).
    # t=2 re-enter: cum_equity = init (no PnL), size = 1.0.
    np.testing.assert_allclose(out[2, 0], 1.0, rtol=0.01)


def test_equity_proportional_short_position_scales_after_profit() -> None:
    """Short held through a price drop -> cum_pnl > 0 -> size at next entry > 1.0."""
    base = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    se = np.array([[True, True], [False, False], [True, True]])
    sx = np.array([[False, False], [True, True], [False, False]])
    le = np.zeros_like(se)
    lx = np.zeros_like(se)
    # Short opens at 100, price drops to 50 over 2 bars (held 50+50 pnl),
    # exit at 50, then re-enter at 50.
    close = np.array([[100.0, 100.0], [50.0, 50.0], [50.0, 50.0]])
    out = resolve_size(
        SIZE_EQUITY_PROPORTIONAL, base,
        long_entries=le, long_exits=lx,
        short_entries=se, short_exits=sx,
        close=close, init_cash=10_000.0,
    )
    # t=0 short at 100.
    # t=1 mtm +50 (price drop), then exit +50 (realised) = cum_equity = 10100.
    # t=2 re-enter: size = 1.0 * 10100/10000 = 1.01.
    np.testing.assert_allclose(out[2, 0], 1.01, rtol=0.01)


# ---------------------------------------------------------------------------
# anti_martingale
# ---------------------------------------------------------------------------
def test_anti_martingale_no_bonus_initially() -> None:
    """No cum_pnl yet -> no bonus, size = base."""
    base = np.array([[2.0, 0.0]])
    le = np.array([[True, False]])
    lx = np.array([[False, False]])
    close = np.array([[100.0, 100.0]])
    out = resolve_size(
        SIZE_ANTI_MARTINGALE, base,
        long_entries=le, long_exits=lx,
        short_entries=np.zeros_like(le), short_exits=None,
        close=close, init_cash=10_000.0,
        sizing_kwargs={"trigger_pnl": 1_000.0, "max_size": 5.0},
    )
    assert out[0, 0] == 2.0


def test_anti_martingale_bonus_after_profit() -> None:
    """After cum_pnl > trigger, size = base + bonus."""
    # 3 bars: t=0 enter, t=1 exit (captures 50 pnl), t=2 re-enter (bonus applies).
    base = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    le = np.array([[True, False], [False, False], [True, False]])
    lx = np.array([[False, False], [True, False], [False, False]])
    close = np.array([[100.0, 100.0], [150.0, 100.0], [150.0, 100.0]])
    out = resolve_size(
        SIZE_ANTI_MARTINGALE, base,
        long_entries=le, long_exits=lx,
        short_entries=np.zeros_like(le), short_exits=None,
        close=close, init_cash=10_000.0,
        sizing_kwargs={"trigger_pnl": 1_000.0, "max_size": 5.0},
    )
    # t=0 enter at 100, t=1 exit at 150 (cum_pnl=50), t=2 re-enter (bonus=0.05).
    np.testing.assert_allclose(out[2, 0], 1.05, rtol=0.01)


def test_anti_martingale_max_size_cap() -> None:
    """Size never exceeds max_size even if cum_pnl is huge."""
    base = np.array([[1.0, 1.0], [1.0, 1.0]])
    le = np.array([[True, False], [True, False]])
    lx = np.array([[False, False], [False, False]])
    close = np.array([[100.0, 100.0], [10_000.0, 100.0]])  # huge profit
    out = resolve_size(
        SIZE_ANTI_MARTINGALE, base,
        long_entries=le, long_exits=lx,
        short_entries=np.zeros_like(le), short_exits=None,
        close=close, init_cash=10_000.0,
        sizing_kwargs={"trigger_pnl": 100.0, "max_size": 3.0},
    )
    assert out[1, 0] <= 3.0


def test_anti_martingale_short_entry_uses_signed_position() -> None:
    """The short-entry branch in anti_martingale should set size correctly."""
    base = np.array([[1.0, 1.0], [1.0, 1.0]])
    se = np.array([[True, False], [True, False]])
    sx = np.array([[False, False], [False, False]])
    le = np.zeros_like(se)
    lx = np.zeros_like(se)
    close = np.array([[100.0, 100.0], [100.0, 100.0]])
    out = resolve_size(
        SIZE_ANTI_MARTINGALE, base,
        long_entries=le, long_exits=lx,
        short_entries=se, short_exits=sx,
        close=close, init_cash=10_000.0,
        sizing_kwargs={"trigger_pnl": 1_000.0, "max_size": 5.0},
    )
    # No PnL yet -> size = base_size = 1.0.
    assert out[0, 0] == 1.0


# ---------------------------------------------------------------------------
# invalid mode
# ---------------------------------------------------------------------------
def test_invalid_mode_raises() -> None:
    le = np.zeros((1, 1), dtype=bool)
    with pytest.raises(ValueError, match="sizing_mode 必须是"):
        resolve_size(
            "bogus", 1.0, le, None, le, None,
            np.zeros((1, 1)), 10_000.0,
        )
