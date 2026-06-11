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
    resolve_size,
)


# ---------- _broadcast_size ----------
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


# ---------- _entry_signal_mask ----------
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


# ---------- fixed mode ----------
def test_fixed_mode_returns_base_size() -> None:
    base = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    signals = np.array([[True, False], [False, True], [False, False]])
    out = resolve_size(SIZE_FIXED, base, signals, np.zeros((3, 2)), init_cash=10_000.0)
    np.testing.assert_array_equal(out, base)


# ---------- equity_proportional ----------
def test_equity_proportional_starts_at_base() -> None:
    """At t=0, equity = init_cash, so scale = 1.0, size = base."""
    base = np.array([[2.0, 3.0], [0.0, 0.0]])
    signals = np.array([[True, True], [False, False]])
    close = np.array([[100.0, 200.0], [100.0, 200.0]])
    out = resolve_size(
        SIZE_EQUITY_PROPORTIONAL, base, signals, close, init_cash=10_000.0,
    )
    # At t=0, scale = 1.0, so size = base.
    assert out[0, 0] == 2.0
    assert out[0, 1] == 3.0
    # Non-entry bars have 0.
    assert out[1, 0] == 0.0


def test_equity_proportional_scales_after_profit() -> None:
    """After a profitable round, equity grows, so next entry size is scaled up."""
    base = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
    signals = np.array([[True, False], [False, False], [False, True]])
    # Big price jump at t=1 (profit) then close stays elevated.
    close = np.array([[100.0, 100.0], [200.0, 100.0], [200.0, 100.0]])
    out = resolve_size(
        SIZE_EQUITY_PROPORTIONAL, base, signals, close, init_cash=10_000.0,
    )
    # At t=0 entry: scale=1, size=1
    # At t=1: no entry, position[0] held; mtm +1*100 = +100 (so equity ~ 10100)
    # At t=2 entry: scale = 10100/10000 = 1.01, size ≈ 1.01
    assert out[2, 1] > out[0, 0]  # scaled up by profit
    np.testing.assert_allclose(out[2, 1], 1.01, rtol=0.05)


def test_equity_proportional_with_zero_or_negative_equity() -> None:
    """When cum_equity <= 0, scale = 0, no entry allowed."""
    base = np.array([[1.0, 1.0]])
    signals = np.array([[True, True]])
    close = np.array([[100.0, 100.0]])
    # init_cash=0 means cum_equity = 0, branch goes to else.
    out = resolve_size(
        SIZE_EQUITY_PROPORTIONAL, base, signals, close, init_cash=0.0,
    )
    assert out[0, 0] == 0.0
    assert out[0, 1] == 0.0


# ---------- anti_martingale ----------
def test_anti_martingale_no_bonus_initially() -> None:
    """No cum_pnl yet -> no bonus, size = base."""
    base = np.array([[2.0, 0.0]])
    signals = np.array([[True, False]])
    close = np.array([[100.0, 100.0]])
    out = resolve_size(
        SIZE_ANTI_MARTINGALE, base, signals, close, init_cash=10_000.0,
        sizing_kwargs={"trigger_pnl": 1_000.0, "max_size": 5.0},
    )
    assert out[0, 0] == 2.0


def test_anti_martingale_bonus_after_profit() -> None:
    """After cum_pnl > trigger, size = base + bonus."""
    base = np.array([[1.0, 1.0], [1.0, 1.0]])
    signals = np.array([[True, False], [False, True]])
    # 2 bars: t=0 enter, t=1 price jumps to make cum_pnl = 500 (below trigger=1000)
    # t=2 (but T=2): no more entries
    close = np.array([[100.0, 100.0], [150.0, 100.0]])  # +50 pnl
    out = resolve_size(
        SIZE_ANTI_MARTINGALE, base, signals, close, init_cash=10_000.0,
        sizing_kwargs={"trigger_pnl": 1_000.0, "max_size": 5.0},
    )
    # At t=1, cum_pnl = 50, bonus = 50/1000 = 0.05.  size = 1 + 0.05 = 1.05.
    np.testing.assert_allclose(out[1, 1], 1.05, rtol=0.01)


def test_anti_martingale_max_size_cap() -> None:
    """Size never exceeds max_size even if cum_pnl is huge."""
    base = np.array([[1.0, 1.0], [1.0, 1.0]])
    signals = np.array([[True, False], [False, True]])
    close = np.array([[100.0, 100.0], [10_000.0, 100.0]])  # huge profit
    out = resolve_size(
        SIZE_ANTI_MARTINGALE, base, signals, close, init_cash=10_000.0,
        sizing_kwargs={"trigger_pnl": 100.0, "max_size": 3.0},
    )
    assert out[1, 1] <= 3.0


# ---------- invalid mode ----------
def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="sizing_mode 必须是"):
        resolve_size("bogus", 1.0, np.zeros((1, 1), dtype=bool), np.zeros((1, 1)), 10_000.0)
