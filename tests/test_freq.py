"""Tests for src/vbt_futures/utils.py::infer_bars_per_year and friends."""

from __future__ import annotations

import pandas as pd
import pytest

from vbt_futures.utils import infer_bars_per_year


def _daily_index(n_days: int) -> pd.DatetimeIndex:
    """Build n_days of business-day timestamps, one per day."""
    return pd.bdate_range(start="2024-01-02", periods=n_days)


def _intraday_index(n_bars_per_day: int, n_days: int) -> pd.DatetimeIndex:
    """Build intraday business-day timestamps with n_bars_per_day bars.

    Uses ``pd.Timedelta`` to support counts that overflow 24h/day.
    Bars within a day are spaced 1 hour apart starting at 09:00.
    Only the *count* per day matters for these tests.
    """
    days = pd.bdate_range(start="2024-01-02", periods=n_days)
    times: list[pd.Timestamp] = []
    for d in days:
        for i in range(n_bars_per_day):
            times.append(d + pd.Timedelta(hours=9 + i))
    return pd.DatetimeIndex(times)


def test_infer_bars_per_year_daily_returns_252() -> None:
    idx = _daily_index(252)
    result = infer_bars_per_year(idx, trading_days_per_year=252)
    assert result == 252.0


def test_infer_bars_per_year_1h_day_session_returns_1008() -> None:
    """4 hours of trading per day (9, 10, 11, 12) x 252 = 1008."""
    idx = _intraday_index(n_bars_per_day=4, n_days=252)
    result = infer_bars_per_year(idx, trading_days_per_year=252)
    assert result == 1008.0


def test_infer_bars_per_year_15min_returns_4032() -> None:
    """16 fifteen-minute bars per day x 252 = 4032."""
    idx = _intraday_index(n_bars_per_day=16, n_days=252)
    result = infer_bars_per_year(idx, trading_days_per_year=252)
    assert result == 4032.0


def test_infer_bars_per_year_uses_median_robust_to_half_day() -> None:
    """One half-day (2 bars instead of 4) shouldn't shift the result.

    251 full days (4 bars) + 1 half day (2 bars) => 251*4+2 = 1006 bars.
    Counts per day: [4, 4, ..., 4 (251 times), 2]. Median = 4.
    Result = 4 * 252 = 1008 (NOT 1006).
    """
    full_days = pd.bdate_range(start="2024-01-02", periods=251)
    full_bars = pd.DatetimeIndex(
        [d + pd.Timedelta(hours=h) for d in full_days for h in [9, 10, 11, 12]],
    )
    half_day = pd.Timestamp("2024-12-31")
    half_bars = pd.DatetimeIndex(
        [half_day + pd.Timedelta(hours=9), half_day + pd.Timedelta(hours=10)],
    )
    idx = full_bars.append(half_bars)  # type: ignore[attr-defined]
    result = infer_bars_per_year(idx, trading_days_per_year=252)
    assert result == 1008.0


def test_infer_bars_per_year_trading_days_override() -> None:
    """For crypto (24/7) use 365."""
    idx = _intraday_index(n_bars_per_day=24, n_days=30)
    result = infer_bars_per_year(idx, trading_days_per_year=365)
    # 24 * 365 = 8760
    assert result == 8760.0


def test_infer_bars_per_year_handles_int_index_type() -> None:
    """Return type must be a plain float, not numpy scalar."""
    idx = _daily_index(10)
    result = infer_bars_per_year(idx)
    assert isinstance(result, float)


def test_infer_bars_per_year_empty_index() -> None:
    """Empty index -> 0.0."""
    idx = pd.DatetimeIndex([])
    assert infer_bars_per_year(idx) == 0.0


def test_infer_bars_per_year_non_datetime_index_raises() -> None:
    with pytest.raises(ValueError, match="infer_bars_per_year 要求 pd.DatetimeIndex"):
        infer_bars_per_year(pd.Index([1, 2, 3]))  # type: ignore[arg-type]


def test_infer_bars_per_year_non_positive_trading_days_raises() -> None:
    with pytest.raises(ValueError, match="trading_days_per_year 必须 > 0"):
        infer_bars_per_year(_daily_index(5), trading_days_per_year=0)
