"""Tests for src/vbt_futures/portfolio.py (derived properties)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vbt_futures.portfolio import FuturesPortfolio
from vbt_futures.records import FUTURES_ORDER_DT
from vbt_futures.spec import FuturesSpec


@pytest.fixture
def two_col_records() -> np.ndarray:
    """3 orders, 2 columns, manually constructed for unit tests."""
    arr = np.empty(3, dtype=FUTURES_ORDER_DT)
    arr[0] = (0, 0, 0,  1.0, 100.0, 0.0,  100.0, 0, 0.0)   # OPEN_LONG col 0
    arr[1] = (1, 1, 0, -1.0, 200.0, 0.0,  50.0,  2, 0.0)   # OPEN_SHORT col 1
    arr[2] = (2, 0, 1, -1.0, 102.0, 0.0, -100.0, 1, 20.0)  # CLOSE_LONG col 0, pnl 20
    return arr


@pytest.fixture
def two_col_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, N = 2, 2
    cash = np.array([9_900.0, 10_020.0])
    position = np.array([[1.0, -1.0], [0.0, -1.0]])
    margin_locked = np.array([[100.0, 50.0], [0.0, 50.0]])
    return cash, position, margin_locked


@pytest.fixture
def simple_portfolio(two_col_records: np.ndarray, two_col_state) -> FuturesPortfolio:
    cash, pos, mrg = two_col_state
    T = len(cash)
    close = pd.DataFrame(
        [[100.0, 200.0], [102.0, 198.0]],
        columns=["RB", "HC"],
        index=pd.bdate_range("2024-01-02", periods=T),
    )
    return FuturesPortfolio(
        close=close,
        specs=(
            FuturesSpec("RB", 10.0, 0.10),
            FuturesSpec("HC", 10.0, 0.05),
        ),
        init_cash=10_000.0,
        freq="1D",
        bars_per_year=252.0,
        trading_days_per_year=252,
        _order_records=two_col_records,
        _cash=cash,
        _position=pos,
        _margin_locked=mrg,
    )


def test_cash_series_index_aligns(two_col_records, two_col_state) -> None:
    cash, pos, mrg = two_col_state
    close = pd.DataFrame(
        [[100.0, 200.0], [102.0, 198.0]],
        columns=["RB", "HC"],
        index=pd.bdate_range("2024-01-02", periods=2),
    )
    p = FuturesPortfolio(
        close=close, specs=(FuturesSpec("RB", 10.0, 0.10), FuturesSpec("HC", 10.0, 0.05)),
        init_cash=10_000.0, freq="1D", bars_per_year=252.0, trading_days_per_year=252,
        _order_records=two_col_records, _cash=cash, _position=pos, _margin_locked=mrg,
    )
    s = p.cash
    assert isinstance(s, pd.Series)
    assert len(s) == 2
    assert s.index.equals(close.index)


def test_position_dataframe_columns_align(simple_portfolio: FuturesPortfolio) -> None:
    df = simple_portfolio.position
    assert df.shape == (2, 2)
    assert list(df.columns) == ["RB", "HC"]


def test_margin_locked_dataframe(simple_portfolio: FuturesPortfolio) -> None:
    df = simple_portfolio.margin_locked
    assert df.shape == (2, 2)
    assert df.iloc[0, 0] == 100.0
    assert df.iloc[1, 1] == 50.0


def test_equity_equals_cash_plus_margin_locked(simple_portfolio: FuturesPortfolio) -> None:
    """Critical invariant: equity = cash + sum(margin_locked)."""
    eq = simple_portfolio.equity
    cash = simple_portfolio.cash
    mrg = simple_portfolio.margin_locked
    expected = cash.values + mrg.sum(axis=1).values
    np.testing.assert_array_equal(eq.values, expected)


def test_returns_first_bar_is_zero(simple_portfolio: FuturesPortfolio) -> None:
    r = simple_portfolio.returns
    assert r.iloc[0] == 0.0  # first bar pct_change is NaN -> 0


def test_drawdown_is_non_positive(simple_portfolio: FuturesPortfolio) -> None:
    dd = simple_portfolio.drawdown
    assert (dd <= 0.0).all()
