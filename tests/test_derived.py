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


def test_orders_dataframe_has_expected_columns(simple_portfolio: FuturesPortfolio) -> None:
    o = simple_portfolio.orders
    assert list(o.columns) == [
        "id", "col", "idx", "size", "price", "fees",
        "margin", "side", "pnl", "symbol", "timestamp",
    ]
    assert len(o) == 3
    assert o["symbol"].iloc[0] == "RB"
    assert o["symbol"].iloc[1] == "HC"


def test_trades_pairing_yields_one_trade(simple_portfolio: FuturesPortfolio) -> None:
    """3 orders: OPEN_LONG(RB), OPEN_SHORT(HC), CLOSE_LONG(RB) -> 1 trade (RB)."""
    t = simple_portfolio.trades
    assert list(t.columns) == [
        "col", "symbol", "entry_time", "entry_price", "exit_time",
        "exit_price", "size", "pnl", "fees", "duration_bars",
        "is_liquidated",
    ]
    assert len(t) == 1
    assert t["symbol"].iloc[0] == "RB"
    assert t["size"].iloc[0] == 1.0
    assert t["pnl"].iloc[0] == 20.0
    assert t["is_liquidated"].iloc[0] == False  # noqa: E712 (numpy bool)


def test_trades_pairing_with_reversal(two_col_state) -> None:
    """Reversal (close+open) creates two separate trades in the same bar."""
    # Add a CLOSE_LONG+OPEN_SHORT pair to col 0 at bar 1.
    recs = np.empty(5, dtype=FUTURES_ORDER_DT)
    recs[0] = (0, 0, 0,  1.0, 100.0, 0.0,  100.0, 0, 0.0)   # OPEN_LONG
    recs[1] = (1, 1, 0, -1.0, 200.0, 0.0,   50.0, 2, 0.0)   # OPEN_SHORT HC
    recs[2] = (2, 0, 1, -1.0, 102.0, 0.0, -100.0, 1, 20.0)  # CLOSE_LONG
    recs[3] = (3, 0, 1, -1.0, 102.0, 0.0,  102.0, 2, 0.0)   # OPEN_SHORT
    recs[4] = (4, 0, 1,  1.0, 100.0, 0.0, -102.0, 3, 20.0)  # CLOSE_SHORT pnl 20
    cash, pos, mrg = two_col_state
    T = 2
    close = pd.DataFrame(
        [[100.0, 200.0], [102.0, 198.0]],
        columns=["RB", "HC"],
        index=pd.bdate_range("2024-01-02", periods=T),
    )
    p = FuturesPortfolio(
        close=close,
        specs=(FuturesSpec("RB", 10.0, 0.10), FuturesSpec("HC", 10.0, 0.05)),
        init_cash=10_000.0, freq="1D", bars_per_year=252.0, trading_days_per_year=252,
        _order_records=recs, _cash=cash, _position=pos, _margin_locked=mrg,
    )
    t = p.trades
    # Expect 2 trades: RB long (recs 0+2) and RB short (recs 3+4 from reversal).
    # HC has only an OPEN_SHORT and no matching close -> no trade.
    assert len(t) == 2
    assert t["symbol"].tolist() == ["RB", "RB"]
    assert t["pnl"].tolist()[0] == 20.0
    assert t["pnl"].tolist()[1] == 20.0
    assert t["size"].tolist()[1] == -1.0  # short side


def test_orders_empty_when_no_records() -> None:
    """orders returns empty DataFrame with the expected columns when no orders."""
    close = pd.DataFrame(
        [[100.0, 200.0]],
        columns=["RB", "HC"],
        index=pd.bdate_range("2024-01-02", periods=1),
    )
    p = FuturesPortfolio(
        close=close, specs=(FuturesSpec("RB", 10.0, 0.10), FuturesSpec("HC", 10.0, 0.05)),
        init_cash=10_000.0, freq="1D", bars_per_year=252.0, trading_days_per_year=252,
        _order_records=np.empty(0, dtype=FUTURES_ORDER_DT),
        _cash=np.array([10_000.0]),
        _position=np.array([[0.0, 0.0]]),
        _margin_locked=np.array([[0.0, 0.0]]),
    )
    assert len(p.orders) == 0
    assert len(p.trades) == 0


def test_trades_skips_close_without_open() -> None:
    """A close-side order without a prior open should not produce a trade row."""
    recs = np.empty(1, dtype=FUTURES_ORDER_DT)
    recs[0] = (0, 0, 0, -1.0, 100.0, 0.0, 0.0, 1, 0.0)  # CLOSE_LONG with no prior open
    close = pd.DataFrame(
        [[100.0]],
        columns=["RB"],
        index=pd.bdate_range("2024-01-02", periods=1),
    )
    p = FuturesPortfolio(
        close=close, specs=(FuturesSpec("RB", 10.0, 0.10),),
        init_cash=10_000.0, freq="1D", bars_per_year=252.0, trading_days_per_year=252,
        _order_records=recs,
        _cash=np.array([10_000.0]),
        _position=np.array([[0.0]]),
        _margin_locked=np.array([[0.0]]),
    )
    assert len(p.trades) == 0


def test_plot_handles_no_orders() -> None:
    """plot() should still produce a figure when there are no orders."""
    close = pd.DataFrame(
        [[100.0]],
        columns=["RB"],
        index=pd.bdate_range("2024-01-02", periods=1),
    )
    p = FuturesPortfolio(
        close=close, specs=(FuturesSpec("RB", 10.0, 0.10),),
        init_cash=10_000.0, freq="1D", bars_per_year=252.0, trading_days_per_year=252,
        _order_records=np.empty(0, dtype=FUTURES_ORDER_DT),
        _cash=np.array([10_000.0]),
        _position=np.array([[0.0]]),
        _margin_locked=np.array([[0.0]]),
    )
    fig = p.plot()
    # 1 close + 1 equity + 1 drawdown = 3 traces (no order markers).
    assert len(fig.data) == 3
