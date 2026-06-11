"""End-to-end portfolio tests: from_simals wiring + stats() output."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from vbt_futures.portfolio import FuturesPortfolio
from vbt_futures.records import FUTURES_ORDER_DT
from vbt_futures.spec import FuturesSpec


@pytest.fixture
def portfolio() -> FuturesPortfolio:
    """A simple 3-bar, 1-contract portfolio that earns 20 RMB in profit."""
    close = pd.DataFrame(
        [[100.0], [101.0], [102.0]],
        columns=["RB"],
        index=pd.bdate_range("2024-01-02", periods=3),
    )
    # Order flow: open long at 100 (t=0), close long at 102 (t=2).
    # mtm at t=1: +10
    recs = np.empty(2, dtype=FUTURES_ORDER_DT)
    recs[0] = (0, 0, 0,  1.0, 100.0, 0.0,  100.0, 0, 0.0)   # OPEN_LONG
    recs[1] = (1, 0, 2, -1.0, 102.0, 0.0, -100.0, 1, 20.0)  # CLOSE_LONG
    # cash trajectory:
    #   t=0: 10000 - 100 = 9900
    #   t=1: 9900 + 10 = 9910 (mtm)
    #   t=2: 9910 + 100 (margin release) + 20 (pnl) = 10030
    cash = np.array([9900.0, 9910.0, 10030.0])
    position = np.array([[1.0], [1.0], [0.0]])
    margin_locked = np.array([[100.0], [100.0], [0.0]])
    return FuturesPortfolio(
        close=close, specs=(FuturesSpec("RB", 10.0, 0.10),),
        init_cash=10_000.0, freq="1D", bars_per_year=252.0, trading_days_per_year=252,
        _order_records=recs, _cash=cash, _position=position, _margin_locked=margin_locked,
    )


def test_stats_returns_expected_fields(portfolio: FuturesPortfolio) -> None:
    s = portfolio.stats()
    expected_fields = {
        "Start", "End", "Period", "Init Cash", "Final Equity",
        "Total Return [%]", "Annualized Return [%]", "Sharpe Ratio",
        "Sortino Ratio", "Max Drawdown [%]", "Total Trades",
        "Win Rate [%]", "Profit Factor", "Avg Trade PnL", "Avg Win",
        "Avg Loss", "Win/Loss Ratio", "Max Position", "Total Fees",
        "Liquidations", "Bars Per Year", "Trading Days Per Year",
    }
    assert set(s.index) == expected_fields


def test_stats_final_equity_matches(portfolio: FuturesPortfolio) -> None:
    s = portfolio.stats()
    assert s["Final Equity"] == pytest.approx(10_030.0)
    assert s["Init Cash"] == 10_000.0
    assert s["Total Return [%]"] == pytest.approx(0.3)


def test_stats_total_trades_is_one(portfolio: FuturesPortfolio) -> None:
    s = portfolio.stats()
    assert s["Total Trades"] == 1
    assert s["Liquidations"] == 0


def test_stats_bars_per_year_field(portfolio: FuturesPortfolio) -> None:
    s = portfolio.stats()
    assert s["Bars Per Year"] == 252.0
    assert s["Trading Days Per Year"] == 252


def test_stats_max_position(portfolio: FuturesPortfolio) -> None:
    s = portfolio.stats()
    assert s["Max Position"] == 1.0


def test_plot_returns_plotly_figure_with_expected_traces(portfolio: FuturesPortfolio) -> None:
    fig = portfolio.plot()
    assert isinstance(fig, go.Figure)
    # At least 2 traces: close line + equity line.
    assert len(fig.data) >= 2


def test_to_vbt_orders_maps_side_correctly(portfolio: FuturesPortfolio) -> None:
    df = portfolio.to_vbt_orders()
    # Same number of rows.
    assert len(df) == len(portfolio.orders)
    # OPEN_LONG (side=0) -> vbt Buy (0).  CLOSE_LONG (side=1) -> vbt Sell (1).
    assert df.iloc[0]["side"] == 0
    assert df.iloc[1]["side"] == 1


def test_to_vbt_orders_empty_when_no_orders() -> None:
    close = pd.DataFrame([[100.0]], columns=["RB"], index=pd.bdate_range("2024-01-02", periods=1))
    p = FuturesPortfolio(
        close=close, specs=(FuturesSpec("RB", 10.0, 0.10),),
        init_cash=10_000.0, freq="1D", bars_per_year=252.0, trading_days_per_year=252,
        _order_records=np.empty(0, dtype=FUTURES_ORDER_DT),
        _cash=np.array([10_000.0]),
        _position=np.array([[0.0]]),
        _margin_locked=np.array([[0.0]]),
    )
    df = p.to_vbt_orders()
    assert len(df) == 0
    assert list(df.columns) == ["id", "col", "idx", "size", "price", "fees", "side"]
