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


def test_from_signals_smoke() -> None:
    """End-to-end: 2 contracts, 5 daily bars, simple entry/exit pattern."""
    import vbt_futures as vbtf

    close = pd.DataFrame(
        [[100.0, 200.0], [101.0, 199.0], [102.0, 198.0], [101.0, 197.0], [103.0, 195.0]],
        columns=["RB", "HC"],
        index=pd.bdate_range("2024-01-02", periods=5),
    )
    long_entries = pd.DataFrame(
        [[True, False], [False, False], [False, True], [False, False], [False, False]],
        columns=close.columns, index=close.index,
    )
    long_exits = pd.DataFrame(
        [[False, False], [False, False], [False, False], [False, False], [True, True]],
        columns=close.columns, index=close.index,
    )
    pf = vbtf.from_signals(
        close=close, long_entries=long_entries, long_exits=long_exits,
        specs=[
            vbtf.FuturesSpec("RB", mult=10.0, margin_rate=0.10),
            vbtf.FuturesSpec("HC", mult=10.0, margin_rate=0.10),
        ],
        init_cash=100_000.0,
        freq="1D",
    )
    assert pf.cash.shape == (5,)
    assert pf.position.shape == (5, 2)
    # RB long: open 100, close 103 -> pnl 1*3*10=30.  HC long: open 198, close 195 -> pnl -30.
    assert len(pf.orders) == 4  # 2 opens + 2 closes
    assert pf.stats()["Total Trades"] == 2


def test_from_signals_propagates_validation_error() -> None:
    """Invalid input (e.g. negative close) should raise ValueError."""
    import vbt_futures as vbtf

    close = pd.DataFrame(
        [[-100.0, 200.0]],
        columns=["RB", "HC"],
        index=pd.bdate_range("2024-01-02", periods=1),
    )
    with pytest.raises(ValueError, match="close 含负值"):
        vbtf.from_signals(
            close=close,
            specs=[
                vbtf.FuturesSpec("RB", mult=10.0, margin_rate=0.10),
                vbtf.FuturesSpec("HC", mult=10.0, margin_rate=0.10),
            ],
            init_cash=100_000.0,
        )


def test_from_signals_size_as_array() -> None:
    """size can be a 1D array (one size per column) instead of a scalar."""
    import vbt_futures as vbtf

    close = pd.DataFrame(
        [[100.0, 200.0]],
        columns=["RB", "HC"],
        index=pd.bdate_range("2024-01-02", periods=1),
    )
    pf = vbtf.from_signals(
        close=close,
        long_entries=pd.DataFrame([[True, True]], columns=close.columns, index=close.index),
        specs=[
            vbtf.FuturesSpec("RB", mult=10.0, margin_rate=0.10),
            vbtf.FuturesSpec("HC", mult=10.0, margin_rate=0.10),
        ],
        size=np.array([2.0, 3.0]),  # 2 lots RB, 3 lots HC
        init_cash=100_000.0,
    )
    assert pf.orders.iloc[0]["size"] == 2.0
    assert pf.orders.iloc[1]["size"] == 3.0


def test_from_signals_size_as_2d_dataframe() -> None:
    """size can also be a (T, N) DataFrame (per-bar, per-column sizes)."""
    import vbt_futures as vbtf

    close = pd.DataFrame(
        [[100.0, 200.0], [101.0, 199.0]],
        columns=["RB", "HC"],
        index=pd.bdate_range("2024-01-02", periods=2),
    )
    long_entries = pd.DataFrame(
        [[True, True], [False, False]],
        columns=close.columns, index=close.index,
    )
    pf = vbtf.from_signals(
        close=close, long_entries=long_entries,
        specs=[
            vbtf.FuturesSpec("RB", mult=10.0, margin_rate=0.10),
            vbtf.FuturesSpec("HC", mult=10.0, margin_rate=0.10),
        ],
        size=pd.DataFrame(
            [[2.0, 3.0], [4.0, 5.0]],
            columns=close.columns, index=close.index,
        ),
        init_cash=100_000.0,
    )
    assert pf.orders.iloc[0]["size"] == 2.0
    assert pf.orders.iloc[1]["size"] == 3.0


def test_from_signals_bars_per_year_override() -> None:
    """When bars_per_year is supplied, it's stored unchanged."""
    import vbt_futures as vbtf

    close = pd.DataFrame(
        [[100.0, 200.0]],
        columns=["RB", "HC"],
        index=pd.bdate_range("2024-01-02", periods=1),
    )
    pf = vbtf.from_signals(
        close=close,
        specs=[
            vbtf.FuturesSpec("RB", mult=10.0, margin_rate=0.10),
            vbtf.FuturesSpec("HC", mult=10.0, margin_rate=0.10),
        ],
        init_cash=100_000.0,
        bars_per_year=4032.0,
    )
    assert pf.bars_per_year == 4032.0


def test_from_signals_short_only() -> None:
    """Only short_entries -> open and close shorts."""
    import vbt_futures as vbtf

    close = pd.DataFrame(
        [[100.0], [99.0], [98.0]],
        columns=["RB"],
        index=pd.bdate_range("2024-01-02", periods=3),
    )
    short_entries = pd.DataFrame(
        [[True], [False], [False]],
        columns=close.columns, index=close.index,
    )
    short_exits = pd.DataFrame(
        [[False], [False], [True]],
        columns=close.columns, index=close.index,
    )
    pf = vbtf.from_signals(
        close=close, short_entries=short_entries, short_exits=short_exits,
        specs=[vbtf.FuturesSpec("RB", mult=10.0, margin_rate=0.10)],
        init_cash=10_000.0,
    )
    # Short: open 100, close 98 -> pnl -1 * (98-100) * 10 = 20.
    assert len(pf.orders) == 2
    assert pf.orders.iloc[0]["side"] == 2  # OPEN_SHORT
    assert pf.orders.iloc[1]["side"] == 3  # CLOSE_SHORT
    assert pf.orders.iloc[1]["pnl"] == 20.0


def test_from_signals_margin_mode_portfolio_runs() -> None:
    """margin_mode='portfolio' should be accepted and produce a valid run."""
    import vbt_futures as vbtf

    close = pd.DataFrame(
        [[100.0, 200.0]] * 80,
        columns=["RB", "HC"],
        index=pd.bdate_range("2024-01-02", periods=80),
    )
    long_entries = pd.DataFrame(
        [[False, False]] * 60 + [[True, True]] * 20,
        columns=close.columns, index=close.index,
    )
    pf = vbtf.from_signals(
        close=close, long_entries=long_entries,
        specs=[
            vbtf.FuturesSpec("RB", mult=10.0, margin_rate=0.10),
            vbtf.FuturesSpec("HC", mult=10.0, margin_rate=0.10),
        ],
        init_cash=100_000.0,
        margin_mode="portfolio",
        margin_lookback=20,
    )
    assert pf.position.iloc[-1, 0] == 1.0
    assert pf.position.iloc[-1, 1] == 1.0


def test_from_signals_invalid_margin_mode_raises() -> None:
    import vbt_futures as vbtf

    close = pd.DataFrame(
        [[100.0, 200.0]],
        columns=["RB", "HC"],
        index=pd.bdate_range("2024-01-02", periods=1),
    )
    with pytest.raises(ValueError, match="margin_mode 必须是"):
        vbtf.from_signals(
            close=close,
            specs=[
                vbtf.FuturesSpec("RB", mult=10.0, margin_rate=0.10),
                vbtf.FuturesSpec("HC", mult=10.0, margin_rate=0.10),
            ],
            init_cash=100_000.0,
            margin_mode="bogus",
        )


def test_from_signals_invalid_margin_lookback_raises() -> None:
    import vbt_futures as vbtf

    close = pd.DataFrame(
        [[100.0, 200.0]],
        columns=["RB", "HC"],
        index=pd.bdate_range("2024-01-02", periods=1),
    )
    with pytest.raises(ValueError, match="margin_lookback 必须是"):
        vbtf.from_signals(
            close=close,
            specs=[
                vbtf.FuturesSpec("RB", mult=10.0, margin_rate=0.10),
                vbtf.FuturesSpec("HC", mult=10.0, margin_rate=0.10),
            ],
            init_cash=100_000.0,
            margin_mode="portfolio",
            margin_lookback=1,
        )


def test_from_signals_equity_proportional_sizing() -> None:
    import vbt_futures as vbtf

    close = pd.DataFrame(
        [[100.0]] * 10,
        columns=["RB"],
        index=pd.bdate_range("2024-01-02", periods=10),
    )
    long_entries = pd.DataFrame(
        [[True], [False], [False], [False], [False], [False], [False], [False], [False], [False]],
        columns=close.columns, index=close.index,
    )
    pf = vbtf.from_signals(
        close=close, long_entries=long_entries,
        specs=[vbtf.FuturesSpec("RB", mult=10.0, margin_rate=0.10)],
        init_cash=10_000.0,
        sizing_mode="equity_proportional",
        size=1.0,
    )
    # Just verify the run succeeded and produced a position.
    assert pf.position.iloc[-1, 0] == 1.0


def test_from_signals_anti_martingale_sizing() -> None:
    import vbt_futures as vbtf

    close = pd.DataFrame(
        [[100.0]] * 10,
        columns=["RB"],
        index=pd.bdate_range("2024-01-02", periods=10),
    )
    long_entries = pd.DataFrame(
        [[True], [False], [False], [False], [False], [False], [False], [False], [False], [False]],
        columns=close.columns, index=close.index,
    )
    pf = vbtf.from_signals(
        close=close, long_entries=long_entries,
        specs=[vbtf.FuturesSpec("RB", mult=10.0, margin_rate=0.10)],
        init_cash=10_000.0,
        sizing_mode="anti_martingale",
        size=1.0,
        sizing_kwargs={"trigger_pnl": 100.0, "max_size": 5.0},
    )
    assert pf.position.iloc[-1, 0] == 1.0


def test_from_signals_invalid_sizing_mode_raises() -> None:
    import vbt_futures as vbtf

    close = pd.DataFrame(
        [[100.0, 200.0]],
        columns=["RB", "HC"],
        index=pd.bdate_range("2024-01-02", periods=1),
    )
    with pytest.raises(ValueError, match="sizing_mode 必须是"):
        vbtf.from_signals(
            close=close,
            specs=[
                vbtf.FuturesSpec("RB", mult=10.0, margin_rate=0.10),
                vbtf.FuturesSpec("HC", mult=10.0, margin_rate=0.10),
            ],
            init_cash=100_000.0,
            sizing_mode="bogus",
        )
