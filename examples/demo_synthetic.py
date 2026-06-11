"""vbt-futures demo: 3 synthetic contracts + dual-moving-average strategy.

Showcases all major features (Plan 3):
- Multi-contract (3 columns) portfolio
- Portfolio Margin (data-driven covariance)
- 3 sizing modes (fixed / equity_proportional / anti_martingale)
- stats() / plot() / orders / trades

Run:
    python examples/demo_synthetic.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import vbt_futures as vbtf


def make_synthetic_close(
    n_days: int,
    n_contracts: int,
    base_prices: list[float],
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a (n_days, n_contracts) close DataFrame with random-walk prices.

    Columns 0, 1 are highly correlated; column 2 is independent -> good
    demo for Portfolio Margin discount effect.
    """
    rng = np.random.default_rng(seed)
    cols = [f"CONTRACT_{i}" for i in range(n_contracts)]
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    out = np.empty((n_days, n_contracts), dtype=np.float64)
    # Shared noise for cols 0, 1 (high correlation); independent for col 2.
    shared = rng.normal(0.0, 0.01, size=n_days)
    for c in range(n_contracts):
        if c < 2:
            ret = shared + rng.normal(0.0, 0.002, size=n_days)  # tiny idiosyncratic
        else:
            ret = rng.normal(0.0, 0.015, size=n_days)  # independent
        prices = base_prices[c] * np.exp(np.cumsum(ret))
        out[:, c] = prices
    return pd.DataFrame(out, columns=cols, index=idx)


def make_sma_signals(close: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.DataFrame:
    """Long-only cross-over signals: long_entries = fast crosses above slow."""
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    above = fast_ma > slow_ma
    prev_above = above.shift(1).fillna(False)
    long_entries = (above & ~prev_above).fillna(False).astype(bool)
    long_exits = (~above & prev_above).fillna(False).astype(bool)
    return long_entries, long_exits


def run_backtest(label: str, **kwargs) -> vbtf.FuturesPortfolio:
    """Run a backtest and print compact stats."""
    print(f"\n--- {label} ---")
    pf = vbtf.from_signals(**kwargs)
    s = pf.stats()
    summary = s[[
        "Init Cash", "Final Equity", "Total Return [%]",
        "Annualized Return [%]", "Sharpe Ratio", "Max Drawdown [%]",
        "Total Trades", "Win Rate [%]", "Max Position", "Total Fees",
    ]]
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(summary.to_string())
    return pf


def main() -> None:
    print("[1/5] Generating 500-day synthetic data for 3 contracts ...")
    close = make_synthetic_close(
        n_days=500, n_contracts=3,
        base_prices=[100.0, 200.0, 50.0],
    )
    print(f"      close shape: {close.shape}")
    print(f"      cols 0, 1 highly correlated; col 2 independent (good for PMargin demo)")

    print("[2/5] Generating SMA(5/20) cross-over signals ...")
    long_entries, long_exits = make_sma_signals(close, fast=5, slow=20)

    common = dict(
        close=close,
        long_entries=long_entries,
        long_exits=long_exits,
        specs=[
            vbtf.FuturesSpec("CONTRACT_0", mult=10.0, margin_rate=0.10, fees=2e-4),
            vbtf.FuturesSpec("CONTRACT_1", mult=10.0, margin_rate=0.10, fees=2e-4),
            vbtf.FuturesSpec("CONTRACT_2", mult=100.0, margin_rate=0.12, fees=2e-4),
        ],
        init_cash=200_000.0,
        freq="1D",
    )

    # Run 3 variants to showcase Plan 3 features.
    pf_fixed_gross = run_backtest(
        "Run 1: fixed sizing + gross margin (baseline)",
        **common, size=1.0,
    )

    pf_eqprop_gross = run_backtest(
        "Run 2: equity_proportional sizing + gross margin",
        **common, sizing_mode="equity_proportional", size=1.0,
    )

    pf_antimart_portfolio = run_backtest(
        "Run 3: anti_martingale sizing + PORTFOLIO margin (the new stuff)",
        **common,
        sizing_mode="anti_martingale", size=1.0,
        sizing_kwargs={"trigger_pnl": 1000.0, "max_size": 5.0},
        margin_mode="portfolio", margin_lookback=60,
    )

    print("\n[3/5] Comparing total margin locked (last bar):")
    print(f"  Run 1 (gross):              {pf_fixed_gross.margin_locked.iloc[-1].sum():>12,.2f}")
    print(f"  Run 2 (gross):              {pf_eqprop_gross.margin_locked.iloc[-1].sum():>12,.2f}")
    print(f"  Run 3 (portfolio, discount): {pf_antimart_portfolio.margin_locked.iloc[-1].sum():>12,.2f}")
    print("  -> portfolio margin releases cash for diversification benefit.")

    print("\n[4/5] Plotting the most interesting run (anti_martingale + portfolio margin) ...")
    try:
        os.makedirs("output", exist_ok=True)
        fig = pf_antimart_portfolio.plot()
        out_path = "output/demo_antimart_portfolio.html"
        fig.write_html(out_path)
        print(f"      saved to: {out_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"      (plot skipped: {exc})")

    print("\n[5/5] Trade log (first 5 trades):")
    print(pf_antimart_portfolio.trades.head().to_string())


if __name__ == "__main__":
    main()
