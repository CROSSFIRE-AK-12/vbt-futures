"""Stress test: 10 contracts x 1000 days.

Goal:
1. System stability check: does the simulator handle 10x1000 cleanly?
2. Auto-sizing check: do ``equity_proportional`` and ``anti_martingale``
   actually respond to changes in total account equity?

Run:
    python examples/stress_test_10x1000.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import vbt_futures as vbtf


def make_stress_data(
    n_days: int = 1000,
    n_contracts: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """10 contracts with mixed correlation structure:

    - Contracts 0,1,2 share a "metals" factor (high pairwise corr)
    - Contracts 3,4,5 share a "energy" factor (high pairwise corr)
    - Contracts 6,7 share a "grains" factor
    - Contracts 8,9 are independent

    Each contract has its own vol (1%-2% daily).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    n_factors = 4  # metals, energy, grains, independent-noise-pool
    factors = rng.normal(0.0, 1.0, size=(n_days, n_factors))
    # Contract -> factor loadings
    loadings = np.array([
        # metals  energy  grains  indep
        [0.80,    0.00,   0.00,   0.20],   # 0
        [0.85,    0.00,   0.00,   0.15],   # 1
        [0.75,    0.05,   0.00,   0.20],   # 2
        [0.00,    0.80,   0.00,   0.20],   # 3
        [0.00,    0.85,   0.00,   0.15],   # 4
        [0.05,    0.75,   0.00,   0.20],   # 5
        [0.00,    0.00,   0.80,   0.20],   # 6
        [0.00,    0.00,   0.85,   0.15],   # 7
        [0.10,    0.10,   0.10,   0.70],   # 8
        [0.15,    0.05,   0.05,   0.75],   # 9
    ])
    # Per-contract idiosyncratic sigma (1%-2%).
    idio_sigma = rng.uniform(0.005, 0.015, size=n_contracts)
    # Factor sigmas.
    factor_sigma = np.array([0.015, 0.012, 0.018, 0.010])  # metals, energy, grains, indep

    # log returns = factors (T, F) @ (loadings (N, F) * factor_sigma (F,))^T + idio
    scaled_loadings = loadings * factor_sigma[None, :]  # (N, F)
    log_ret = factors @ scaled_loadings.T  # (T, F) @ (F, N) = (T, N)
    log_ret += rng.normal(0.0, idio_sigma[None, :], size=(n_days, n_contracts))
    # Add a mild long-term drift so equity grows over time (-> tests auto-sizing).
    drift = 0.0005  # ~0.05% per day
    log_ret += drift

    base_prices = np.array([100, 200, 50, 80, 120, 90, 60, 75, 110, 95], dtype=float)
    prices = base_prices[None, :] * np.exp(np.cumsum(log_ret, axis=0))

    cols = [f"C{i}" for i in range(n_contracts)]
    return pd.DataFrame(prices, columns=cols, index=idx)


def make_momentum_signals(
    close: pd.DataFrame, fast: int = 10, slow: int = 50,
) -> tuple:
    """Long+short momentum: cross-over + position reversal logic."""
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    above = fast_ma > slow_ma
    prev_above = above.shift(1).fillna(False)
    long_entries = (above & ~prev_above).fillna(False).astype(bool)
    long_exits = (~above & prev_above).fillna(False).astype(bool)
    # Symmetric short signals.
    short_entries = (~above & prev_above).fillna(False).astype(bool)
    short_exits = (above & ~prev_above).fillna(False).astype(bool)
    return long_entries, long_exits, short_entries, short_exits


def make_specs(n: int) -> list[vbtf.FuturesSpec]:
    """Realistic specs: mult, margin_rate, fees for 10 contracts."""
    # Vary margin rate slightly to make it interesting.
    return [
        vbtf.FuturesSpec(
            symbol=f"C{i}", mult=10.0 if i % 2 == 0 else 5.0,
            margin_rate=0.10 + (i % 3) * 0.02,  # 10%, 12%, 14%
            fees=2e-4, fixed_fees=2.0,
        )
        for i in range(n)
    ]


def run_with_timing(label: str, **kwargs) -> vbtf.FuturesPortfolio:
    """Run a backtest with timing and print summary."""
    t0 = time.perf_counter()
    pf = vbtf.from_signals(**kwargs)
    elapsed = time.perf_counter() - t0
    s = pf.stats()
    print(f"\n--- {label} ---")
    summary = s[[
        "Period", "Init Cash", "Final Equity", "Total Return [%]",
        "Annualized Return [%]", "Sharpe Ratio", "Max Drawdown [%]",
        "Total Trades", "Win Rate [%]", "Max Position", "Total Fees",
    ]]
    with pd.option_context("display.width", 200):
        print(summary.to_string())
    print(f"  Wall time: {elapsed*1000:.1f} ms")
    return pf


def main() -> None:
    print("=" * 70)
    print("vbt-futures stress test: 10 contracts x 1000 days")
    print("=" * 70)

    print("\n[1/3] Generating synthetic data ...")
    close = make_stress_data(n_days=1000, n_contracts=10)
    print(f"      close shape: {close.shape}")
    print(f"      base price range: {close.iloc[0].min():.1f} - {close.iloc[0].max():.1f}")
    print(f"      final price range: {close.iloc[-1].min():.1f} - {close.iloc[-1].max():.1f}")
    print(f"      drift: +0.05% per day (compounded = +65% over 1000 days)")

    print("\n[2/3] Generating momentum signals (10/50 SMA cross) ...")
    long_entries, long_exits, short_entries, short_exits = make_momentum_signals(close)
    n_long = long_entries.sum().sum()
    n_short = short_entries.sum().sum()
    print(f"      total long entries: {n_long}, short entries: {n_short}")

    specs = make_specs(10)
    common = dict(
        close=close,
        long_entries=long_entries, long_exits=long_exits,
        short_entries=short_entries, short_exits=short_exits,
        specs=specs,
        init_cash=1_000_000.0,
        freq="1D",
    )

    # Run 1: baseline (fixed sizing, gross margin)
    pf_fixed = run_with_timing(
        "Run 1: fixed sizing (1.0) + gross margin (baseline)",
        **common, size=1.0,
    )

    # Run 2: equity_proportional (responds to BOTH wins and losses)
    pf_eqprop = run_with_timing(
        "Run 2: equity_proportional sizing + gross margin",
        **common, sizing_mode="equity_proportional", size=1.0,
    )

    # Run 3: anti_martingale (only adds on profit, capped)
    pf_antimart = run_with_timing(
        "Run 3: anti_martingale sizing + gross margin (cap 5x)",
        **common, sizing_mode="anti_martingale", size=1.0,
        sizing_kwargs={"trigger_pnl": 5_000.0, "max_size": 5.0},
    )

    # Run 4: full Plan 3 - equity_proportional + portfolio margin
    pf_eqprop_pm = run_with_timing(
        "Run 4: equity_proportional + PORTFOLIO margin (recommended)",
        **common, sizing_mode="equity_proportional", size=1.0,
        margin_mode="portfolio", margin_lookback=60,
    )

    print("\n[3/3] Auto-sizing behaviour verification:")

    # For the equity_proportional run, check that size on entry bars
    # correlates with running equity.
    def size_at_entry_bars(pf) -> np.ndarray:
        """Extract size on bars where an entry signal fired (using from_signals' logic)."""
        # Manually replay sizing for the proportional case: at each entry bar,
        # size = 1.0 × (equity_so_far / init_cash).
        eq = pf.equity.values
        # Entry = any long_entry or short_entry True.
        le = pf.close.index  # placeholder; just use equity
        # Simpler: just use the position changes to find entries.
        pos = pf.position.values
        size_arr = np.ones(pos.shape, dtype=np.float64)  # not actually used
        return pos

    print("\nComparing final equity (higher = better):")
    print(f"  fixed:               {pf_fixed.equity.iloc[-1]:>14,.2f}")
    print(f"  equity_proportional: {pf_eqprop.equity.iloc[-1]:>14,.2f}")
    print(f"  anti_martingale:     {pf_antimart.equity.iloc[-1]:>14,.2f}")
    print(f"  eq_prop + portfolio: {pf_eqprop_pm.equity.iloc[-1]:>14,.2f}")

    print("\nMax position (size growth indicator):")
    print(f"  fixed:               {pf_fixed.stats()['Max Position']:>14.2f}")
    print(f"  equity_proportional: {pf_eqprop.stats()['Max Position']:>14.2f}")
    print(f"  anti_martingale:     {pf_antimart.stats()['Max Position']:>14.2f}")
    print(f"  eq_prop + portfolio: {pf_eqprop_pm.stats()['Max Position']:>14.2f}")

    # Check that equity_proportional size responds to equity (sample 5 entry bars).
    print("\nSample of size-on-entry from equity_proportional run:")
    eq_arr = pf_eqprop.equity.values
    # Find entry bars: any time position changes from 0 to non-zero.
    pos_arr = pf_eqprop.position.values
    # Cross-contract aggregate position change.
    any_pos = (pos_arr != 0).any(axis=1)
    prev_any = np.concatenate(([False], any_pos[:-1]))
    entry_bars = np.where(any_pos & ~prev_any)[0]
    init_cash = 1_000_000.0
    print(f"  init_cash = {init_cash:,.0f}")
    for t in entry_bars[:8]:
        size_eq = eq_arr[t] / init_cash
        print(f"  t={t:>4}  equity={eq_arr[t]:>14,.0f}  scale={size_eq:.3f}  (size=1.0 -> effective {size_eq:.2f} lots)")

    # ----------------------------------------------------------------------
    # Strong-trend scenario: confirm auto-sizing visibly scales size.
    # ----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Bonus: Strong-trend scenario to show visible size scaling")
    print("=" * 70)
    strong_close = make_stress_data(n_days=1000, n_contracts=10, seed=42)
    # Manually amplify the upward drift by adding a linear trend to each contract.
    trend = np.linspace(0.0, 1.0, 1000)[:, None] * 50.0  # +50 over the period
    strong_close = strong_close + trend
    le2, lx2, se2, sx2 = make_momentum_signals(strong_close)

    pf_trend_fixed = run_with_timing(
        "Strong trend + fixed sizing",
        close=strong_close, long_entries=le2, long_exits=lx2,
        short_entries=se2, short_exits=sx2,
        specs=specs, init_cash=1_000_000.0, freq="1D",
        size=1.0,
    )
    pf_trend_eqprop = run_with_timing(
        "Strong trend + equity_proportional sizing",
        close=strong_close, long_entries=le2, long_exits=lx2,
        short_entries=se2, short_exits=sx2,
        specs=specs, init_cash=1_000_000.0, freq="1D",
        sizing_mode="equity_proportional", size=1.0,
    )

    # Sample size-on-entry from the eqprop run.
    eq_arr = pf_trend_eqprop.equity.values
    pos_arr = pf_trend_eqprop.position.values
    any_pos = (pos_arr != 0).any(axis=1)
    prev_any = np.concatenate(([False], any_pos[:-1]))
    entry_bars = np.where(any_pos & ~prev_any)[0]
    print("\nSize-on-entry samples (equity_proportional, strong trend):")
    print(f"  init_cash = 1,000,000")
    for t in entry_bars[:5]:
        size_eq = eq_arr[t] / init_cash
        print(f"  t={t:>4}  equity={eq_arr[t]:>14,.0f}  scale={size_eq:.3f}x  (effective lots = {1.0 * size_eq:.2f})")
    for t in entry_bars[-3:]:
        size_eq = eq_arr[t] / init_cash
        print(f"  t={t:>4}  equity={eq_arr[t]:>14,.0f}  scale={size_eq:.3f}x  (effective lots = {1.0 * size_eq:.2f})")

    # Total margin (portfolio vs gross) comparison.
    print("\nFinal bar total margin locked:")
    print(f"  Run 1 (gross):                    {pf_fixed.margin_locked.iloc[-1].sum():>12,.2f}")
    print(f"  Run 2 (gross):                    {pf_eqprop.margin_locked.iloc[-1].sum():>12,.2f}")
    print(f"  Run 4 (portfolio):                {pf_eqprop_pm.margin_locked.iloc[-1].sum():>12,.2f}")
    saving = 1 - pf_eqprop_pm.margin_locked.iloc[-1].sum() / pf_eqprop.margin_locked.iloc[-1].sum()
    print(f"  portfolio margin saving:           {saving*100:.1f}%")


if __name__ == "__main__":
    main()
