"""Stress test: 10 contracts x 1000 days.

Goal:
1. System stability check: does the simulator handle 10x1000 cleanly?
2. Auto-sizing check: do ``equity_proportional`` and ``anti_martingale``
   actually respond to changes in total account equity?

The data is HAND-CRAFTED with 10 cycles of 50-day up + 50-day down trends
(so equity swings by 1.65x per up-cycle and 1/1.65x per down-cycle).
This gives a CLEAR visual signal of when size scales up and down.

Run:
    python examples/stress_test_10x1000.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import vbt_futures as vbtf


# ---------------------------------------------------------------------------
# Data generation: 10 contracts with strong, repeated trends.
# ---------------------------------------------------------------------------
def make_stress_data(
    n_days: int = 1000,
    n_contracts: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """10 contracts with STRONG trends.

    Layout (1000 days, 10 cycles of 100 days):
      - 50 days UP   at +1% daily drift
      - 50 days DOWN at -1% daily drift

    Per-cycle equity swing: 1.65x (up) or 0.6x (down).
    Cumulative effect: 1.65^10 ≈ 149x if you stay long, or volatile
    wave if you hold through both directions.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    daily_drift = np.zeros(n_days)
    for cycle_start in range(0, n_days, 100):
        up_end = min(cycle_start + 50, n_days)
        dn_end = min(cycle_start + 100, n_days)
        daily_drift[cycle_start:up_end] =  0.01
        daily_drift[up_end:dn_end]     = -0.01

    base_prices = np.array([100, 200, 50, 80, 120, 90, 60, 75, 110, 95], dtype=float)
    n = len(base_prices)
    idio = rng.normal(0.0, 0.003, size=(n_days, n))
    log_ret = daily_drift[:, None] + idio
    prices = base_prices[None, :] * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(prices, columns=[f"C{i}" for i in range(n)], index=idx)


# ---------------------------------------------------------------------------
# Signal generators.
# ---------------------------------------------------------------------------
def make_uponly_signal(close: pd.DataFrame) -> tuple:
    """Long at the start of each up-trend, exit at the end (no shorts).

    Equity grows monotonically as we capture every up-phase.
    """
    T, N = close.shape
    long_entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    long_exits   = pd.DataFrame(False, index=close.index, columns=close.columns)
    short_entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    short_exits   = pd.DataFrame(False, index=close.index, columns=close.columns)
    for cycle_start in range(0, T, 100):
        up_start = cycle_start
        up_end   = min(cycle_start + 50, T)
        if up_start < T:
            long_entries.iloc[up_start] = True
        if up_end < T:
            long_exits.iloc[up_end] = True
    return long_entries, long_exits, short_entries, short_exits


def make_holdthrough_signal(close: pd.DataFrame) -> tuple:
    """Buy at bar 0, hold through EVERYTHING until the end.

    Equity swings up/down with the cycles - peak at every up-end,
    dip at every down-end.
    """
    T, N = close.shape
    long_entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    long_exits   = pd.DataFrame(False, index=close.index, columns=close.columns)
    short_entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    short_exits   = pd.DataFrame(False, index=close.index, columns=close.columns)
    long_entries.iloc[0] = True
    return long_entries, long_exits, short_entries, short_exits


def make_oscillating_signal(close: pd.DataFrame) -> tuple:
    """Capture BOTH up- and down-phases (long then short alternating).

    Equity oscillates but with mild overall gain (because the strategy
    alternates directions each cycle, no compounding).

    More importantly: many entries at different equity levels -> shows
    scaling both up and down.
    """
    T, N = close.shape
    long_entries  = pd.DataFrame(False, index=close.index, columns=close.columns)
    long_exits    = pd.DataFrame(False, index=close.index, columns=close.columns)
    short_entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    short_exits   = pd.DataFrame(False, index=close.index, columns=close.columns)
    for cycle_start in range(0, T, 100):
        up_start = cycle_start
        up_end   = min(cycle_start + 50, T)
        dn_start = up_end
        dn_end   = min(cycle_start + 100, T)
        # Long the up-phase.
        if up_start < T:
            long_entries.iloc[up_start] = True
        if up_end < T:
            long_exits.iloc[up_end] = True
        # Short the down-phase.
        if dn_start < T:
            short_entries.iloc[dn_start] = True
        if dn_end < T:
            short_exits.iloc[dn_end] = True
    return long_entries, long_exits, short_entries, short_exits


def make_losing_then_winning_signal(close: pd.DataFrame) -> tuple:
    """Buy at the TOP of each up-trend (BAD entry timing) -> lose money.

    After losing, equity drops -> size shrinks (smaller loss next time).
    After each down-trend, the strategy exits with the loss and waits.
    Demonstrates SIZE SCALING DOWN on losses.
    """
    T, N = close.shape
    long_entries  = pd.DataFrame(False, index=close.index, columns=close.columns)
    long_exits    = pd.DataFrame(False, index=close.index, columns=close.columns)
    short_entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    short_exits   = pd.DataFrame(False, index=close.index, columns=close.columns)
    # Buy at start of each down-trend (i.e., right after the peak).  We LOSE
    # on every cycle since price keeps falling.  Equity decreases, size shrinks.
    for cycle_start in range(0, T, 100):
        up_end   = min(cycle_start + 50, T)
        dn_start = up_end
        if dn_start < T:
            long_entries.iloc[dn_start] = True           # bad entry
        dn_end   = min(cycle_start + 100, T)
        if dn_end < T:
            long_exits.iloc[dn_end] = True              # exit at the bottom
    return long_entries, long_exits, short_entries, short_exits


# ---------------------------------------------------------------------------
# Specs and helpers.
# ---------------------------------------------------------------------------
def make_specs(n: int) -> list[vbtf.FuturesSpec]:
    """Realistic specs for 10 contracts: vary mult and margin_rate."""
    return [
        vbtf.FuturesSpec(
            symbol=f"C{i}", mult=10.0 if i % 2 == 0 else 5.0,
            margin_rate=0.10 + (i % 3) * 0.02,
            fees=2e-4, fixed_fees=2.0,
        )
        for i in range(n)
    ]


def run_with_timing(label: str, **kwargs) -> vbtf.FuturesPortfolio:
    """Run a backtest with timing and print compact stats."""
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
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(summary.to_string())
    print(f"  Wall time: {elapsed*1000:.1f} ms")
    return pf


def show_size_timeline(pf, label: str) -> None:
    """Show size-on-entry timeline demonstrating scaling."""
    eq_arr = pf.equity.values
    pos_arr = pf.position.values
    any_pos = (pos_arr != 0).any(axis=1)
    prev_any = np.concatenate(([False], any_pos[:-1]))
    entry_bars = np.where(any_pos & ~prev_any)[0]
    init_cash = pf.init_cash

    if len(entry_bars) == 0:
        # No entries -> can't show size scaling.  Skip the size timeline.
        T = len(eq_arr)
        print(f"\n  No entries in {label}; size doesn't scale (size = 1.0).")
        print(f"  Equity still varies with the data (see Scenario A for big swings).")
        return

    print(f"\n  SIZE-ON-ENTRY TIMELINE  ({label}, equity_proportional)")
    print(f"  {'bar':>5}  {'equity':>14}  {'change%':>8}  {'scale':>7}  {'lots':>6}")
    prev_eq = init_cash
    min_scale = np.inf
    max_scale = -np.inf
    for t in entry_bars:
        size_eq = eq_arr[t] / init_cash
        pct_change = (eq_arr[t] / prev_eq - 1.0) * 100
        min_scale = min(min_scale, size_eq)
        max_scale = max(max_scale, size_eq)
        print(f"  t={t:>4}  {eq_arr[t]:>14,.0f}  {pct_change:>+7.2f}%  "
              f"{size_eq:>6.3f}x  {size_eq:>5.2f}")
        prev_eq = eq_arr[t]

    swing_pct = (max_scale - min_scale) / min_scale * 100
    print(f"\n  Size range across entries: {min_scale:.3f}x .. {max_scale:.3f}x  "
          f"({swing_pct:.0f}% swing)")

    # Equity curve at key cycle boundaries.
    T = len(eq_arr)
    print(f"\n  Equity curve at cycle boundaries:")
    print(f"  {'bar':>5}  {'equity':>14}  {'cycle phase'}")
    sample_bars = [0, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 950, 999]
    for t in sample_bars:
        if t < T:
            phase = "↑ uptrend" if (t // 50) % 2 == 0 else "↓ downtrend"
            print(f"  t={t:>4}  {eq_arr[t]:>14,.0f}  {phase}")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("vbt-futures stress test: 10 contracts x 1000 days")
    print("=" * 70)

    print("\n[1/3] Generating synthetic data (10 cycles, +1%/-1% daily drift) ...")
    close = make_stress_data(n_days=1000, n_contracts=10)
    print(f"      close shape: {close.shape}")
    print(f"      base price range: {close.iloc[0].min():.1f} - {close.iloc[0].max():.1f}")
    print(f"      final price range: {close.iloc[-1].min():.1f} - {close.iloc[-1].max():.1f}")
    print(f"      expected: each up-cycle grows equity 1.65x; each down-cycle shrinks 0.6x")

    specs = make_specs(10)
    init_cash = 1_000_000.0

    # -----------------------------------------------------------------
    # SCENARIO A: "up-only" strategy -> equity only grows.
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCENARIO A: UP-only strategy (capture only the up-phases)")
    print("=" * 70)
    leA, lxA, seA, sxA = make_uponly_signal(close)
    n_long = leA.sum().sum()
    print(f"  {n_long} long entries (10 up-cycles), 0 short entries")

    common_A = dict(
        close=close, long_entries=leA, long_exits=lxA,
        short_entries=seA, short_exits=sxA,
        specs=specs, init_cash=init_cash, freq="1D",
    )
    pf_A_fixed = run_with_timing("A1: fixed sizing (10.0 lots)", **common_A, size=10.0)
    pf_A_eqprop = run_with_timing(
        "A2: equity_proportional sizing (10 lots base)", **common_A,
        sizing_mode="equity_proportional", size=10.0,
    )
    show_size_timeline(pf_A_eqprop, "SCENARIO A (UP-only)")

    # -----------------------------------------------------------------
    # SCENARIO B: "hold-through" strategy -> equity swings up AND down.
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCENARIO B: HOLD-through strategy (1 entry, never exits)")
    print("=" * 70)
    print("  1 long entry at bar 0, no exits.  Equity follows the data waves.")
    leB, lxB, seB, sxB = make_holdthrough_signal(close)

    common_B = dict(
        close=close, long_entries=leB, long_exits=lxB,
        short_entries=seB, short_exits=sxB,
        specs=specs, init_cash=init_cash, freq="1D",
    )
    pf_B_fixed = run_with_timing("B1: fixed sizing (10.0 lots)", **common_B, size=10.0)
    pf_B_eqprop = run_with_timing(
        "B2: equity_proportional sizing", **common_B,
        sizing_mode="equity_proportional", size=10.0,
    )
    show_size_timeline(pf_B_eqprop, "SCENARIO B (HOLD-through)")

    # -----------------------------------------------------------------
    # SCENARIO C: "bad-timing" strategy -> lose money -> size shrinks.
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCENARIO C: BAD-timing strategy (size scales DOWN on losses)")
    print("=" * 70)
    leC, lxC, seC, sxC = make_losing_then_winning_signal(close)
    print(f"  10 bad entries + 10 exits (buy at the top, exit at the bottom)")

    common_C = dict(
        close=close, long_entries=leC, long_exits=lxC,
        short_entries=seC, short_exits=sxC,
        specs=specs, init_cash=init_cash, freq="1D",
    )
    pf_C_fixed = run_with_timing("C1: fixed sizing (10.0 lots)", **common_C, size=10.0)
    pf_C_eqprop = run_with_timing(
        "C2: equity_proportional sizing (10 lots base)", **common_C,
        sizing_mode="equity_proportional", size=10.0,
    )
    show_size_timeline(pf_C_eqprop, "SCENARIO C (BAD-timing)")

    # -----------------------------------------------------------------
    # SCENARIO D: alternating long/short -> many entries, mixed equity.
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SCENARIO D: ALTERNATING long/short (oscillating equity, many entries)")
    print("=" * 70)
    leD, lxD, seD, sxD = make_oscillating_signal(close)
    n_leD = leD.sum().sum()
    n_seD = seD.sum().sum()
    print(f"  {n_leD} long + {n_seD} short entries (20 total)")

    common_D = dict(
        close=close, long_entries=leD, long_exits=lxD,
        short_entries=seD, short_exits=sxD,
        specs=specs, init_cash=init_cash, freq="1D",
    )
    pf_D_eqprop = run_with_timing(
        "D: equity_proportional sizing (10 lots base)", **common_D,
        sizing_mode="equity_proportional", size=10.0,
    )
    show_size_timeline(pf_D_eqprop, "SCENARIO D (ALTERNATING)")

    # -----------------------------------------------------------------
    # Comparison summary.
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON (Scenario B - hold-through)")
    print("=" * 70)
    print(f"  {'metric':<25}  {'fixed':>16}  {'equity_prop':>16}")
    s_fix = pf_B_fixed.stats()
    s_eqp = pf_B_eqprop.stats()
    rows = [
        ("Final Equity",          f"{s_fix['Final Equity']:>16,.0f}",  f"{s_eqp['Final Equity']:>16,.0f}"),
        ("Total Return [%]",      f"{s_fix['Total Return [%]']:>16.2f}", f"{s_eqp['Total Return [%]']:>16.2f}"),
        ("Max Drawdown [%]",      f"{s_fix['Max Drawdown [%]']:>16.2f}", f"{s_eqp['Max Drawdown [%]']:>16.2f}"),
        ("Sharpe Ratio",          f"{s_fix['Sharpe Ratio']:>16.3f}",      f"{s_eqp['Sharpe Ratio']:>16.3f}"),
        ("Total Fees",            f"{s_fix['Total Fees']:>16,.2f}",     f"{s_eqp['Total Fees']:>16,.2f}"),
        ("Final Margin Locked",   f"{pf_B_fixed.margin_locked.iloc[-1].sum():>16,.2f}",
                                    f"{pf_B_eqprop.margin_locked.iloc[-1].sum():>16,.2f}"),
    ]
    for name, a, b in rows:
        print(f"  {name:<25}  {a}  {b}")

    # -----------------------------------------------------------------
    # Write the size-vs-equity plots for Scenarios A and C.
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("WRITING PLOTS")
    print("=" * 70)
    os.makedirs("output", exist_ok=True)
    # Scenario A: size grows with equity.
    fig_A = pf_A_eqprop.plot_sizing(
        base_size=10.0, sizing_mode="equity_proportional",
    )
    out_A = "output/stress_A_sizing_grows.html"
    fig_A.write_html(out_A)
    print(f"  Scenario A (size GROWS with equity):  {out_A}")

    # Scenario C: size shrinks with equity.
    fig_C = pf_C_eqprop.plot_sizing(
        base_size=10.0, sizing_mode="equity_proportional",
    )
    out_C = "output/stress_C_sizing_shrinks.html"
    fig_C.write_html(out_C)
    print(f"  Scenario C (size SHRINKS with equity): {out_C}")


if __name__ == "__main__":
    main()
