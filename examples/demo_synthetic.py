"""vbt-futures demo: 3 synthetic contracts + dual-moving-average strategy.

Run:
    python examples/demo_synthetic.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import vbt_futures as vbtf


def make_synthetic_close(
    n_days: int,
    n_contracts: int,
    base_prices: list[float],
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a (n_days, n_contracts) close DataFrame with random-walk prices."""
    rng = np.random.default_rng(seed)
    cols = [f"CONTRACT_{i}" for i in range(n_contracts)]
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    out = np.empty((n_days, n_contracts), dtype=np.float64)
    for c in range(n_contracts):
        log_ret = rng.normal(0.0, 0.01, size=n_days)  # 1% daily sigma
        prices = base_prices[c] * np.exp(np.cumsum(log_ret))
        out[:, c] = prices
    return pd.DataFrame(out, columns=cols, index=idx)


def make_sma_signals(close: pd.DataFrame, fast: int = 5, slow: int = 20) -> tuple:
    """Cross-over signals: long_entries = fast crosses above slow, etc."""
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    above = fast_ma > slow_ma
    prev_above = above.shift(1).fillna(False)
    long_entries = above & ~prev_above
    long_exits = ~above & prev_above
    return (
        long_entries.fillna(False).astype(bool),
        long_exits.fillna(False).astype(bool),
    )


def main() -> None:
    print("[1/4] Generating 500-day synthetic data for 3 contracts ...")
    close = make_synthetic_close(
        n_days=500, n_contracts=3,
        base_prices=[100.0, 200.0, 50.0],
    )
    print(f"      close shape: {close.shape}")

    print("[2/4] Generating SMA(5/20) cross-over signals ...")
    long_entries, long_exits = make_sma_signals(close, fast=5, slow=20)

    print("[3/4] Running vbt-futures backtest ...")
    pf = vbtf.from_signals(
        close=close,
        long_entries=long_entries,
        long_exits=long_exits,
        specs=[
            vbtf.FuturesSpec("CONTRACT_0", mult=10.0, margin_rate=0.10, fees=2e-4),
            vbtf.FuturesSpec("CONTRACT_1", mult=10.0, margin_rate=0.10, fees=2e-4),
            vbtf.FuturesSpec("CONTRACT_2", mult=100.0, margin_rate=0.12, fees=2e-4),
        ],
        size=1.0,
        init_cash=200_000.0,
        freq="1D",
    )

    print("[4/4] Stats:")
    stats = pf.stats()
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(stats.to_string())

    # Optionally save a plot.
    try:
        import os
        os.makedirs("output", exist_ok=True)
        fig = pf.plot()
        out_path = "output/demo_synthetic.html"
        fig.write_html(out_path)
        print(f"\nPlot saved to: {out_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
