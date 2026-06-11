"""Micro-benchmark for vbt-futures simulator.

Measures per-call time of simulate_futures_nb on a 1000-bar x 5-col grid.
Target: < 5 ms/call after JIT warmup.

NOTE: do NOT run with NUMBA_DISABLE_JIT=1 — the whole point is to measure JIT speed.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from vbt_futures.simulator import simulate_futures_nb


def main() -> None:
    T, N = 1000, 5
    rng = np.random.default_rng(0)
    close = np.cumprod(1.0 + rng.normal(0.0, 0.01, (T, N)), axis=0) * 100.0
    long_entries = np.zeros((T, N), dtype=bool)
    long_entries[::50, :] = True
    long_exits = np.zeros((T, N), dtype=bool)
    long_exits[25::50, :] = True
    size = np.ones((T, N), dtype=np.float64)
    mult = np.full(N, 10.0)
    margin_rate = np.full(N, 0.10)
    fees = np.zeros(N)
    fixed_fees = np.zeros(N)
    slippage = np.zeros(N)
    flat_conflict_code = np.full(N, 2, dtype=np.int8)

    # Warmup (JIT compile).
    for _ in range(3):
        simulate_futures_nb(
            close=close, long_entries=long_entries, long_exits=long_exits,
            short_entries=np.zeros_like(long_entries), short_exits=np.zeros_like(long_exits),
            size=size, mult=mult, margin_rate=margin_rate, fees=fees,
            fixed_fees=fixed_fees, slippage=slippage,
            flat_conflict_code=flat_conflict_code, init_cash=100_000.0,
        )

    # Time 100 iterations.
    n_iter = 100
    t0 = time.perf_counter()
    for _ in range(n_iter):
        simulate_futures_nb(
            close=close, long_entries=long_entries, long_exits=long_exits,
            short_entries=np.zeros_like(long_entries), short_exits=np.zeros_like(long_exits),
            size=size, mult=mult, margin_rate=margin_rate, fees=fees,
            fixed_fees=fixed_fees, slippage=slippage,
            flat_conflict_code=flat_conflict_code, init_cash=100_000.0,
        )
    elapsed = time.perf_counter() - t0
    per_call_ms = elapsed / n_iter * 1000.0
    print(f"Grid: {T} bars x {N} cols")
    print(f"Iters: {n_iter}")
    print(f"Per-call: {per_call_ms:.3f} ms")
    target = 5.0
    if per_call_ms < target:
        print(f"OK: under {target} ms target")
    else:
        print(f"WARN: exceeds {target} ms target")


if __name__ == "__main__":
    main()
