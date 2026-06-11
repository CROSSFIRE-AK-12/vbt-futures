"""Core futures-aware simulator: @njit main loop.

Re-uses vectorbt's data structures (none yet) but otherwise does not depend
on vbt at runtime.  Public function:

    simulate_futures_nb(close, long_entries, long_exits, short_entries,
                        short_exits, size, mult, margin_rate, fees,
                        fixed_fees, slippage, flat_conflict_code,
                        init_cash) -> (orders, cash, position, margin_locked)
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .records import FUTURES_ORDER_DT, make_empty_records


@njit(cache=True)
def simulate_futures_nb(
    close: np.ndarray,
    long_entries: np.ndarray,
    long_exits: np.ndarray,
    short_entries: np.ndarray,
    short_exits: np.ndarray,
    size: np.ndarray,
    mult: np.ndarray,
    margin_rate: np.ndarray,
    fees: np.ndarray,
    fixed_fees: np.ndarray,
    slippage: np.ndarray,
    flat_conflict_code: np.ndarray,
    init_cash: float,
):
    """Run a futures backtest.  Returns ``(order_records, cash, position,
    margin_locked)`` as four NumPy arrays.

    See spec §4.2 for the input contract and §5.2 for the per-bar algorithm.
    """
    T, N = close.shape
    # ---- per-call state (scalars for account, arrays per column) ----
    cash = init_cash
    position = np.zeros(N, dtype=np.float64)
    avg_price = np.zeros(N, dtype=np.float64)
    margin_locked = np.zeros(N, dtype=np.float64)
    prev_close = np.empty(N, dtype=np.float64)
    liquidated = np.zeros(N, dtype=np.bool_)
    # Initialise prev_close from the first bar.
    for col in range(N):
        prev_close[col] = close[0, col]

    # ---- pre-allocate output record buffer ----
    max_orders = 2 * T * N
    orders = make_empty_records(max_orders)
    order_idx = 0

    # ---- output time series ----
    out_cash = np.empty(T, dtype=np.float64)
    out_position = np.empty((T, N), dtype=np.float64)
    out_margin = np.empty((T, N), dtype=np.float64)

    for t in range(T):
        # ---------- STEP 1: mark-to-market ----------
        for col in range(N):
            if not liquidated[col] and position[col] != 0.0:
                cash += position[col] * (close[t, col] - prev_close[col]) * mult[col]

        # ---------- STEP 2: signals (placeholder, filled in by later tasks) ----------
        # The two-pass signal handling will be added in Tasks 9+.

        # ---------- STEP 3: dynamic margin recompute ----------
        for col in range(N):
            if position[col] != 0.0:
                new_margin = abs(position[col]) * close[t, col] * mult[col] * margin_rate[col]
                cash -= new_margin - margin_locked[col]
                margin_locked[col] = new_margin
            else:
                # Position is 0: any residual margin must already be 0.
                if margin_locked[col] != 0.0:  # pragma: no cover (defensive)
                    cash += margin_locked[col]
                    margin_locked[col] = 0.0

        # ---------- STEP 4: liquidation (added in Task 18) ----------
        # No-op until then.

        # ---------- STEP 5: snapshot ----------
        out_cash[t] = cash
        for col in range(N):
            out_position[t, col] = position[col]
            out_margin[t, col] = margin_locked[col]
            prev_close[col] = close[t, col]

    # Slice the record buffer to the actual filled count.
    if order_idx == 0:
        return orders[:0], out_cash, out_position, out_margin
    return orders[:order_idx], out_cash, out_position, out_margin
