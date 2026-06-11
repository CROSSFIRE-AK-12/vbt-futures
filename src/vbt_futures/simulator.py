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

from .enums import (
    CLOSE_LONG,
    CLOSE_SHORT,
    FLAT_CONFLICT_CODE,
    LIQUIDATED,
    OPEN_LONG,
    OPEN_SHORT,
)
from .records import FUTURES_ORDER_DT, make_empty_records

# FLAT_CONFLICT_CODE is exposed publicly; declare alias for the simulator
# to keep the njit body terse.
_FLAT_LONG = FLAT_CONFLICT_CODE["long"]
_FLAT_SHORT = FLAT_CONFLICT_CODE["short"]
_FLAT_SKIP = FLAT_CONFLICT_CODE["skip"]


@njit(cache=True)
def _do_close(
    col: int,
    price: float,
    bar_idx: int,
    cash: float,
    position: np.ndarray,
    avg_price: np.ndarray,
    margin_locked: np.ndarray,
    mult: np.ndarray,
    fees: np.ndarray,
    fixed_fees: np.ndarray,
    slippage: np.ndarray,
    orders: np.ndarray,
    order_idx: int,
    side_override: int = -1,
) -> tuple[float, int]:
    """Close any open position on column col.  Returns (new_cash, new_order_idx).

    If ``side_override`` is given (e.g. ``LIQUIDATED``), that side code is
    written to the record; otherwise it is inferred from the sign of the
    current position.
    """
    pos = position[col]
    s = 1.0 if pos > 0.0 else -1.0
    adj_price = price * (1.0 - s * slippage[col])
    size_abs = abs(pos)
    notional = size_abs * adj_price * mult[col]
    fee_paid = notional * fees[col] + size_abs * fixed_fees[col]
    realized = pos * (adj_price - avg_price[col]) * mult[col]
    released_margin = margin_locked[col]

    new_cash = cash + released_margin + realized - fee_paid

    if side_override == LIQUIDATED:
        side_out = LIQUIDATED
    elif pos > 0.0:
        side_out = CLOSE_LONG
    else:
        side_out = CLOSE_SHORT

    rec = orders[order_idx]
    rec["id"] = order_idx
    rec["col"] = col
    rec["idx"] = bar_idx
    rec["size"] = -pos
    rec["price"] = adj_price
    rec["fees"] = fee_paid
    rec["margin"] = -released_margin
    rec["side"] = side_out
    rec["pnl"] = realized - fee_paid

    cash = new_cash
    position[col] = 0.0
    avg_price[col] = 0.0
    margin_locked[col] = 0.0
    order_idx += 1
    return cash, order_idx


@njit(cache=True)
def _try_open(
    col: int,
    signed_size: float,
    price: float,
    bar_idx: int,
    cash: float,
    position: np.ndarray,
    avg_price: np.ndarray,
    margin_locked: np.ndarray,
    mult: np.ndarray,
    margin_rate: np.ndarray,
    fees: np.ndarray,
    fixed_fees: np.ndarray,
    slippage: np.ndarray,
    orders: np.ndarray,
    order_idx: int,
) -> tuple[float, int]:
    """Try to open a position.  Returns ``(new_cash, new_order_idx)``.

    On rejection (insufficient cash) returns the cash unchanged and does not
    write a record.
    """
    s = 1.0 if signed_size > 0.0 else -1.0
    adj_price = price * (1.0 + s * slippage[col])
    notional = abs(signed_size) * adj_price * mult[col]
    req_margin = notional * margin_rate[col]
    req_fee = notional * fees[col] + abs(signed_size) * fixed_fees[col]

    if cash < req_margin + req_fee:
        return cash, order_idx

    new_cash = cash - req_margin - req_fee
    new_margin_locked = margin_locked[col] + req_margin
    new_position = position[col] + signed_size
    if position[col] == 0.0:
        new_avg_price = adj_price
    else:  # pragma: no cover (reversal closes first)
        new_avg_price = avg_price[col]

    side = OPEN_LONG if signed_size > 0.0 else OPEN_SHORT
    rec = orders[order_idx]
    rec["id"] = order_idx
    rec["col"] = col
    rec["idx"] = bar_idx
    rec["size"] = signed_size
    rec["price"] = adj_price
    rec["fees"] = req_fee
    rec["margin"] = req_margin
    rec["side"] = side
    rec["pnl"] = 0.0

    cash = new_cash
    position[col] = new_position
    avg_price[col] = new_avg_price
    margin_locked[col] = new_margin_locked
    order_idx += 1
    return cash, order_idx


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
        # ---------- STEP 0: NaN guard ----------
        # If a column's close at this bar is NaN, skip all per-column work
        # for this bar (keep prior state, no orders, no margin diff).
        for col in range(N):
            if np.isnan(close[t, col]):
                liquidated[col] = True  # mark as "frozen" - no further activity

        # ---------- STEP 1: mark-to-market ----------
        for col in range(N):
            if not liquidated[col] and position[col] != 0.0:
                cash += position[col] * (close[t, col] - prev_close[col]) * mult[col]

        # ---------- STEP 2: signals (two-pass evaluation) ----------
        for col in range(N):
            if liquidated[col]:
                continue
            size_at_t = size[t, col]

            # --- PASS 1: handle existing position (exit / reversal) ---
            # Only REVERSAL (close + open opposite side) skips PASS 2.
            # Plain exit falls through to PASS 2 to allow same-bar re-entry.
            skip_pass2 = False
            if position[col] > 0.0:
                # Long held.
                if short_entries[t, col]:
                    # Reversal: close long + open short.
                    cash, order_idx = _do_close(
                        col, close[t, col], t, cash, position, avg_price,
                        margin_locked, mult, fees, fixed_fees, slippage,
                        orders, order_idx,
                    )
                    cash, order_idx = _try_open(
                        col, -size_at_t, close[t, col], t, cash, position,
                        avg_price, margin_locked, mult, margin_rate, fees,
                        fixed_fees, slippage, orders, order_idx,
                    )
                    skip_pass2 = True
                elif long_exits[t, col]:
                    cash, order_idx = _do_close(
                        col, close[t, col], t, cash, position, avg_price,
                        margin_locked, mult, fees, fixed_fees, slippage,
                        orders, order_idx,
                    )
                    # fall through to PASS 2 for same-bar re-entry
            elif position[col] < 0.0:
                # Short held.
                if long_entries[t, col]:
                    # Reversal: close short + open long.
                    cash, order_idx = _do_close(
                        col, close[t, col], t, cash, position, avg_price,
                        margin_locked, mult, fees, fixed_fees, slippage,
                        orders, order_idx,
                    )
                    cash, order_idx = _try_open(
                        col, size_at_t, close[t, col], t, cash, position,
                        avg_price, margin_locked, mult, margin_rate, fees,
                        fixed_fees, slippage, orders, order_idx,
                    )
                    skip_pass2 = True
                elif short_exits[t, col]:
                    cash, order_idx = _do_close(
                        col, close[t, col], t, cash, position, avg_price,
                        margin_locked, mult, fees, fixed_fees, slippage,
                        orders, order_idx,
                    )
                    # fall through to PASS 2 for same-bar re-entry
            if skip_pass2:
                continue

            # --- PASS 2: handle entry from flat ---
            if position[col] != 0.0:
                continue
            le = long_entries[t, col]
            se = short_entries[t, col]
            if le and se:
                fc = flat_conflict_code[col]
                if fc == _FLAT_LONG:
                    cash, order_idx = _try_open(
                        col, size_at_t, close[t, col], t, cash, position,
                        avg_price, margin_locked, mult, margin_rate, fees,
                        fixed_fees, slippage, orders, order_idx,
                    )
                elif fc == _FLAT_SHORT:
                    cash, order_idx = _try_open(
                        col, -size_at_t, close[t, col], t, cash, position,
                        avg_price, margin_locked, mult, margin_rate, fees,
                        fixed_fees, slippage, orders, order_idx,
                    )
                # _FLAT_SKIP => no-op
            elif le:
                cash, order_idx = _try_open(
                    col, size_at_t, close[t, col], t, cash, position,
                    avg_price, margin_locked, mult, margin_rate, fees,
                    fixed_fees, slippage, orders, order_idx,
                )
            elif se:
                cash, order_idx = _try_open(
                    col, -size_at_t, close[t, col], t, cash, position,
                    avg_price, margin_locked, mult, margin_rate, fees,
                    fixed_fees, slippage, orders, order_idx,
                )

        # ---------- STEP 3: dynamic margin recompute ----------
        for col in range(N):
            if liquidated[col]:
                continue
            if position[col] != 0.0:
                new_margin = abs(position[col]) * close[t, col] * mult[col] * margin_rate[col]
                cash -= new_margin - margin_locked[col]
                margin_locked[col] = new_margin
            else:
                # Position is 0: any residual margin must already be 0.
                if margin_locked[col] != 0.0:  # pragma: no cover (defensive)
                    cash += margin_locked[col]
                    margin_locked[col] = 0.0

        # ---------- STEP 4: liquidation ----------
        # If total equity is <= 0, close all open positions and mark them
        # as liquidated so no further activity is allowed.
        total_equity = cash
        for col in range(N):
            total_equity += margin_locked[col]
        if total_equity <= 0.0:
            for col in range(N):
                if position[col] != 0.0 and not liquidated[col]:
                    cash, order_idx = _do_close(
                        col, close[t, col], t, cash, position, avg_price,
                        margin_locked, mult, fees, fixed_fees, slippage,
                        orders, order_idx, side_override=LIQUIDATED,
                    )
                    liquidated[col] = True

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
