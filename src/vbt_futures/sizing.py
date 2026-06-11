"""Sizing modes for vbt-futures.

Each sizing mode takes the entry signal DataFrame and returns a (T, N) size
array.  The simulator itself does not change; this layer just pre-computes
the per-bar per-contract size before the simulator is invoked.

Modes:
    - fixed              : same as the ``size`` param (default)
    - equity_proportional: size scales linearly with running equity
    - anti_martingale     : size grows as cum-PnL grows
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Mode names (string constants used by from_signals()).
SIZE_FIXED: str = "fixed"
SIZE_EQUITY_PROPORTIONAL: str = "equity_proportional"
SIZE_ANTI_MARTINGALE: str = "anti_martingale"

_VALID_MODES: tuple[str, ...] = (
    SIZE_FIXED, SIZE_EQUITY_PROPORTIONAL, SIZE_ANTI_MARTINGALE,
)


def _broadcast_size(
    size: float | np.ndarray | pd.DataFrame, T: int, N: int,
) -> np.ndarray:
    """Convert any of scalar/1D/2D size into a (T, N) float64 array."""
    if np.isscalar(size):
        return np.full((T, N), float(size), dtype=np.float64)
    arr = np.asarray(size, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != N:
            raise ValueError(
                f"size 1D array must have length {N} (cols), got {arr.shape[0]}"
            )
        return np.broadcast_to(arr, (T, N)).copy()
    if arr.ndim == 2:
        if arr.shape != (T, N):
            raise ValueError(
                f"size 2D array must be shape {(T, N)}, got {arr.shape}"
            )
        return arr.copy()
    raise ValueError(f"size must be scalar, 1D, or 2D; got ndim={arr.ndim}")


def _size_fixed(
    base_size: np.ndarray, _signals: np.ndarray, _close: np.ndarray, _init_cash: float,
) -> np.ndarray:
    """Return the base_size as-is (1 entry per signal)."""
    return base_size


def _size_equity_proportional(
    base_size: np.ndarray, signals: np.ndarray, close: np.ndarray, init_cash: float,
) -> np.ndarray:
    """Scale size by running equity curve (with mark-to-market).

    We run a forward pass that tracks per-contract equity contributions.
    On any non-signal bar, size is 0 (no position open).  On signal bars,
    size = base_size * (equity_so_far / init_equity), floored at 0.
    """
    T, N = base_size.shape
    out = np.zeros((T, N), dtype=np.float64)
    position = np.zeros(N, dtype=np.float64)
    avg_price = np.zeros(N, dtype=np.float64)
    prev_close = np.empty(N, dtype=np.float64)
    for col in range(N):
        prev_close[col] = close[0, col]
    cash = init_cash
    cum_equity = init_cash
    # long_entries, long_exits, etc.  We only need 'entries' = long|short entries
    # = signals[t, col] > 0.  The caller passes a 2D array where 1 = any entry.
    for t in range(T):
        # Update equity via mark-to-market before deciding size.
        for col in range(N):
            if position[col] != 0.0:
                cash += position[col] * (close[t, col] - prev_close[col])
        cum_equity = cash
        for col in range(N):
            cum_equity += position[col] * close[t, col]  # notional value
        # Scale by equity.
        if cum_equity > 0.0:
            scale = cum_equity / init_cash
        else:
            scale = 0.0
        # Apply: where there is an entry signal, set size = base * scale.
        # `signals` is bool: True on entry bars.
        for col in range(N):
            if signals[t, col] and position[col] == 0.0:
                out[t, col] = max(base_size[t, col] * scale, 0.0)
        # Crude position tracking: assume signal[t] opens at close[t]
        for col in range(N):
            if signals[t, col] and position[col] == 0.0:
                position[col] = base_size[t, col]
                avg_price[col] = close[t, col]
            elif (not signals[t, col]) and position[col] != 0.0:
                # Naive: any non-entry bar closes.  Caller controls via their
                # own signal logic; we just need a forward estimate of equity.
                cash += position[col] * (close[t, col] - avg_price[col])
                position[col] = 0.0
                avg_price[col] = 0.0
        for col in range(N):
            prev_close[col] = close[t, col]
    return out


def _size_anti_martingale(
    base_size: np.ndarray, signals: np.ndarray, _close: np.ndarray, _init_cash: float,
    trigger_pnl: float = 1_000.0,
    max_size: float = 5.0,
) -> np.ndarray:
    """Size grows by 1 unit per `trigger_pnl` of cumulative PnL, capped at `max_size`.

    Negative PnL does NOT decrease size (anti-martingale is the inverse of
    martingale).  The base_size is the minimum; we add 0.0 *|cum_pnl|/trigger
    of bonus (with size = base_size + bonus) when cum_pnl > 0.
    """
    T, N = base_size.shape
    out = base_size.copy()
    cum_pnl = 0.0
    # Approximate PnL via a simple equity tracker.
    position = np.zeros(N, dtype=np.float64)
    avg_price = np.zeros(N, dtype=np.float64)
    prev_close = np.empty(N, dtype=np.float64)
    for col in range(N):
        prev_close[col] = _close[0, col]
    for t in range(T):
        # mtM
        for col in range(N):
            if position[col] != 0.0:
                cum_pnl += position[col] * (_close[t, col] - prev_close[col])
        # Decide size on entry bars based on cum_pnl.
        if cum_pnl > 0.0:
            bonus = cum_pnl / max(trigger_pnl, 1e-9)
        else:
            bonus = 0.0
        for col in range(N):
            if signals[t, col] and position[col] == 0.0:
                out[t, col] = min(base_size[t, col] + bonus, max_size)
        # Naive position tracking.
        for col in range(N):
            if signals[t, col] and position[col] == 0.0:
                position[col] = out[t, col]
                avg_price[col] = _close[t, col]
            elif (not signals[t, col]) and position[col] != 0.0:
                position[col] = 0.0
                avg_price[col] = 0.0
        for col in range(N):
            prev_close[col] = _close[t, col]
    return out


def resolve_size(
    mode: str,
    base_size: float | np.ndarray | pd.DataFrame,
    signals: np.ndarray,
    close: np.ndarray,
    init_cash: float,
    sizing_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute the (T, N) size array for the given mode.

    ``base_size`` is the user-supplied input (scalar / 1D / 2D).
    ``signals`` is a bool (T, N) array — True on any entry bar (long or short).
    """
    T, N = signals.shape
    base = _broadcast_size(base_size, T, N)
    if mode == SIZE_FIXED:
        return _size_fixed(base, signals, close, init_cash)
    if mode == SIZE_EQUITY_PROPORTIONAL:
        return _size_equity_proportional(base, signals, close, init_cash)
    if mode == SIZE_ANTI_MARTINGALE:
        kwargs = sizing_kwargs or {}
        return _size_anti_martingale(
            base, signals, close, init_cash,
            trigger_pnl=float(kwargs.get("trigger_pnl", 1_000.0)),
            max_size=float(kwargs.get("max_size", 5.0)),
        )
    raise ValueError(
        f"sizing_mode 必须是 {sorted(_VALID_MODES)}, 收到 '{mode}'"
    )


def _entry_signal_mask(
    long_entries: np.ndarray | None,
    short_entries: np.ndarray | None,
) -> np.ndarray:
    """OR together long and short entry signals (any entry = True)."""
    T, N = (long_entries if long_entries is not None else short_entries).shape
    out = np.zeros((T, N), dtype=bool)
    if long_entries is not None:
        out |= long_entries
    if short_entries is not None:
        out |= short_entries
    return out
