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
    base_size: np.ndarray,
    long_entries: np.ndarray,
    long_exits: np.ndarray,
    short_entries: np.ndarray,
    short_exits: np.ndarray,
    close: np.ndarray,
    init_cash: float,
) -> np.ndarray:
    """Scale size by running equity curve (with mark-to-market).

    Forward-iteration that tracks SIGNED per-contract position:
      - On ``long_entries[t, col]``: position goes to +size.
      - On ``short_entries[t, col]``: position goes to -size.
      - On exits: position -> 0 and PnL is realised into cash.
    The ``scale`` is the current cash equity (no notional) / init_cash.

    The simulator uses ``size_arr[t, col].abs()`` for the trade size, so
    the sign here only affects the forward-iteration's PnL tracking.
    """
    T, N = base_size.shape
    out = np.zeros((T, N), dtype=np.float64)
    position = np.zeros(N, dtype=np.float64)  # SIGNED: +long, -short
    avg_price = np.zeros(N, dtype=np.float64)
    prev_close = np.empty(N, dtype=np.float64)
    for col in range(N):
        prev_close[col] = close[0, col]
    cash = init_cash
    for t in range(T):
        # 1. Mark-to-market held positions (PnL flows into cash).
        for col in range(N):
            if position[col] != 0.0:
                cash += position[col] * (close[t, col] - prev_close[col])
        # 2. Compute scale = cash / init_cash.
        if cash > 0.0:
            scale = cash / init_cash
        else:
            scale = 0.0
        # 3. Set size for entry bars.
        for col in range(N):
            if long_entries[t, col] and position[col] == 0.0:
                out[t, col] = max(base_size[t, col] * scale, 0.0)
            elif short_entries[t, col] and position[col] == 0.0:
                out[t, col] = max(base_size[t, col] * scale, 0.0)
        # 4. Process exits: close position, realise PnL into cash.
        for col in range(N):
            if (long_exits[t, col] or short_exits[t, col]) and position[col] != 0.0:
                cash += position[col] * (close[t, col] - avg_price[col])
                position[col] = 0.0
                avg_price[col] = 0.0
        # 5. Process entries: open position (signed).
        for col in range(N):
            if long_entries[t, col] and position[col] == 0.0:
                position[col] = out[t, col]
                avg_price[col] = close[t, col]
            elif short_entries[t, col] and position[col] == 0.0:
                position[col] = -out[t, col]
                avg_price[col] = close[t, col]
        for col in range(N):
            prev_close[col] = close[t, col]
    return out


def _size_anti_martingale(
    base_size: np.ndarray,
    long_entries: np.ndarray,
    long_exits: np.ndarray,
    short_entries: np.ndarray,
    short_exits: np.ndarray,
    _close: np.ndarray,
    _init_cash: float,
    trigger_pnl: float = 1_000.0,
    max_size: float = 5.0,
) -> np.ndarray:
    """Size grows by 1 unit per ``trigger_pnl`` of cumulative PnL, capped.

    Signed positions; same direction logic as ``_size_equity_proportional``.
    """
    T, N = base_size.shape
    out = base_size.copy()
    cum_pnl = 0.0
    position = np.zeros(N, dtype=np.float64)  # SIGNED
    avg_price = np.zeros(N, dtype=np.float64)
    prev_close = np.empty(N, dtype=np.float64)
    for col in range(N):
        prev_close[col] = _close[0, col]
    for t in range(T):
        # mtm
        for col in range(N):
            if position[col] != 0.0:
                cum_pnl += position[col] * (_close[t, col] - prev_close[col])
        # bonus
        if cum_pnl > 0.0:
            bonus = cum_pnl / max(trigger_pnl, 1e-9)
        else:
            bonus = 0.0
        # entries
        for col in range(N):
            if (long_entries[t, col] or short_entries[t, col]) and position[col] == 0.0:
                out[t, col] = min(base_size[t, col] + bonus, max_size)
        # exits
        for col in range(N):
            if (long_exits[t, col] or short_exits[t, col]) and position[col] != 0.0:
                position[col] = 0.0
                avg_price[col] = 0.0
        # entries (signed)
        for col in range(N):
            if long_entries[t, col] and position[col] == 0.0:
                position[col] = out[t, col]
                avg_price[col] = _close[t, col]
            elif short_entries[t, col] and position[col] == 0.0:
                position[col] = -out[t, col]
                avg_price[col] = _close[t, col]
        for col in range(N):
            prev_close[col] = _close[t, col]
    return out


def resolve_size(
    mode: str,
    base_size: float | np.ndarray | pd.DataFrame,
    long_entries: np.ndarray,
    long_exits: np.ndarray,
    short_entries: np.ndarray,
    short_exits: np.ndarray,
    close: np.ndarray,
    init_cash: float,
    sizing_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute the (T, N) size array for the given mode."""
    T, N = long_entries.shape
    base = _broadcast_size(base_size, T, N)
    if long_exits is None:
        long_exits = np.zeros_like(long_entries)
    if short_exits is None:
        short_exits = np.zeros_like(short_entries)
    entry_mask = long_entries | short_entries
    if mode == SIZE_FIXED:
        return _size_fixed(base, entry_mask, close, init_cash)
    if mode == SIZE_EQUITY_PROPORTIONAL:
        return _size_equity_proportional(
            base, long_entries, long_exits, short_entries, short_exits,
            close, init_cash,
        )
    if mode == SIZE_ANTI_MARTINGALE:
        kwargs = sizing_kwargs or {}
        return _size_anti_martingale(
            base, long_entries, long_exits, short_entries, short_exits,
            close, init_cash,
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


def _exit_signal_mask(
    long_exits: np.ndarray | None,
    short_exits: np.ndarray | None,
) -> np.ndarray:
    """OR together long and short exit signals (any exit = True)."""
    if long_exits is None and short_exits is None:
        return None  # type: ignore[return-value]
    T, N = (long_exits if long_exits is not None else short_exits).shape
    out = np.zeros((T, N), dtype=bool)
    if long_exits is not None:
        out |= long_exits
    if short_exits is not None:
        out |= short_exits
    return out
