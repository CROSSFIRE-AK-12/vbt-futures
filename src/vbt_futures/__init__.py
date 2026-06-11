"""vbt-futures: backtrader-style futures backtesting on vectorbt's numba framework.

Public API:
    from_signals(...)  - main entry point
    FuturesSpec        - per-contract specification
    FuturesPortfolio   - post-simulation view
"""
from __future__ import annotations

from .portfolio import FuturesPortfolio
from .spec import FuturesSpec

__version__ = "0.1.0"
__all__ = ["from_signals", "FuturesSpec", "FuturesPortfolio"]


def from_signals(
    close,
    *,
    long_entries=None,
    long_exits=None,
    short_entries=None,
    short_exits=None,
    specs=None,
    size=1.0,
    init_cash=100_000.0,
    freq=None,
    bars_per_year=None,
    trading_days_per_year=252,
    margin_mode="gross",
    margin_lookback=60,
    margin_z_score=1.65,
    sizing_mode="fixed",
    sizing_kwargs=None,
):
    """Run a futures backtest from a DataFrame of close prices and boolean
    entry/exit signal DataFrames.

    See ``docs/superpowers/specs/2026-06-11-vbt-futures-design.md`` §6.1 for
    the full parameter contract.
    """
    # Lazy imports to avoid a circular dependency between portfolio/simulator
    # and this module.
    from . import utils
    from .enums import FLAT_CONFLICT_CODE
    from .simulator import simulate_futures_nb
    from .sizing import (
        SIZE_ANTI_MARTINGALE,
        SIZE_EQUITY_PROPORTIONAL,
        SIZE_FIXED,
        _entry_signal_mask,
        resolve_size,
    )

    # ---- validate ----
    utils._validate_inputs(
        close, long_entries, long_exits, short_entries, short_exits,
        specs, size, init_cash, freq, bars_per_year, trading_days_per_year,
    )
    if margin_mode not in ("gross", "portfolio"):
        raise ValueError(
            f"margin_mode 必须是 'gross' | 'portfolio', 收到 '{margin_mode}'"
        )
    if not isinstance(margin_lookback, int) or margin_lookback < 2:
        raise ValueError(
            f"margin_lookback 必须是 >= 2 的整数, 收到 {margin_lookback}"
        )
    if sizing_mode not in (SIZE_FIXED, SIZE_EQUITY_PROPORTIONAL, SIZE_ANTI_MARTINGALE):
        raise ValueError(
            f"sizing_mode 必须是 {sorted([SIZE_FIXED, SIZE_EQUITY_PROPORTIONAL, SIZE_ANTI_MARTINGALE])}, "
            f"收到 '{sizing_mode}'"
        )

    T, N = close.shape

    # ---- helpers for array conversion ----
    def _to_bool(df, name):
        if df is None:
            return np.zeros((T, N), dtype=bool)
        return df.values

    def _to_size_array(s):
        if np.isscalar(s):
            return np.full((T, N), float(s), dtype=np.float64)
        arr = np.asarray(s, dtype=np.float64)
        if arr.ndim == 1:
            arr = np.broadcast_to(arr, (T, N)).copy()
        return arr

    # ---- broadcast signals to numpy ----
    le = _to_bool(long_entries, "long_entries")
    lx = _to_bool(long_exits, "long_exits")
    se = _to_bool(short_entries, "short_exits")
    sx = _to_bool(short_exits, "short_exits")

    # ---- resolve size (per sizing mode) ----
    if sizing_mode == SIZE_FIXED:
        size_arr = _to_size_array(size)
    else:
        size_arr = resolve_size(
            sizing_mode, size,
            le, lx, se, sx,
            close.values.astype(np.float64), float(init_cash), sizing_kwargs,
        )

    # ---- split specs into per-column arrays ----
    mult = np.array([s.mult for s in specs], dtype=np.float64)
    margin_rate = np.array([s.margin_rate for s in specs], dtype=np.float64)
    fees = np.array([s.fees for s in specs], dtype=np.float64)
    fixed_fees = np.array([s.fixed_fees for s in specs], dtype=np.float64)
    slippage = np.array([s.slippage for s in specs], dtype=np.float64)
    flat_conflict_code = np.array(
        [FLAT_CONFLICT_CODE[s.flat_conflict] for s in specs],
        dtype=np.int8,
    )

    # ---- determine bars_per_year ----
    if bars_per_year is None:
        if isinstance(close.index, pd.DatetimeIndex):
            bars_per_year = utils.infer_bars_per_year(close.index, trading_days_per_year)
        else:  # pragma: no cover (already validated above)
            raise ValueError("bars_per_year 推断失败: 非 DatetimeIndex")
    bars_per_year = float(bars_per_year)

    # ---- run simulator ----
    orders, cash, position, margin_locked = simulate_futures_nb(
        close=close.values.astype(np.float64),
        long_entries=le, long_exits=lx, short_entries=se, short_exits=sx,
        size=size_arr,
        mult=mult, margin_rate=margin_rate, fees=fees,
        fixed_fees=fixed_fees, slippage=slippage,
        flat_conflict_code=flat_conflict_code,
        init_cash=float(init_cash),
        margin_mode=0 if margin_mode == "gross" else 1,
        margin_lookback=margin_lookback,
        z_score=margin_z_score,
    )

    # ---- wrap in portfolio ----
    return FuturesPortfolio(
        close=close,
        specs=tuple(specs),
        init_cash=float(init_cash),
        freq=freq,
        bars_per_year=bars_per_year,
        trading_days_per_year=trading_days_per_year,
        _order_records=orders,
        _cash=cash,
        _position=position,
        _margin_locked=margin_locked,
        _entry_mask_long=le,
        _entry_mask_long_exit=lx,
        _entry_mask_short=se,
        _entry_mask_short_exit=sx,
    )


import numpy as np
import pandas as pd  # noqa: E402  (used by callers; keep import here for linting)
