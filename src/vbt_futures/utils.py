"""Utilities for vbt-futures.

Includes input validation and helpers for deriving annualisation factors
from a ``DatetimeIndex``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "infer_bars_per_year",
    "_validate_inputs",
]


def infer_bars_per_year(
    index: pd.DatetimeIndex,
    trading_days_per_year: int = 252,
) -> float:
    """Estimate the number of bars per year for annualisation.

    Counts the number of bars per calendar date and returns the **median**
    count multiplied by ``trading_days_per_year``.  The median is used (rather
    than the mean) so that half-day sessions or one-off cancellations do not
    bias the result.

    Args:
        index: Timestamped bar index.
        trading_days_per_year: Multiplier applied to the median bar count
            (252 for A-shares / domestic futures, 365 for crypto).

    Raises:
        ValueError: If ``trading_days_per_year`` is not positive.
    """
    if trading_days_per_year <= 0:
        raise ValueError(
            f"trading_days_per_year 必须 > 0, 收到 {trading_days_per_year}",
        )
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(
            "infer_bars_per_year 要求 pd.DatetimeIndex, 收到 "
            f"{type(index).__name__}",
        )
    if len(index) == 0:
        return 0.0
    dates = np.asarray(index.date)
    _, counts = np.unique(dates, return_counts=True)
    median_bpd = float(np.median(counts))
    return float(median_bpd * trading_days_per_year)


def _validate_inputs(
    close: pd.DataFrame,
    long_entries: pd.DataFrame | None,
    long_exits: pd.DataFrame | None,
    short_entries: pd.DataFrame | None,
    short_exits: pd.DataFrame | None,
    specs: list,
    size: float | np.ndarray | pd.DataFrame,
    init_cash: float,
    freq: str | pd.Timedelta | None,
    bars_per_year: float | None,
    trading_days_per_year: int,
) -> None:
    """Validate inputs to ``from_signals``.  Raises ``ValueError`` on any issue."""
    # ---- shape / column / dtypes ----
    if not isinstance(close, pd.DataFrame):
        raise ValueError("close 必须是 pd.DataFrame, 收到 " + type(close).__name__)
    if len(specs) != close.shape[1]:
        raise ValueError(
            f"len(specs)={len(specs)} 不匹配 close 的列数 {close.shape[1]}",
        )

    def _check_signal(name: str, sig: pd.DataFrame | None) -> None:
        if sig is None:
            return
        if not isinstance(sig, pd.DataFrame):
            raise ValueError(f"{name} 必须是 pd.DataFrame, 收到 {type(sig).__name__}")
        if sig.shape != close.shape:
            raise ValueError(
                f"{name} 形状 {sig.shape} 不匹配 close {close.shape}",
            )
        if sig.dtypes.iloc[0] != bool or (sig.dtypes != bool).any():
            raise ValueError(
                f"{name} 必须 bool 类型, 收到 {sig.dtypes.iloc[0]}",
            )

    _check_signal("long_entries", long_entries)
    _check_signal("long_exits", long_exits)
    _check_signal("short_entries", short_entries)
    _check_signal("short_exits", short_exits)

    # ---- specs ----
    for s in specs:
        if s.margin_rate <= 0:
            raise ValueError(f"FuturesSpec({s.symbol}).margin_rate 必须 > 0")
        if s.mult <= 0:
            raise ValueError(f"FuturesSpec({s.symbol}).mult 必须 > 0")
        if s.flat_conflict not in ("long", "short", "skip"):
            raise ValueError(
                f"FuturesSpec({s.symbol}).flat_conflict 必须是 "
                f"'long'|'short'|'skip', 收到 '{s.flat_conflict}'",
            )

    # ---- init_cash / size / close ----
    if init_cash <= 0:
        raise ValueError(f"init_cash 必须 > 0, 收到 {init_cash}")

    if np.any(np.asarray(size) <= 0):
        raise ValueError("size 含非正数; 方向由信号决定, size 必须 > 0")

    if np.any(close.values < 0):
        raise ValueError("close 含负值; 期货价格必须 > 0")

    # ---- freq / bars_per_year / trading_days_per_year ----
    if freq is not None:
        try:
            pd.Timedelta(freq)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"freq='{freq}' 不能解析为 pd.Timedelta") from exc

    if bars_per_year is not None and bars_per_year <= 0:
        raise ValueError(f"bars_per_year 必须 > 0, 收到 {bars_per_year}")

    if trading_days_per_year <= 0:
        raise ValueError(
            f"trading_days_per_year 必须 > 0, 收到 {trading_days_per_year}",
        )

    # ---- either user-supplied bars_per_year or auto-infer possible ----
    if bars_per_year is None and not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError(
            "无法推断 bars_per_year: index 不是 DatetimeIndex 且未传 bars_per_year; "
            "请显式传 bars_per_year",
        )
