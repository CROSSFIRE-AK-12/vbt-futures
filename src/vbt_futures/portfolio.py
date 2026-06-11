"""Public-facing wrapper around the simulator output."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from .records import FUTURES_ORDER_DT
from .spec import FuturesSpec


@dataclass(frozen=True)
class FuturesPortfolio:
    """Immutable post-simulation view of one backtest run."""

    close: pd.DataFrame
    specs: tuple[FuturesSpec, ...]
    init_cash: float
    freq: str | pd.Timedelta | None
    bars_per_year: float
    trading_days_per_year: int
    _order_records: np.ndarray
    _cash: np.ndarray
    _position: np.ndarray
    _margin_locked: np.ndarray

    # ---------- derived scalars (cached) ----------
    @cached_property
    def cash(self) -> pd.Series:
        """Available cash at the end of each bar (excludes locked margin)."""
        return pd.Series(self._cash, index=self.close.index, name="cash")

    @cached_property
    def position(self) -> pd.DataFrame:
        """Net position per contract at the end of each bar (+N long, -N short)."""
        return pd.DataFrame(
            self._position, index=self.close.index, columns=self.close.columns,
        )

    @cached_property
    def margin_locked(self) -> pd.DataFrame:
        """Locked margin per contract at the end of each bar."""
        return pd.DataFrame(
            self._margin_locked, index=self.close.index, columns=self.close.columns,
        )

    @cached_property
    def equity(self) -> pd.Series:
        """Total account equity = cash + sum(margin_locked)."""
        eq = self._cash + self._margin_locked.sum(axis=1)
        return pd.Series(eq, index=self.close.index, name="equity")

    @cached_property
    def returns(self) -> pd.Series:
        """Bar-to-bar returns of equity (first bar is 0.0)."""
        prev = np.concatenate(([self._cash[0] + self._margin_locked[0].sum()],
                                self._cash + self._margin_locked.sum(axis=1)))[:-1]
        eq = self._cash + self._margin_locked.sum(axis=1)
        r = np.zeros_like(eq)
        nonzero = prev != 0.0
        r[nonzero] = (eq[nonzero] - prev[nonzero]) / prev[nonzero]
        return pd.Series(r, index=self.close.index, name="returns")

    @cached_property
    def drawdown(self) -> pd.Series:
        """Running drawdown as a non-positive fraction (0 at first bar)."""
        eq = self._cash + self._margin_locked.sum(axis=1)
        cummax = np.maximum.accumulate(eq)
        dd = np.where(cummax > 0.0, eq / cummax - 1.0, 0.0)
        return pd.Series(dd, index=self.close.index, name="drawdown")
